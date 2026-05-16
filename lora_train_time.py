import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    project_root = SCRIPT_DIR.parents[0]
    default_model_path = project_root / "Model" / "Qwen3-VL-4B-Instruct"
    default_data_root = project_root / "dataset" / "EgoCross_support_set"
    default_train_json = default_data_root / "train.json"
    default_output_dir = SCRIPT_DIR / "lora_output" / "qwen3vl_time_lora"

    parser = argparse.ArgumentParser(description="Time-prompt LoRA fine-tuning for EgoCross Qwen3-VL")
    parser.add_argument("--model-path", type=str, default=str(default_model_path))
    parser.add_argument("--train-json", type=str, default=str(default_train_json))
    parser.add_argument("--data-root", type=str, default=str(default_data_root))
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir))
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--image-max-side", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--default-fps", type=float, default=0.5)
    parser.add_argument("--surgery-fps", type=float, default=1.0)
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--report-to", type=str, default="none")
    return parser.parse_args()


def keep_output_inside_dudu(output_arg: str) -> Path:
    output_dir = Path(output_arg)
    if not output_dir.is_absolute():
        return SCRIPT_DIR / output_dir

    resolved_output = output_dir.resolve()
    try:
        resolved_output.relative_to(SCRIPT_DIR)
        return resolved_output
    except ValueError:
        safe_output = SCRIPT_DIR / resolved_output.name
        print(f"Output dir is outside dudu; redirecting to: {safe_output}")
        return safe_output


def sample_frames(frame_paths: List[str], max_frames: int) -> List[str]:
    if len(frame_paths) <= max_frames:
        return frame_paths
    if max_frames <= 1:
        return [frame_paths[0]]

    indices = [round(i * (len(frame_paths) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [frame_paths[i] for i in indices]


def parse_question_and_options(user_content: str) -> tuple[str, List[str]]:
    text = user_content.replace("<image>", "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", []

    question = lines[0]
    options = [line for line in lines[1:] if re.match(r"^[A-D][\.:)]\s*", line, re.IGNORECASE)]
    if not options:
        options = lines[1:]
    return question, options


def build_prompt(question_text: str, options: List[str], fps: float) -> str:
    option_text = "\n".join(options)
    return (
        "You are solving a multiple-choice visual question from egocentric video frames.\n"
        "Only answer with one uppercase letter: A, B, C, or D.\n"
        "Do not output any extra words.\n\n"
        f" Sampling rate: {fps} frames per second (FPS)\n"
        f"Question: {question_text}\n"
        f"Options:\n{option_text}"
    )


def get_sample_fps(sample: dict[str, Any], default_fps: float, surgery_fps: float) -> float:
    domain = str(sample.get("domain", ""))
    image_paths = sample.get("images", [])
    first_path = image_paths[0] if image_paths else ""

    if domain == "surgery":
        return surgery_fps
    if "EgoSurgery" in first_path:
        return 1.0
    if "CholecTrack20" in first_path and ("VID25" in first_path or "VID111" in first_path):
        return 1.0
    return default_fps


def load_images(image_paths: Iterable[Path], image_max_side: int) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in image_paths:
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        image.thumbnail((image_max_side, image_max_side))
        images.append(image)
    return images


def close_images(images: Iterable[Image.Image]) -> None:
    for image in images:
        image.close()


class EgoCrossTimeLoraDataset(Dataset):
    def __init__(
        self,
        train_json: Path,
        data_root: Path,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        max_frames: int,
        image_max_side: int,
        max_length: int,
        default_fps: float,
        surgery_fps: float,
    ):
        with train_json.open("r", encoding="utf-8") as f:
            samples = json.load(f)

        self.samples = [sample for sample in samples if self._is_valid(sample)]
        self.data_root = data_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.image_max_side = image_max_side
        self.max_length = max_length
        self.default_fps = default_fps
        self.surgery_fps = surgery_fps

        if not self.samples:
            raise ValueError(f"No valid samples found in {train_json}")

    @staticmethod
    def _is_valid(sample: dict[str, Any]) -> bool:
        messages = sample.get("messages", [])
        return len(messages) >= 2 and bool(sample.get("images"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        user_content = sample["messages"][0]["content"]
        answer = str(sample["messages"][1]["content"]).strip().upper()
        question, options = parse_question_and_options(user_content)

        selected_images = sample_frames(sample["images"], self.max_frames)
        image_paths = [self.data_root / image_path.lstrip("/") for image_path in selected_images]
        images = load_images(image_paths, self.image_max_side)
        if not images:
            raise RuntimeError(f"No readable images for sample index {idx}")

        fps = get_sample_fps(sample, self.default_fps, self.surgery_fps)
        interval = 1.0 / fps
        content: list[dict[str, Any]] = []
        for frame_index, image in enumerate(images):
            content.append(
                {
                    "type": "text",
                    "text": f"<time_start> {frame_index * interval:.1f} <time_end>",
                }
            )
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": build_prompt(question, options, fps)})

        messages = [{"role": "user", "content": content}]

        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
                padding_side="left",
            )
            inputs = {key: value.tolist() for key, value in inputs.items()}

            response = self.tokenizer(answer, add_special_tokens=False)
            eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
            input_ids = inputs["input_ids"][0] + response["input_ids"] + [eos_token_id]
            attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
            labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [eos_token_id]

            if len(input_ids) > self.max_length:
                raise RuntimeError(
                    f"Sample index {idx} token length {len(input_ids)} exceeds max_length={self.max_length}. "
                    "Lower --max-frames or --image-max-side."
                )

            image_grid_thw = torch.tensor(inputs["image_grid_thw"])
            if image_grid_thw.ndim == 3:
                image_grid_thw = image_grid_thw.squeeze(0)
            if image_grid_thw.ndim == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)

            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "pixel_values": torch.tensor(inputs["pixel_values"]),
                "image_grid_thw": image_grid_thw,
            }
        finally:
            close_images(images)


def build_model(args: argparse.Namespace):
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )

    model.config.use_cache = False
    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[module.strip() for module in args.target_modules.split(",") if module.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    train_json = Path(args.train_json)
    data_root = Path(args.data_root)
    output_dir = keep_output_inside_dudu(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)

    dataset = EgoCrossTimeLoraDataset(
        train_json=train_json,
        data_root=data_root,
        processor=processor,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        image_max_side=args.image_max_side,
        max_length=args.max_length,
        default_fps=args.default_fps,
        surgery_fps=args.surgery_fps,
    )
    print(f"Loaded {len(dataset)} training samples from {train_json}")

    model = build_model(args)

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to=args.report_to,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved LoRA adapter to: {output_dir}")


if __name__ == "__main__":
    main()
