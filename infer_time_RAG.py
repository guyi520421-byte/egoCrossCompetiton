import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# conda activate egoVqa2

CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_model_path = project_root / "Model" / "Qwen3-VL-4B-Instruct"
    # default_lora_adapter="/root/data/wy/test/now/vqa/dudu/lora_output/qwen3vl_time_lora_15f_768"
    default_testData_json = project_root / "dataset" / "egocross_testbed" / "egocross_testbed_imgs.json"
    default_submit_json = project_root / "dataset" / "egocross_testbed" / "submission_sample.json"
    default_data_root = project_root / "dataset"
    default_rag_root = project_root / "dataset" / "cholecT50_2" #"cholecT50_part"
    default_output = Path(__file__).resolve().parent / "qwen3vl_fps_rag_onlySurg510.json"

    parser = argparse.ArgumentParser(description="Qwen3-VL-4B-Instruct + CholecT50 RAG for EgoCross testbed")
    parser.add_argument("--model-path", type=str, default=str(default_model_path), help="Local model path or HF repo id")
    # parser.add_argument("--lora-adapter", type=str, default=str(default_lora_adapter), help="LoRA adapter directory")
    parser.add_argument("--testData-json", type=str, default=str(default_testData_json), help="Path to egocross_testbed_imgs.json")
    parser.add_argument("--submission-sample", type=str, default=str(default_submit_json), help="Path to submission_sample.json")
    parser.add_argument("--data-root", type=str, default=str(default_data_root), help="Dataset root directory containing egocross_testbed")
    parser.add_argument("--rag-root", type=str, default=str(default_rag_root), help="Path to CholecT50 reference images and annotations")
    parser.add_argument("--disable-rag", action="store_true", help="Disable CholecT50 annotation RAG")
    parser.add_argument("--rag-top-k", type=int, default=6, help="Number of retrieved CholecT50 annotation summaries")
    parser.add_argument("--rag-visual-examples", type=int, default=2, help="Number of retrieved CholecT50 reference images to prepend")
    parser.add_argument("--output", type=str, default=str(default_output), help="Output prediction json path")
    parser.add_argument("--max-frames", type=int, default=12, help="Maximum number of frames per sample")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.5)
    return parser.parse_args()


def sample_frames(frame_paths: List[str], max_frames: int):
    idxs = [i for i in range(len(frame_paths))]
    if len(frame_paths) <= max_frames:
        return frame_paths, idxs

    idxs = [round(i * (len(frame_paths) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [frame_paths[i] for i in idxs], idxs


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("_", " ").replace("-", " ")).strip()


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", normalize_text(text)))


def bbox_to_region(x: float, y: float, w: float, h: float) -> str:
    cx = min(max(x + w / 2, 0.0), 1.0)
    cy = min(max(y + h / 2, 0.0), 1.0)
    horiz = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    vert = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "center"
    return "center" if horiz == "center" and vert == "center" else f"{vert}-{horiz}"


def category_lookup(categories: Dict[str, Dict[str, str]], name: str, value: Any) -> str:
    try:
        key = str(int(value))
    except (TypeError, ValueError):
        return ""
    return categories.get(name, {}).get(key, "")


def build_cholec_rag_index(rag_root: Path) -> Dict[str, Any]:
    records = []
    global_categories: Dict[str, Dict[str, str]] = {}

    for json_path in sorted(rag_root.glob("VID*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", {})
        if not global_categories:
            global_categories = categories

        video_id = str(data.get("video") or json_path.stem.replace("VID", ""))
        image_dir = rag_root / f"images{video_id}"

        for frame_key, anns in data.get("annotations", {}).items():
            instruments = set()
            phases = set()
            regions = []
            triplets = set()
            verbs = set()
            targets = set()

            for ann in anns:
                if len(ann) < 15:
                    continue
                instrument = category_lookup(categories, "instrument", ann[1])
                triplet = category_lookup(categories, "triplet", ann[0])
                verb = category_lookup(categories, "verb", ann[7])
                target = category_lookup(categories, "target", ann[8])
                phase = category_lookup(categories, "phase", ann[-1])

                if instrument:
                    instruments.add(instrument)
                    regions.append(
                        f"{instrument}@{bbox_to_region(float(ann[3]), float(ann[4]), float(ann[5]), float(ann[6]))}"
                    )
                if triplet:
                    triplets.add(triplet.replace(",", " "))
                if verb and verb != "null_verb":
                    verbs.add(verb)
                if target and target != "null_target":
                    targets.add(target)
                if phase:
                    phases.add(phase)

            if not instruments and not triplets and not phases:
                continue

            image_path = image_dir / f"{int(frame_key):06d}.png"
            summary = (
                f"VID{video_id} frame {frame_key}: "
                f"phase={', '.join(sorted(phases)) or 'unknown'}; "
                f"instruments={', '.join(sorted(instruments)) or 'none'}; "
                f"actions={', '.join(sorted(triplets)) or 'none'}; "
                f"regions={', '.join(regions[:6]) or 'none'}"
            )
            label_text = " ".join(sorted(instruments | phases | triplets | verbs | targets))
            records.append(
                {
                    "video": video_id,
                    "frame": str(frame_key),
                    "image_path": image_path,
                    "summary": summary,
                    "tokens": tokenize(summary + " " + label_text),
                    "labels": sorted(instruments | phases | triplets | verbs | targets),
                }
            )

    phase_order = []
    if global_categories.get("phase"):
        phase_order = [
            global_categories["phase"][k].replace("-", " ")
            for k in sorted(global_categories["phase"], key=lambda x: int(x))
        ]

    instruments = []
    if global_categories.get("instrument"):
        instruments = [
            global_categories["instrument"][k]
            for k in sorted(global_categories["instrument"], key=lambda x: int(x))
        ]

    return {
        "records": records,
        "phase_order": phase_order,
        "instruments": instruments,
    }


def retrieve_cholec_rag(sample: Dict[str, Any], rag_index: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    if top_k <= 0 or not rag_index.get("records"):
        return []

    query = " ".join(
        [
            sample.get("question_text", ""),
            sample.get("question_type", ""),
            sample.get("primary_category", ""),
            " ".join(sample.get("options", [])),
        ]
    )
    query_norm = normalize_text(query)
    query_tokens = tokenize(query)
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for rec in rag_index["records"]:
        overlap = len(query_tokens & rec["tokens"])
        phrase_hits = 0
        for label in rec["labels"]:
            label_norm = normalize_text(label)
            if label_norm and label_norm in query_norm:
                phrase_hits += 1
        score = overlap + phrase_hits * 5
        if "next phase" in query_norm and any("phase" in normalize_text(label) for label in rec["labels"]):
            score += 1
        if score > 0:
            scored.append((score, rec))

    scored.sort(key=lambda item: (-item[0], item[1]["video"], int(item[1]["frame"])))
    return [rec for _, rec in scored[:top_k]]


def build_rag_context(sample: Dict[str, Any], rag_index: Dict[str, Any], top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
    retrieved = retrieve_cholec_rag(sample, rag_index, top_k)
    lines = [
        "CholecT50 RAG reference annotations follow. They are reference knowledge only; answer the current EgoCross video frames.",
    ]
    if rag_index.get("instruments"):
        lines.append("Known surgical instruments: " + ", ".join(rag_index["instruments"]) + ", specimen-bag.")
    if rag_index.get("phase_order"):
        lines.append("Observed surgical phase order: " + " -> ".join(rag_index["phase_order"]) + ".")

    if retrieved:
        lines.append("Most relevant annotated reference frames:")
        for i, rec in enumerate(retrieved, start=1):
            lines.append(f"{i}. {rec['summary']}")

    return "\n".join(lines), retrieved


def make_rag_visual_content(retrieved: List[Dict[str, Any]], max_examples: int) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    content = []
    opened_images = []
    for rec in retrieved[: max(0, max_examples)]:
        image_path = rec["image_path"]
        if not image_path.exists():
            continue
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((768, 768))
        opened_images.append(img)
        content.append(
            {
                "type": "text",
                "text": "[RAG reference image from CholecT50, not part of the question video]\n" + rec["summary"],
            }
        )
        content.append({"type": "image", "image": img})
    return content, opened_images


def build_prompt(question_text: str, options: List[str], fps=0.5, interval=2) -> str:
    option_text = "\n".join(options)
    return (
        "You are solving a multiple-choice visual question from egocentric video frames.\n"
        "If CholecT50 RAG reference annotations/images are provided, use them only to calibrate surgical instruments, actions, phases, and regions.\n"
        "The answer must be based on the current video frames after the time markers.\n"
        "Only answer with one uppercase letter: A, B, C, or D.\n"
        "Do not output any extra words.\n\n"
        f" Sampling rate: {fps} frames per second (FPS)\n"
        f"Question: {question_text}\n"
        f"Options:\n{option_text}"
    )


def parse_choice(text: str) -> str:
    if not text:
        return "A"
    text = text.strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    m = CHOICE_RE.search(text)
    return m.group(1).upper() if m else "None"


def resolve_frame_paths(video_paths: List[str], data_root: Path) -> List[Path]:
    resolved = []
    for p in video_paths:
        rel = p.lstrip("/")
        abs_path = data_root / rel
        resolved.append(abs_path)
    return resolved


def getContent(images, all_paths, indices, fps):
    content = []
    selected_frame_indices = []
    for idx in indices:
        path = all_paths[idx]
        filename = os.path.basename(path)
        match = re.search(r"_(\d+)(?:\.\w+)?$", filename)
        if match is None:
            print(f"Warning: cannot parse frame index from {filename}")
            continue
        frame_idx = int(match.group(1))
        selected_frame_indices.append(frame_idx)

    if len(selected_frame_indices) != len(indices):
        print("Error: selected frame indices length != indices length")
        return content

    start_idx = selected_frame_indices[0]
    for img, frame_idx in zip(images, selected_frame_indices):
        timestamp = (frame_idx - start_idx) / fps
        content.append(
            {
                "type": "text",
                "text": f"<time_start> {timestamp:.1f} <time_end>",
            }
        )
        content.append(
            {
                "type": "image",
                "image": img,
            }
        )
    return content


def main() -> None:
    args = parse_args()
    testData_json = Path(args.testData_json)
    template_json = Path(args.submission_sample)
    data_root = Path(args.data_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with testData_json.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    with template_json.open("r", encoding="utf-8") as f:
        sub_template = json.load(f)
    sub_template = {item["id"]: item for item in sub_template}

    rag_index = {"records": [], "phase_order": [], "instruments": []}
    if not args.disable_rag:
        rag_index = build_cholec_rag_index(Path(args.rag_root))
        print(f"Loaded CholecT50 RAG index: {len(rag_index['records'])} annotated frames from {args.rag_root}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    results = []

    for sample in tqdm(samples, desc="Infer"):
        dataset_name = sample.get("dataset", "")
        if (dataset_name != "EgoSurgery" and dataset_name != "CholecTrack20"):
        # if  dataset_name != "CholecTrack20":
            continue
        sid = sample["id"]
        question_text = sample["question_text"]
        options = sample["options"]
        line = sub_template[sid]
        frame_paths_all = resolve_frame_paths(sample["video_path"], data_root)

        frame_paths, idxs = sample_frames(frame_paths_all, args.max_frames)

        images = []
        valid_paths = []
        for p in frame_paths:
            if p.exists():
                img = Image.open(p).convert("RGB")
                img.thumbnail((1024, 1024))
                images.append(img)
                valid_paths.append(str(p))

        # if not images:
        #     line["answer"] = "A"
        #     results.append(line)
        #     continue

        fps = 0.5
        interval = 2.0
        prompt = build_prompt(question_text, options, fps, interval)

        content = []
        rag_images = []
        if (dataset_name == "CholecTrack20" or dataset_name == "EgoSurgery")and rag_index.get("records"):
            rag_context, retrieved = build_rag_context(sample, rag_index, args.rag_top_k)
            content.append({"type": "text", "text": rag_context})
            rag_content, rag_images = make_rag_visual_content(retrieved, args.rag_visual_examples)
            content.extend(rag_content)

        content.append({"type": "text", "text": "Current EgoCross question video frames begin below."})
        content.extend(getContent(images, frame_paths_all, idxs, fps))
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = args.temperature > 0
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        generated_ids = model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(sid, output_text)
        pred = parse_choice(output_text)
        line["answer"] = pred
        results.append(line)

        for img in images:
            img.close()
        for img in rag_images:
            img.close()

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} predictions to: {output_path}")


if __name__ == "__main__":
    main()
