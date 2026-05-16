import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import json
import re
from pathlib import Path
from typing import List
import torch
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm
from utils import convert2jsonformat
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
# conda activate egoVqa2

CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_model_path = project_root / "Model" / "Qwen3-VL-4B-Instruct"
    default_testData_json = project_root / "dataset" / "egocross_testbed" / "egocross_testbed_imgs.json" 
    default_submit_json = project_root / "dataset" / "egocross_testbed" / "submission_sample.json" 
    default_data_root = project_root / "dataset"
    default_output = Path(__file__).resolve().parent / "qwen3vl_fps_rag_510.json"
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B-Instruct baseline for EgoCross testbed")
    parser.add_argument("--model-path", type=str, default=str(default_model_path), help="Local model path or HF repo id")
    parser.add_argument("--testData-json", type=str, default=str(default_testData_json), help="Path to egocross_testbed_imgs.json")
    parser.add_argument("--submission-sample", type=str, default=str(default_submit_json), help="Path to egocross_testbed_imgs.json")
    parser.add_argument("--data-root", type=str, default=str(default_data_root), help="Dataset root directory containing egocross_testbed")
    parser.add_argument("--output", type=str, default=str(default_output), help="Output prediction json path")
    parser.add_argument("--max-frames", type=int, default=12, help="Maximum number of frames per sample")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.5)
    return parser.parse_args()


def sample_frames(frame_paths: List[str], max_frames: int):
    idxs=[i for i in range(len(frame_paths))]
    if len(frame_paths) <= max_frames:
        return frame_paths,idxs

    idxs = [round(i * (len(frame_paths) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [frame_paths[i] for i in idxs], idxs


def build_prompt(question_text: str, options: List[str],fps=0.5,interval=2) -> str:
    option_text = "\n".join(options)
    return (
        "You are solving a multiple-choice visual question from egocentric video frames.\n"
        "Only answer with one uppercase letter: A, B, C, or D.\n"
        "Do not output any extra words.\n\n"
        f" Sampling rate: {fps} frames per second (FPS)\n"
        f"Question: {question_text}\n"
        f"Options:\n{option_text}"
    )

    """
    构建 vLLM messages 中的 user content。
    格式：[时间戳文字] + [图像] + [时间戳文字] + [图像] + ... + [问题+选项]
    """
    option_text = "\n".join(options)
    timestamps = get_frame_timestamps(len(images_list), fps)
    content = []

    for i, (img, ts) in enumerate(zip(images_list, timestamps)):
        # 每帧前插入时间戳标注
        content.append({
            "type": "text",
            "text": f"[frame {i} | timestamps: {ts:.1f}s]"
        })
        content.append({
            "type": "image",
            "image": img
        })
        content.append({
            f"Question: {question_text}\n"
            f"Options:\n{option_text}"
        })
    return content

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
    selected_frame_indices = [] #文件名称里代表的数字序号
    # 根据indices去valid_paths里找真实frame id
    for idx in indices:
        path = all_paths[idx]
        filename = os.path.basename(path)
        # 提取最后一个_后的数字
        match = re.search(r'_(\d+)(?:\.\w+)?$', filename)
        if match is None:
            print(f"Warning: cannot parse frame index from {filename}")
            continue
        frame_idx = int(match.group(1))
        selected_frame_indices.append(int(frame_idx))

    if len(selected_frame_indices) != len(indices):
        print("Error: selected frame indices length != indices length")
        return content

    # 第一张作为时间0点
    start_idx = selected_frame_indices[0]
    # images 和 selected_frame_indices 对齐
    for img, frame_idx in zip(images, selected_frame_indices):

        timestamp = (frame_idx - start_idx ) / fps
        content.append({
            "type": "text",
            "text": f"<time_start> {timestamp:.1f} <time_end>"
        })
        content.append({
            "type": "image",
            "image": img
        })
    return content

def main() -> None:
    args = parse_args()
    testData_json = Path(args.testData_json)
    template_json = Path(args.submission_sample)
    data_root = Path(args.data_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    #测试集读取数据
    with testData_json.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    #提交模板
    with template_json.open("r", encoding="utf-8") as f:
        sub_template = json.load(f)
    sub_template={item["id"]: item for item in sub_template}#成字典了
        

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    results = []

    for sample in tqdm(samples, desc="Infer"):
        sid = sample["id"]
        question_text = sample["question_text"]
        options = sample["options"]
        line  =sub_template[sid]
        frame_paths_all = resolve_frame_paths(sample["video_path"], data_root)
        
        #限制抽帧的数量
        frame_paths, idxs = sample_frames(frame_paths_all, args.max_frames)

        images = []
        valid_paths = []
        for p in frame_paths:
            if p.exists():
                img = Image.open(p).convert("RGB")
                img.thumbnail((1024, 1024))  # 限制最长边 <= 512（等比例缩放）
                images.append(img)
                valid_paths.append(str(p))
                
        if not images:
            results.append(
                {
                    "id": sid,
                    "pred": "A",
                    "raw_response": "",
                    "used_frames": [],
                }
            )
            continue

        # if sample['dataset']=='CholecTrack20' and ("VID25" in sample["video_path"][0] or "VID111" in sample["video_path"][0] ):
        #     fps=1.0
        #     interval=1.0
        #     print("CholecTrack20视频VID25和VID111所有EgoSurgery片段均以 1 FPS 的速率提供")
        # else:
        #     fps=0.5
        #     interval=2.0
        fps=0.5
        interval=2.0
        prompt = build_prompt(question_text, options, fps, interval)
        # content = [{"type": "image", "image": img} for img in images]
        content = getContent(images,frame_paths_all ,idxs ,fps)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        #0
        inputs = processor.apply_chat_template( messages, tokenize=True, add_generation_prompt=True,return_dict=True,return_tensors="pt" )
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
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        #1
        # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info(messages)
        # inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        # inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        # with torch.no_grad():
        #     out_ids = model.generate(**inputs, 
        #                             max_new_tokens=args.max_new_tokens,                              
        #                              )
        # gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        # output_text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print( sid,output_text)
        pred = parse_choice(output_text)
        line["answer"]=pred
        results.append(line )

        for img in images:
            img.close()

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} predictions to: {output_path}")


if __name__ == "__main__":
    main()
