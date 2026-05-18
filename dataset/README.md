---
license:
  - cc-by-sa-4.0
  - cc-by-nc-4.0
tags:
  - egocentric-video
  - visual-question-answering
  - multimodal
  - evaluation
  - benchmark
  - training
pretty_name: EgoCross
language:
  - en
---

# EgoCross

[Homepage](https://egocross-benchmark.github.io/) | Challenge (coming soon) | [HuggingFace Dataset](https://huggingface.co/datasets/myuniverse/EgoCross) | [arXiv](https://arxiv.org/abs/2508.10729) | [GitHub](https://github.com/MyUniverse0726/EgoCross)

EgoCross focuses on cross-domain egocentric video question answering with two complementary parts:

- `egocross_testbed/`: benchmark testbed for evaluation, including the `egocross_testbed_imgs.json` test query file.
- `EgoCross_support_set/`: support-set training data for model adaptation.

## Benchmark Testbed (`egocross_testbed`)

### Description

The benchmark couples egocentric videos from five public sources:

- `CholecTrack20`
- `EgoSurgery`
- `EgoPet`
- `ENIGMA`
- `ExtrameSportFPV`

It provides 957 multi-choice QA pairs across four reasoning categories:

| Category | QA pairs |
|---|---:|
| Counting | 114 |
| Localization | 284 |
| Identification | 398 |
| Prediction | 161 |

For each QA pair, a short sequence of RGB frames is extracted around the evidence window and stored as JPEG files.

Unless otherwise noted, frames are sampled at 0.5 FPS from source videos.
`CholecTrack20` videos `VID25` and `VID111`, and all `EgoSurgery` clips, are provided at 1 FPS.

### Directory Layout

```text
egocross_testbed/
├── egocross_testbed_imgs.json
├── CholecTrack20/
│   └── generated/
│       └── VIDxx/frames/<qid>/frame_00000.jpg ...
├── EgoSurgery/
│   └── generated/
│       └── xx/frames/<qid>/frame_00000.jpg ...
├── ENIGMA/
│   └── generated/
│       └── xxx/frames/<qid>/frame_00000.jpg ...
├── ExtrameSportFPV/
│   └── generated/
│       └── VIDxxx/frames/<qid>/frame_00000.jpg ...
└── EgoPet/
    └── generated/
        └── xxx/frames/<qid>/frame_00000.jpg ...
```

### QA Annotation Format

The benchmark test query file is stored at `egocross_testbed/egocross_testbed_imgs.json`.
It uses the following schema:

- `id`, `dataset`, `question_id`
- `primary_category`, `question_type`
- `question_text`, `options`
- `correct_option_letter`, `answer_text`, `detailed_answer`
- `original_video_fps`
- `video_path` (list of frame paths)

### Evaluation

Use `egocross_testbed/egocross_testbed_imgs.json` as the evaluation question file.
The referenced frame paths in `video_path` resolve under `egocross_testbed/<dataset>/generated/...`.
Evaluation scripts can refer to [EgoCrossCodes](https://github.com/EgoCross-Benchmark/EgoCrossCodes).

## Support Set (`EgoCross_support_set`)

### Description

The support set contains 80 multi-choice QA samples in ShareGPT multimodal format:

| Domain | Source | Samples |
|---|---|---:|
| Animal | EgoPet | 20 |
| Industry | ENIGMA | 20 |
| XSports | ExtrameSportFPV | 20 |
| Surgery | CholecTrack20 | 20 |

Total frame count is 1259 images.

### Files

- `EgoCross_support_set/train.json`
- `EgoCross_support_set/train_animal.json`
- `EgoCross_support_set/train_industry.json`
- `EgoCross_support_set/train_xsports.json`
- `EgoCross_support_set/train_surgery.json`
- `EgoCross_support_set/dataset_info.json`
- `EgoCross_support_set/frames/`

### Training Sample Format

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image><image>...<image>Question?\nA. ...\nB. ...\nC. ...\nD. ..."
    },
    {
      "role": "assistant",
      "content": "A"
    }
  ],
  "images": ["frames/.../frame_xxx.jpg"],
  "domain": "animal"
}
```

### Training Config

`EgoCross_support_set/dataset_info.json` is included for LLaMA-Factory style loading:

- formatting: `sharegpt`
- columns: `messages`, `images`
- entries: `egocross`, `egocross_animal`, `egocross_industry`, `egocross_xsports`, `egocross_surgery`

## Train/Eval Scope

- Evaluation data: `egocross_testbed/` plus the benchmark test query file `egocross_testbed/egocross_testbed_imgs.json`.
- Training data: `EgoCross_support_set/` with ShareGPT QA and frame lists.
- These two parts intentionally use different schemas because they target different stages (evaluation vs training).

## HuggingFace Upload Notes

- Keep only this root `README.md` as dataset card.
- Ensure Git LFS is enabled before push (`git lfs install`).
- Upload from repository root so `egocross_testbed/` and `EgoCross_support_set/` are both included.

## Citation

```bibtex
@article{li2025egocross,
  title={Egocross: Benchmarking multimodal large language models for cross-domain egocentric video question answering},
  author={Li, Yanjun and Fu, Yuqian and Qian, Tianwen and Xu, Qi'ao and Dai, Silong and Paudel, Danda Pani and Van Gool, Luc and Wang, Xiaoling},
  journal={arXiv preprint arXiv:2508.10729},
  year={2025}
}
```
