# Cross-domain Egocentric VQA

This project implements a cross-domain egocentric visual question answering (VQA) framework based on Qwen3-VL-4B-Instruct.

## Features

- Qwen3-VL-4B-Instruct
- LoRA adaptation
- Surgical RAG retrieval
- Temporal frame modeling
- Multi-domain inference

## Supported Datasets

- EgoCross
- CholecTrack20
- EgoSurgery
- EgoPet
- XSports

## Environment

```bash
conda create -n egoVqa python=3.10
conda activate egoVqa
pip install -r requirements.txt
```

## Environment Setup

### 1. Model Preparation

Please download the base model manually using ModelScope:

```bash
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir ./models/Qwen3-VL-4B-Instruct
```
### 2. LoRA Checkpoints

The LoRA checkpoints are not included in this repository due to storage limitations. They can be downloaded from https://drive.google.com/drive/folders/1dOjGz46CT59WnUKQEZJEIL9zx6nwziv3?usp=drive_link.
      
After downloading, place them under lora_output/ as follows:
```bash
lora_output/
├── qwen3vl_time_lora/
└── qwen3vl_time_lora_8f512_ep2_lr5e5/
```

## train
```bash
python lora_train_time.py
```
## infer
```bash
python infer_time_loraAux.py
```

## Structure

```text
dataset/
models/
└───Qwen3-VL-4B-Instruct
lora_output/
├── qwen3vl_time_lora/
└── qwen3vl_time_lora_8f512_ep2_lr5e5/
infer_time_codex_loraAux.py
lora_train_time.py
```

