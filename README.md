# EIRM-Net: Synergistic Lightweight Framework for Cross-Domain Few-Shot Visual Learning

> **This repository contains the official implementation of the manuscript:**
> 
> *"Synergistic Lightweight Framework for Cross-Domain Few-Shot Visual Learning"*
> submitted to **The Visual Computer (Springer)**.
> 
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19539187.svg)](https://doi.org/10.5281/zenodo.19539187)
>
> If you find this work useful, please consider citing the manuscript (see [Citation](#citation) below).

---

## Overview

EIRM-Net is a lightweight and efficient architecture for cross-domain few-shot learning, featuring three key innovations:

- **Dynamic Relational Modeling**: Transforms static prototype averaging into learnable relationship preservation
- **Dual-Path Collaborative Learning**: Creates a closed-loop validation system ensuring both discriminative performance and representational integrity
- **Enhanced Invariant Risk Minimization (EIRM)**: Dynamically learns environmental variations to achieve causal invariance learning

---

## Requirements

- **Python**: 3.9.23
- **PyTorch**: 2.5.1+cu121
- **CUDA**: 12.1
- **GPU**: NVIDIA GPU with at least 24GB VRAM recommended (experiments run on RTX 4090)

### Dependencies

```
filelock
fsspec==2024.3.1
Jinja2==3.1.3
MarkupSafe
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.4
pillow==10.3.0
sympy==1.12
torch==2.5.1+cu121
torchvision
typing_extensions==4.11.0
```

---

## Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/dinglei0801/EIRM.git
cd EIRM
```

**Step 2: Create and activate conda environment**
```bash
conda create -n eirm python=3.9.23
conda activate eirm
```

**Step 3: Install PyTorch with CUDA 12.1**
```bash
pip install torch==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Step 4: Install remaining dependencies**
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

**Step 1:** Change the `ROOT_PATH` value in the following files to your local data path:
- `datasets/mini_imagenet.py`
- `datasets/tiered_imagenet.py`
- `datasets/cifarfs.py`

**Step 2:** Download the datasets and place them in the corresponding folders:

| Dataset | Download Source | Target Folder |
|---|---|---|
| **mini**ImageNet | [CSS](https://github.com/anyuexuan/CSS) | `data/mini-imagenet` |
| **tiered**ImageNet | [MetaOptNet](https://github.com/kjunelee/MetaOptNet) | `data/tiered-imagenet` |
| **CIFAR-FS** | [MetaOptNet](https://github.com/kjunelee/MetaOptNet) | `data/cifar-fs` |

---

## Pre-trained Models

Place pre-trained model files in the `save/` folder. To evaluate a pre-trained model, run `test.py` with the appropriate save path as described in the Evaluation section below.

---

## Training

Training follows a **three-stage progressive strategy**:

### Stage 1 — Relational Foundation (miniImageNet)
```bash
python train_stage1.py \
  --dataset mini \
  --train-way 50 \
  --train-batch 100 \
  --save-path ./save/mini-stage1
```

### Stage 2 — Dual-Path Validation (miniImageNet, 1-shot)
```bash
python train_stage2.py \
  --dataset mini \
  --shot 1 \
  --save-path ./save/mini-stage2-1s \
  --stage1-path ./save/cifarfs-stage1 \
  --train-way 20
```

### Stage 3 — Synergistic EIRM (CIFAR-FS, 5-shot)
```bash
bash run_training_with_logging_cifarfs_5.sh
```

---

## Evaluation

### 5-way 1-shot on miniImageNet
```bash
python test.py --dataset mini --shot 1 --save-path ./save/mini-stage3-1s
```

### 5-way 5-shot on miniImageNet
```bash
python test.py --dataset mini --shot 5 --save-path ./save/mini-stage3-5s
```

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{ding2025eirmnet,
  title     = {Synergistic Lightweight Framework for Cross-Domain Few-Shot Visual Learning},
  author    = {Ding, Leilei and Zhao, Shuliang},
  journal   = {The Visual Computer},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.5281/zenodo.19539187}
}
```




## License

This project is released for academic research purposes only.
