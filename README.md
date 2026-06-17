# HSD-Net: Hierarchical Style Decoupling Network for Cross-Domain Few-Shot Learning

Official implementation of **HSD-Net**, a lightweight framework for cross-domain few-shot learning (CD-FSL) based on hierarchical style decoupling and low-texture-aware augmentation.

HSD-Net assigns different roles to different depths of a ResNet-12 backbone: the first two layers are forced into a domain-agnostic, fully-normalized representation (Hard StyleNormGate), while the last two layers focus on extracting cross-domain invariant semantic features (rotation self-supervision + low-texture augmentation). The result is a framework that trains in **~2.7 hours** on a single RTX 4090 — about **2.9× faster than SVasP** — while matching or exceeding state-of-the-art accuracy on five standard CD-FSL benchmarks.

## Repository Structure

```
.
├── models/
│   ├── dd_resnet12.py        
├── datasets/
│   └── cd_dataset_adapter.py     ├── train_dd_eirm_v6.py           # Main training script
├── eval_cd_fixed_v6.py           
├── logs/                         
└── save/                        
```

## Requirements

```bash
pip install torch torchvision numpy scikit-learn pillow
```

Tested with Python 3.9+, PyTorch 2.0+, CUDA 11.8+, on a single NVIDIA RTX 4090.

## Training

HSD-Net is trained independently for the 1-shot and 5-shot settings.

### 1-shot

```bash
python train_dd_eirm_v6.py --dataset mini --shot 1 \
    --irm-coef 0 --con-coef 0 --gate-mode hard \
    --smooth-aug --smooth-prob 0.5 \
    --save-path ./save/v6_combo_1s \
    --max-epoch 200 --gpu 0 \
    2>&1 | tee logs/v6_combo_1s.log
```

### 5-shot

```bash
python train_dd_eirm_v6.py --dataset mini --shot 5 \
    --irm-coef 0 --con-coef 0 --gate-mode hard \
    --smooth-aug --smooth-prob 0.5 \
    --save-path ./save/v6_combo_5s \
    --max-epoch 200 --gpu 0 \
    2>&1 | tee logs/v6_combo_5s.log
```

