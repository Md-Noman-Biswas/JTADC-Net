# GAN + ViT Joint Training for Medical Image Classification

This repository contains the official implementation for our paper:

> **[Your Paper Title Here]**  
> [Your Name(s)], [Conference/Journal Name], [Year]

---

## Overview

A joint training framework combining a **Medical ResSE-UNet GAN** (image denoising/enhancement) with a fine-tuned **Vision Transformer (ViT) classifier**, trained end-to-end on a gallbladder ultrasound dataset.

### Architecture

| Component | Details |
|-----------|---------|
| **Generator** | Medical ResSE-UNet with Residual Squeeze-and-Excitation blocks + Attention Gates |
| **Discriminator** | Self-Attention Discriminator (LSGAN loss) |
| **Classifier** | Fine-tuned `google/vit-base-patch16-224-in21k` |
| **Loss** | LSGAN + L1 Reconstruction (λ=100) + Classification (λ=1) |
| **Training** | Classifier frozen epochs 1–15, unfrozen with LR=1e-4 afterwards |

---

## Repository Structure

```
├── config.py          # Hyperparameters and dataset paths
├── dataset.py         # Data loading, preprocessing, augmentation, generators
├── models.py          # Generator, Discriminator, ViT Classifier definitions
├── train.py           # Joint GAN + Classifier training loop
├── tests.py           # Pre-training sanity checks
├── utils.py           # Helper utilities (ViT selective unfreezing)
├── requirements.txt   # Python dependencies
└── notebook/
    └── gan-classifier-combined-training.ipynb  # Original Kaggle notebook
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

---

## Dataset Setup

Update `SPLIT_ROOT` in `config.py`:

```python
SPLIT_ROOT = "/path/to/your/gallbladder_split"
```

Expected directory layout:

```
gallbladder_split/
├── train/
│   ├── class_name_1/
│   └── class_name_2/
├── val/
└── test/
```

---

## Running

```bash
# Step 1: Verify setup (sanity checks)
python tests.py

# Step 2: Train
python train.py
```

Saved checkpoints:
- `generator_best.keras`
- `classifier_vit_best_joint.keras`

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 |
| Batch size | 8 |
| GAN epochs | 70 |
| Classifier unfreeze epoch | 15 |
| LR – Generator | 2e-4 |
| LR – Discriminator | 1e-4 |
| LR – Classifier (fine-tune) | 1e-4 |
| λ_rec (reconstruction) | 100 |
| λ_cls (classification) | 1 |

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{yourname2025,
  title   = {Your Paper Title Here},
  author  = {Last, First and Co-Author, Name},
  journal = {Journal or Conference Name},
  year    = {2025}
}
```

---

## License

[MIT](LICENSE)
