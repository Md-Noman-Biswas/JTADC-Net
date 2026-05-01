# JTADC-Net: An Adversarial Joint Task-Aware Denoising and Classification Network for Noise Robust Gallbladder Disease Diagnosis

This repository contains the official implementation of a joint training framework combining a **Medical ResSE-UNet GAN** (for image denoising) with a fine-tuned **Vision Transformer (ViT) classifier**, applied to gallbladder ultrasound image classification.

## Dataset Preparation

Download the gallbladder ultrasound dataset from Kaggle: [gallbladder-split](https://data.mendeley.com/datasets/r6h24d2d3y/2). Update the `SPLIT_ROOT` path in `config.py` to point to your local copy.

## Usage

### Step 1: Configure Paths and Hyperparameters

Edit `config.py` to set your dataset path and adjust training parameters if needed.

### Step 2: Run Sanity Checks

Before training, verify your setup:

```bash
python tests.py
```

### Step 3: Train the Model

```bash
python train.py
```

The best checkpoints are saved as:
- `generator_best.keras`
- `classifier_vit_best_joint.keras`

## File Overview

- **`config.py`** — Hyperparameters and dataset paths
- **`models.py`** — Generator (ResSE-UNet), Discriminator, ViT Classifier
- **`dataset.py`** — Data loading, noise augmentation, batch generators
- **`train.py`** — Joint GAN + Classifier training loop
- **`tests.py`** — Pre-training sanity checks
- **`utils.py`** — ViT selective unfreezing utility
- **`notebook/`** — Original Kaggle notebook
