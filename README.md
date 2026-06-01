# JTADC-Net: An Adversarial Joint Task-Aware Denoising and Classification Network for Noise-Robust Gallbladder Disease Diagnosis

Official implementation of a joint training framework combining a **Medical ResSE-UNet GAN** (image denoising) with a fine-tuned **Vision Transformer (ViT) classifier**, for robust gallbladder ultrasound image classification under noisy acquisition conditions.

## 🔗 Live Demo

- **Try it:** [gallbladder-classifier.vercel.app](https://gallbladder-classifier.vercel.app/)
- **Video walkthrough:** [youtu.be/5ZfA27eNfAQ](https://youtu.be/5ZfA27eNfAQ)

## Overview

Real clinical ultrasound images are often degraded by suboptimal acquisition conditions, which undermines the reliability of standard classification models. JTADC-Net addresses this by jointly training a GAN-based denoiser (ResSE-UNet) and a ViT classifier in a single task-aware framework, so denoising is optimized for diagnostic accuracy rather than visual quality alone.

## Dataset Preparation

Download the gallbladder ultrasound dataset from Mendeley Data: [gallbladder dataset](https://data.mendeley.com/datasets/r6h24d2d3y/2). Update the `SPLIT_ROOT` path in `config.py` to point to your local copy.

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
