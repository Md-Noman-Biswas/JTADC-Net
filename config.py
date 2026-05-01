"""
config.py  –  Central hyperparameters and paths.
Edit SPLIT_ROOT to point to your dataset.
"""
import os
import random
import numpy as np
import tensorflow as tf

# ==========================================================
# CONFIG
# ==========================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IMG_SIZE = 224
BATCH_SIZE = 8
GAN_EPOCHS = 70
UNFREEZE_EPOCH = 15

STEPS_PER_EPOCH = 250

LR_G = 2e-4              # Generator learning rate
LR_D = 1e-4              # Discriminator learning rate
LR_C_FINE = 1e-4        # Classifier LR during fine-tuning (matching ViT training)

REAL_LABEL = 1.0         # LSGAN
FAKE_LABEL = 0.0

lambda_rec = 100
lambda_cls = 1

SPLIT_ROOT = "/kaggle/input/gallbladder-split/kaggle/working/gallbladder_split"
TRAIN_DIR = os.path.join(SPLIT_ROOT, "train")
VAL_DIR   = os.path.join(SPLIT_ROOT, "val")
TEST_DIR  = os.path.join(SPLIT_ROOT, "test")

BEST_GEN_PATH = "generator_best.keras"
BEST_CLS_PATH = "classifier_vit_best_joint.keras"

# ViT configuration
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"



# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
