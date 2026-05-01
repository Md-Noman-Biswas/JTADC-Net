"""
dataset.py  –  Dataset loading, preprocessing, and data generators.
"""
import os, random
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from transformers import ViTImageProcessor
import tensorflow as tf

from config import IMG_SIZE, BATCH_SIZE, NUM_CLASSES, VIT_MODEL_NAME, le

# ==========================================================
# DATASET
# ==========================================================
def load_dataset(path):
    P, L = [], []
    for cls in sorted(os.listdir(path)):
        cp = os.path.join(path, cls)
        if not os.path.isdir(cp): continue
        for root, _, files in os.walk(cp):
            for f in files:
                if f.lower().endswith(("jpg","jpeg","png","bmp","tif")):
                    P.append(os.path.join(root, f))
                    L.append(cls)
    return P, L

train_paths, train_labels = load_dataset(TRAIN_DIR)
val_paths, val_labels     = load_dataset(VAL_DIR)
test_paths, test_labels   = load_dataset(TEST_DIR)

class_names = sorted(list(set(train_labels + val_labels + test_labels)))
NUM_CLASSES = len(class_names)
le = {c:i for i,c in enumerate(class_names)}

print("DATA SIZES: Train:", len(train_paths), "Val:", len(val_paths), "Test:", len(test_paths))
print("CLASSES:", class_names)

# ==========================================================
# PREPROCESS FUNCTIONS
# ==========================================================

# Initialize ViT processor globally
print("Loading ViT processor...")
vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
print("ViT processor loaded.")

def vit_preprocess_batch(images_01):
    """
    Convert batch of images in [0,1] range to ViT processor format
    Args:
        images_01: numpy array of shape (batch, H, W, C) in range [0, 1]
    Returns:
        Preprocessed images ready for ViT classifier
    """
    # Handle tensor input
    if hasattr(images_01, 'numpy'):
        images_01 = images_01.numpy()
    
    # Convert to uint8 [0, 255]
    images_uint8 = (images_01 * 255.0).astype(np.uint8)
    
    # Process with ViT processor
    processed = vit_processor(images=list(images_uint8), return_tensors="tf")["pixel_values"]
    
    # Transpose from (B, C, H, W) to (B, H, W, C)
    processed = tf.transpose(processed, [0, 2, 3, 1])
    
    return processed.numpy()

def preprocess(path, noisy=True, level="medium"):
    """
    Load and preprocess image with optional noise
    Returns: (noisy_raw, clean_raw) - both in [0,1] range
    """
    pil = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(pil).astype("float32")  # 0..255

    clean_raw = img / 255.0

    if not noisy:
        return clean_raw, clean_raw

    noisy_raw = clean_raw.copy()

    # Noise parameters based on level
    if level == "low":
        speckle, gauss, blur_p, blur_s, sp_rng = (0.03,0.06),(0.01,0.02),0.2,(0.3,0.6),(0.002,0.006)
    elif level == "high":
        speckle, gauss, blur_p, blur_s, sp_rng = (0.12,0.25),(0.05,0.10),0.6,(1.0,2.0),(0.02,0.05)
    else:  # medium
        speckle, gauss, blur_p, blur_s, sp_rng = (0.06,0.12),(0.02,0.05),0.4,(0.6,1.2),(0.01,0.03)

    # Apply noise operations
    noisy_raw += noisy_raw * np.random.normal(0, np.random.uniform(*speckle), noisy_raw.shape)
    noisy_raw += np.random.normal(0, np.random.uniform(*gauss), noisy_raw.shape)
    
    if np.random.rand() < blur_p:
        sigma = np.random.uniform(*blur_s)
        noisy_raw = gaussian_filter(noisy_raw, sigma=(sigma, sigma, 0))
    
    sp = np.random.uniform(*sp_rng)
    mask = np.random.rand(*noisy_raw.shape[:2])
    noisy_raw[mask < sp/2] = 0.0
    noisy_raw[mask > 1 - sp/2] = 1.0

    noisy_raw = np.clip(noisy_raw, 0, 1)

    return noisy_raw.astype("float32"), clean_raw.astype("float32")

# ==========================================================
# GENERATOR DATA GENERATOR
# ==========================================================
def gan_gen(paths, labels, batch_size=BATCH_SIZE):
    by_class = defaultdict(list)
    for p,l in zip(paths, labels):
        by_class[l].append(p)
    classes = list(by_class.keys())

    while True:
        nb, cb, lab = [], [], []
        for _ in range(batch_size):
            cls = random.choice(classes)
            path = random.choice(by_class[cls])
            noisy_raw, clean_raw = preprocess(path, noisy=True)
            nb.append(noisy_raw)
            cb.append(clean_raw)
            lab.append(to_categorical(le[cls], NUM_CLASSES))
        yield np.array(nb), [np.array(cb), np.array(lab)]

