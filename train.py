"""
train.py  –  Joint GAN + ViT Classifier training loop.

Usage:
    python tests.py   # run sanity checks first
    python train.py   # start training
"""
import os, random, math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from config import *
from dataset import load_dataset, gan_gen, vit_preprocess_batch
from models import build_generator, build_discriminator, build_vit_classifier
from utils import unfreeze_last_n_vit_blocks

# ==========================================================
# UPDATED TRAINING LOOP - MATCHING YOUR METHODOLOGY
# Classifier trained on BOTH clean and denoised images
# Testing on denoised images (actual inference pipeline)
# ==========================================================

best_val_acc = -1
wait = 0
patience = 7

# Reset history
psnr_hist = []
ssim_hist = []
g_loss_hist = []
d_loss_hist = []
train_acc_hist = []
val_acc_hist = []

print("\n" + "="*60)
print("STARTING TRAINING - YOUR METHODOLOGY")
print("Classifier will see BOTH clean and denoised images")
print("Testing on denoised images (matching inference)")
print("="*60 + "\n")

for ep in range(GAN_EPOCHS):
    print(f"\n=========== EPOCH {ep+1}/{GAN_EPOCHS} ===========\n")

    # ---------------------------
    # Unfreeze classifier after epoch 10
    # ---------------------------
    # ==================================================
    # Progressive ViT Unfreezing
    # ==================================================

    if ep == UNFREEZE_EPOCH:
        print(">>> Starting ViT fine-tuning (2 blocks)")
        C.trainable = True
        unfreeze_last_n_vit_blocks(vit_base, 2)
    
        C.compile(
            optimizer=Adam(LR_C_FINE),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
    
    if ep == UNFREEZE_EPOCH + 6:
        print(">>> Expanding to 4 blocks")
        unfreeze_last_n_vit_blocks(vit_base, 4)
    
        C.compile(
            optimizer=Adam(LR_C_FINE),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
    
    if ep == UNFREEZE_EPOCH + 12:
        print(">>> Expanding to 6 blocks")
        unfreeze_last_n_vit_blocks(vit_base, 6)
    
        C.compile(
            optimizer=Adam(LR_C_FINE),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )


    # Adaptive D steps
    n_d_steps = max(2, 3 - ep//15)
    print(f"Training with {n_d_steps} discriminator step(s) per batch")

    # Cache patch shape at start of epoch
    patch_shape = None

    pbar = tqdm(range(STEPS_PER_EPOCH))
    for step in pbar:
        noisy_b, targets = next(gen)
        clean_b = targets[0]
        lbl_b   = targets[1]
        
        # Get patch shape (only need to do this once per epoch)
        if patch_shape is None:
            patch_shape = D.predict(clean_b[:1], verbose=0).shape[1:]
        
        real_y = REAL_LABEL * np.ones((BATCH_SIZE, *patch_shape))
        fake_y = FAKE_LABEL * np.ones((BATCH_SIZE, *patch_shape))

        # ---------------------------
        # TRAIN DISCRIMINATOR
        # ---------------------------
        D.trainable = True
        d_losses_batch = []
        
        for _ in range(n_d_steps):
            # Generate fake images for discriminator training
            fake_b_for_d = G.predict(noisy_b, verbose=0)
            
            d_loss = D.train_on_batch(
                np.concatenate([clean_b, fake_b_for_d]),
                np.concatenate([real_y, fake_y])
            )
            d_losses_batch.append(d_loss[0] if isinstance(d_loss, list) else d_loss)
        
        d_loss = np.mean(d_losses_batch)
        d_loss_hist.append(d_loss)
        
        # ---------------------------
        # TRAIN GENERATOR (FIXED - PROPER GRADIENT FLOW!)
        # ---------------------------
        D.trainable = False
        
        with tf.GradientTape() as tape:
            # Generate fake images INSIDE gradient tape
            fake_gen = G(noisy_b, training=True)
            
            # Get discriminator output (adversarial loss)
            valid_out = D(fake_gen, training=False)
            
            # Get classifier output on THE SAME fake_gen
            # C now has preprocessing built-in, so we pass [0,1] images directly
            cls_out = C(fake_gen, training=False)
            
            # Calculate losses
            adv_loss = tf.keras.losses.MeanSquaredError()(real_y, valid_out)
            rec_loss = tf.keras.losses.MeanAbsoluteError()(clean_b, fake_gen)
            cls_loss = tf.keras.losses.CategoricalCrossentropy()(lbl_b, cls_out)
            
            # Total generator loss
            # Weights: adversarial=1, reconstruction=100, classification=1
            # inside epoch loop
            # if ep < UNFREEZE_EPOCH:
            #     lambda_rec, lambda_cls = 100.0, 0.5
            # elif ep < UNFREEZE_EPOCH + 10:
            #     lambda_rec, lambda_cls = 75.0, 1.0
            # else:
            #     lambda_rec, lambda_cls = 50.0, 1.5
            
            g_total_loss = adv_loss + 100 * rec_loss + cls_loss
        
        # Apply gradients
        g_grads = tape.gradient(g_total_loss, G.trainable_variables)
        g_optimizer.apply_gradients(zip(g_grads, G.trainable_variables))
        g_loss_hist.append(float(g_total_loss))

        # ---------------------------
        # FINE-TUNE CLASSIFIER (YOUR METHODOLOGY!)
        # Train on BOTH clean and denoised images
        # ---------------------------
        if ep >= UNFREEZE_EPOCH:
            # 1. Train on real clean images
            # (Simulates: clean input → GAN → output)
            C.train_on_batch(clean_b, lbl_b)
            
            # 2. Train on denoised images
            # (Simulates: noisy input → GAN → denoised output)
            fake_b_for_cls = G.predict(noisy_b, verbose=0)
            C.train_on_batch(fake_b_for_cls, lbl_b)
            
            # ✅ Classifier learns from BOTH distributions it will see at test time

        pbar.set_postfix({
            'D': f"{np.mean(d_loss_hist[-10:]):.3f}",
            'G': f"{np.mean(g_loss_hist[-10:]):.3f}"
        })
    
    # ---------------------------
    # Validation PSNR & SSIM
    # ---------------------------
    idxs = np.random.choice(len(val_paths), 32, replace=False)
    v_noisy = []
    v_clean = []
    for i in idxs:
        nr, cr = preprocess(val_paths[i], noisy=True)
        v_noisy.append(nr)
        v_clean.append(cr)
    v_noisy = np.array(v_noisy)
    v_clean = np.array(v_clean)

    den = G.predict(v_noisy, verbose=0)
    val_psnr = PSNR(v_clean, den)
    val_ssim = SSIM(v_clean, den)
    
    psnr_hist.append(val_psnr)
    ssim_hist.append(val_ssim)
    print(f"PSNR: {val_psnr:.3f} | SSIM: {val_ssim:.4f}")

    # ---------------------------
    # TRAIN & VAL Accuracy (ON DENOISED - YOUR METHODOLOGY!)
    # ---------------------------
    # Testing on denoised images to match inference pipeline
    train_acc = compute_acc_denoised(train_paths, train_labels, samples=128)
    val_acc   = compute_acc_denoised(val_paths, val_labels, samples=128)

    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)

    print(f"Train Acc (denoised): {train_acc*100:.2f}%  |  Val Acc (denoised): {val_acc*100:.2f}%")

    # ---------------------------
    # Visualization
    # ---------------------------
    sidx = idxs[0]
    nr, cr = preprocess(val_paths[sidx], noisy=True)
    out = G.predict(np.expand_dims(nr, 0), verbose=0)[0]

    def vis(x):
        return np.clip(x*255.0, 0, 255).astype(np.uint8)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(vis(nr)); plt.title("Noisy"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(vis(out)); plt.title("Denoised"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(vis(cr)); plt.title("Clean"); plt.axis("off")
    plt.suptitle(f"Epoch {ep+1} | PSNR {val_psnr:.2f} | SSIM {val_ssim:.3f}")
    plt.show()

    # ---------------------------
    # Save best models
    # ---------------------------
    if ep >= UNFREEZE_EPOCH:
        if val_acc > best_val_acc:
            print(f"✨ Improved Val Acc ({best_val_acc*100:.2f}% → {val_acc*100:.2f}%) — saving models.")
            best_val_acc = val_acc
            wait = 0
            G.save(BEST_GEN_PATH)
            C.save(BEST_CLS_PATH)
        else:
            wait += 1
            print(f"No improvement in Val Acc. Patience: {wait}/{patience}")
            if wait >= patience:
                print(f"⏹ Early stopping triggered. Best Val Acc: {best_val_acc*100:.2f}%")
                break

# ==========================================================
# FINAL TEST RESULTS - YOUR METHODOLOGY
# Testing on DENOISED images (matching inference pipeline)
# ==========================================================
print("\n=========== FINAL TEST RESULTS ===========\n")

# Load noisy test images and denoise them (YOUR METHODOLOGY)
test_noisy = []
test_lbl = []
for p, l in zip(test_paths, test_labels):
    nr, _ = preprocess(p, noisy=True)
    test_noisy.append(nr)
    test_lbl.append(le[l])
test_noisy = np.array(test_noisy)

# Denoise test images (what happens at inference)
denoised = G.predict(test_noisy, verbose=0)

# Classify denoised images
# Model now has preprocessing built-in
logits = C.predict(denoised, verbose=0)

preds = np.argmax(logits, axis=1)
true = np.array(test_lbl)

test_acc = np.mean(preds == true)
print(f"TEST ACCURACY (on denoised images): {test_acc*100:.2f}%\n")
print(classification_report(true, preds, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(true, preds)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix - GAN+ViT (Denoised Images)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ==========================================================
# BONUS: Also test on clean images for comparison
# ==========================================================
print("\n" + "="*60)
print("BONUS: Testing on clean images (no denoising)")
print("="*60 + "\n")

test_clean = []
for p in test_paths:
    _, clean = preprocess(p, noisy=False)
    test_clean.append(clean)
test_clean = np.array(test_clean)

# Classify clean images directly
logits_clean = C.predict(test_clean, verbose=0)
preds_clean = np.argmax(logits_clean, axis=1)

test_acc_clean = np.mean(preds_clean == true)
print(f"TEST ACCURACY (on clean images): {test_acc_clean*100:.2f}%")
print("\nComparison:")
print(f"  Denoised: {test_acc*100:.2f}%")
print(f"  Clean:    {test_acc_clean*100:.2f}%")
print(f"  Difference: {abs(test_acc - test_acc_clean)*100:.2f}%")

# ==========================================================
# TRAINING CURVES
# ==========================================================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(psnr_hist, marker='o', label="PSNR")
plt.title("PSNR Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.grid()
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(ssim_hist, marker='o', label="SSIM", color='green')
plt.title("SSIM Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.grid()
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_acc_hist, label="Train Acc (denoised)")
plt.plot(val_acc_hist, label="Val Acc (denoised)")
plt.title("ViT Classifier Accuracy (on denoised images)")
plt.xlabel("Epoch")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(g_loss_hist, alpha=0.6)
plt.title("Generator Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(d_loss_hist, color='red', alpha=0.6)
plt.title("Discriminator Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid()

plt.tight_layout()
plt.show()

print("\n✅ Training complete!")
print(f"\n📁 Models saved:")
print(f"   Generator: {BEST_GEN_PATH}")
print(f"   ViT Classifier: {BEST_CLS_PATH}")
print("\n📊 Your Methodology:")
print("   - Classifier trained on BOTH clean and denoised images")
print("   - Testing on denoised images (matches inference)")
print("   - GAN receives classification-aware gradients")