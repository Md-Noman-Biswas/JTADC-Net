"""
tests.py  –  Pre-training verification / sanity checks.
Run this before training to catch issues early.

Usage:
    python tests.py
"""
from config import *
from dataset import load_dataset, gan_gen, vit_preprocess_batch
from models import build_generator, build_discriminator, build_vit_classifier

# ==========================================================
# PRE-TRAINING VERIFICATION TESTS
# Run this block before training to catch issues early!
# ==========================================================

print("\n" + "="*70)
print("PRE-TRAINING VERIFICATION TESTS")
print("="*70 + "\n")

test_results = []

def test_status(name, passed, message=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    test_results.append((name, passed))
    print(f"{status}: {name}")
    if message:
        print(f"   └─ {message}")
    print()

# ==========================================================
# TEST 1: Data Loading
# ==========================================================
print("TEST 1: Data Loading")
print("-" * 70)

try:
    assert len(train_paths) > 0, "No training images found"
    assert len(val_paths) > 0, "No validation images found"
    assert len(test_paths) > 0, "No test images found"
    assert len(class_names) >= 2, "Need at least 2 classes"
    
    test_status(
        "Data Loading",
        True,
        f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}, Classes: {len(class_names)}"
    )
except AssertionError as e:
    test_status("Data Loading", False, str(e))

# ==========================================================
# TEST 2: Image Preprocessing
# ==========================================================
print("TEST 2: Image Preprocessing")
print("-" * 70)

try:
    # Test basic preprocessing
    test_path = train_paths[0]
    noisy, clean = preprocess(test_path, noisy=True)
    
    assert noisy.shape == (IMG_SIZE, IMG_SIZE, 3), f"Wrong shape: {noisy.shape}"
    assert clean.shape == (IMG_SIZE, IMG_SIZE, 3), f"Wrong shape: {clean.shape}"
    assert noisy.dtype == np.float32, f"Wrong dtype: {noisy.dtype}"
    assert 0 <= noisy.min() <= noisy.max() <= 1, f"Values out of [0,1]: [{noisy.min()}, {noisy.max()}]"
    assert 0 <= clean.min() <= clean.max() <= 1, f"Values out of [0,1]: [{clean.min()}, {clean.max()}]"
    
    test_status(
        "Image Preprocessing",
        True,
        f"Shape: {noisy.shape}, Range: [{noisy.min():.3f}, {noisy.max():.3f}]"
    )
except Exception as e:
    test_status("Image Preprocessing", False, str(e))

# ==========================================================
# TEST 3: ViT Preprocessing
# ==========================================================
print("TEST 3: ViT Preprocessing")
print("-" * 70)

try:
    # Test single image
    test_img = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    vit_processed = vit_preprocess_batch(test_img)
    
    assert vit_processed.shape == (1, IMG_SIZE, IMG_SIZE, 3), f"Wrong shape: {vit_processed.shape}"
    assert vit_processed.dtype == np.float32, f"Wrong dtype: {vit_processed.dtype}"
    
    # ViT preprocessing should produce normalized values (typically in range ~[-2, 2])
    expected_range = (-3, 3)
    in_range = expected_range[0] <= vit_processed.min() and vit_processed.max() <= expected_range[1]
    
    test_status(
        "ViT Preprocessing",
        in_range,
        f"Shape: {vit_processed.shape}, Range: [{vit_processed.min():.3f}, {vit_processed.max():.3f}]"
    )
except Exception as e:
    test_status("ViT Preprocessing", False, str(e))

# ==========================================================
# TEST 4: Data Generator
# ==========================================================
print("TEST 4: Data Generator")
print("-" * 70)

try:
    # Test generator
    gen_test = gan_gen(train_paths[:50], train_labels[:50], batch_size=BATCH_SIZE)
    noisy_batch, targets = next(gen_test)
    clean_batch, label_batch = targets
    
    assert noisy_batch.shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), f"Wrong noisy shape: {noisy_batch.shape}"
    assert clean_batch.shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), f"Wrong clean shape: {clean_batch.shape}"
    assert label_batch.shape == (BATCH_SIZE, NUM_CLASSES), f"Wrong label shape: {label_batch.shape}"
    assert 0 <= noisy_batch.min() <= noisy_batch.max() <= 1, "Noisy batch values out of range"
    assert 0 <= clean_batch.min() <= clean_batch.max() <= 1, "Clean batch values out of range"
    assert np.allclose(label_batch.sum(axis=1), 1.0), "Labels not one-hot encoded properly"
    
    test_status(
        "Data Generator",
        True,
        f"Batch shapes correct, {BATCH_SIZE} samples per batch"
    )
except Exception as e:
    test_status("Data Generator", False, str(e))

# ==========================================================
# TEST 5: Generator Model
# ==========================================================
print("TEST 5: Generator Model")
print("-" * 70)

try:
    # Test forward pass
    test_input = np.random.rand(2, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    gen_output = G.predict(test_input, verbose=0)
    
    assert gen_output.shape == (2, IMG_SIZE, IMG_SIZE, 3), f"Wrong output shape: {gen_output.shape}"
    assert 0 <= gen_output.min() <= gen_output.max() <= 1, f"Output out of [0,1]: [{gen_output.min()}, {gen_output.max()}]"
    
    # Check if model has trainable parameters
    assert G.count_params() > 0, "Generator has no parameters"
    trainable = sum([tf.size(w).numpy() for w in G.trainable_variables])
    
    test_status(
        "Generator Model",
        True,
        f"Output shape correct, {trainable:,} trainable params"
    )
except Exception as e:
    test_status("Generator Model", False, str(e))

# ==========================================================
# TEST 6: Discriminator Model
# ==========================================================
print("TEST 6: Discriminator Model")
print("-" * 70)

try:
    # Test forward pass
    test_input = np.random.rand(2, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    disc_output = D.predict(test_input, verbose=0)
    
    # PatchGAN outputs a spatial map, not a single value
    assert len(disc_output.shape) == 4, f"Wrong output dims: {disc_output.shape}"
    assert disc_output.shape[0] == 2, f"Wrong batch size: {disc_output.shape[0]}"
    assert 0 <= disc_output.min() <= disc_output.max() <= 1, f"Output out of [0,1]: [{disc_output.min()}, {disc_output.max()}]"
    
    # Check if model has trainable parameters
    assert D.count_params() > 0, "Discriminator has no parameters"
    trainable = sum([tf.size(w).numpy() for w in D.trainable_variables])
    
    test_status(
        "Discriminator Model",
        True,
        f"Output shape: {disc_output.shape}, {trainable:,} trainable params"
    )
except Exception as e:
    test_status("Discriminator Model", False, str(e))

# ==========================================================
# TEST 7: ViT Classifier Model
# ==========================================================
print("TEST 7: ViT Classifier Model")
print("-" * 70)

try:
    # Test with properly preprocessed input
    test_img = np.random.rand(2, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    test_img_vit = vit_preprocess_batch(test_img)
    
    cls_output = C.predict(test_img_vit, verbose=0)
    
    assert cls_output.shape == (2, NUM_CLASSES), f"Wrong output shape: {cls_output.shape}"
    assert np.allclose(cls_output.sum(axis=1), 1.0, atol=1e-5), "Output not softmax (doesn't sum to 1)"
    assert 0 <= cls_output.min() <= cls_output.max() <= 1, "Output values out of [0,1]"
    
    # Check if model has trainable parameters
    total_params = C.count_params()
    trainable = sum([tf.size(w).numpy() for w in C.trainable_variables])
    
    # Initially frozen, so trainable should be 0 or very small
    is_frozen = trainable < total_params * 0.1
    
    test_status(
        "ViT Classifier Model",
        True,
        f"Output shape correct, {total_params:,} total params, {trainable:,} trainable (frozen: {is_frozen})"
    )
except Exception as e:
    test_status("ViT Classifier Model", False, str(e))

# ==========================================================
# TEST 8: Optimizer Configuration
# ==========================================================
print("TEST 8: Optimizer Configuration")
print("-" * 70)

try:
    # Check optimizers exist and have correct learning rates
    assert g_optimizer.learning_rate.numpy() == LR_G, f"Wrong G LR: {g_optimizer.learning_rate.numpy()}"
    assert d_optimizer.learning_rate.numpy() == LR_D, f"Wrong D LR: {d_optimizer.learning_rate.numpy()}"
    
    # Check discriminator is compiled
    assert D.optimizer is not None, "Discriminator not compiled"
    
    test_status(
        "Optimizer Configuration",
        True,
        f"G LR: {LR_G}, D LR: {LR_D}, Clipnorm: 1.0"
    )
except Exception as e:
    test_status("Optimizer Configuration", False, str(e))

# ==========================================================
# TEST 9: Training Loop Components
# ==========================================================
print("TEST 9: Training Loop Components")
print("-" * 70)

try:
    # Test single training step (without updating weights)
    noisy_batch, targets = next(gen)
    clean_batch, label_batch = targets
    
    # Test G prediction
    fake_batch = G.predict(noisy_batch, verbose=0)
    assert fake_batch.shape == clean_batch.shape, "Generator output shape mismatch"
    
    # Test D prediction
    patch_shape = D.predict(clean_batch[:1], verbose=0).shape[1:]
    real_y = np.ones((BATCH_SIZE, *patch_shape))
    fake_y = np.zeros((BATCH_SIZE, *patch_shape))
    assert real_y.shape[0] == BATCH_SIZE, "Wrong real_y batch size"
    assert fake_y.shape[0] == BATCH_SIZE, "Wrong fake_y batch size"
    
    # Test ViT preprocessing in training loop
    fake_batch_vit = vit_preprocess_batch(fake_batch)
    cls_output = C.predict(fake_batch_vit, verbose=0)
    assert cls_output.shape == label_batch.shape, "Classifier output shape mismatch"
    
    test_status(
        "Training Loop Components",
        True,
        f"All components compatible, patch_shape: {patch_shape}"
    )
except Exception as e:
    test_status("Training Loop Components", False, str(e))

# ==========================================================
# TEST 10: Gradient Flow
# ==========================================================
print("TEST 10: Gradient Flow")
print("-" * 70)

try:
    # Test if gradients can be computed
    noisy_batch, targets = next(gen)
    clean_batch, label_batch = targets
    fake_batch = G.predict(noisy_batch, verbose=0)
    fake_batch_vit = vit_preprocess_batch(fake_batch)
    
    patch_shape = D.predict(clean_batch[:1], verbose=0).shape[1:]
    real_y = np.ones((BATCH_SIZE, *patch_shape))
    
    with tf.GradientTape() as tape:
        fake_gen = G(noisy_batch, training=True)
        valid_out = D(fake_gen, training=False)
        cls_out = C(fake_batch_vit, training=False)
        
        adv_loss = tf.keras.losses.MeanSquaredError()(real_y, valid_out)
        rec_loss = tf.keras.losses.MeanAbsoluteError()(clean_batch, fake_gen)
        cls_loss = tf.keras.losses.CategoricalCrossentropy()(label_batch, cls_out)
        
        g_total_loss = adv_loss + 100.0 * rec_loss + cls_loss
    
    g_grads = tape.gradient(g_total_loss, G.trainable_variables)
    
    # Check gradients exist and are not None
    assert g_grads is not None, "Gradients are None"
    assert len(g_grads) > 0, "No gradients computed"
    assert all(g is not None for g in g_grads), "Some gradients are None"
    
    # Check loss values are reasonable
    assert not tf.math.is_nan(g_total_loss), "Loss is NaN"
    assert not tf.math.is_inf(g_total_loss), "Loss is Inf"
    
    test_status(
        "Gradient Flow",
        True,
        f"Gradients computed successfully, Loss: {float(g_total_loss):.4f}"
    )
except Exception as e:
    test_status("Gradient Flow", False, str(e))

# ==========================================================
# TEST 11: Metric Functions
# ==========================================================
print("TEST 11: Metric Functions")
print("-" * 70)

try:
    # Test PSNR
    img1 = np.random.rand(5, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    img2 = np.random.rand(5, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    
    psnr_val = PSNR(img1, img2)
    assert isinstance(psnr_val, float), f"PSNR returned {type(psnr_val)}"
    assert 0 <= psnr_val <= 100, f"PSNR value unreasonable: {psnr_val}"
    
    # Test SSIM
    ssim_val = SSIM(img1, img2)
    assert isinstance(ssim_val, (float, np.floating)), f"SSIM returned {type(ssim_val)}"
    assert 0 <= ssim_val <= 1, f"SSIM value out of range: {ssim_val}"
    
    # Test accuracy computation
    acc = compute_acc(train_paths[:20], train_labels[:20], samples=20)
    assert isinstance(acc, (float, np.floating)), f"Accuracy returned {type(acc)}"
    assert 0 <= acc <= 1, f"Accuracy out of range: {acc}"
    
    test_status(
        "Metric Functions",
        True,
        f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, Sample Acc: {acc*100:.2f}%"
    )
except Exception as e:
    test_status("Metric Functions", False, str(e))

# ==========================================================
# TEST 12: Memory and Batch Size
# ==========================================================
print("TEST 12: Memory and Batch Size")
print("-" * 70)

try:
    # Test if a full training step fits in memory
    noisy_batch, targets = next(gen)
    clean_batch, label_batch = targets
    
    # Full training step
    fake_batch = G.predict(noisy_batch, verbose=0)
    D.predict(np.concatenate([clean_batch, fake_batch]), verbose=0)
    
    fake_batch_vit = vit_preprocess_batch(fake_batch)
    C.predict(fake_batch_vit, verbose=0)
    
    # If we got here, memory is sufficient
    test_status(
        "Memory and Batch Size",
        True,
        f"Batch size {BATCH_SIZE} fits in memory"
    )
except tf.errors.ResourceExhaustedError as e:
    test_status("Memory and Batch Size", False, f"OOM Error - try reducing BATCH_SIZE from {BATCH_SIZE}")
except Exception as e:
    test_status("Memory and Batch Size", False, str(e))

# ==========================================================
# TEST 13: File Paths for Saving
# ==========================================================
print("TEST 13: File Paths for Saving")
print("-" * 70)

try:
    # Check if we can write to the output directory
    import tempfile
    
    # Test if paths are writable
    test_gen_path = BEST_GEN_PATH.replace('.keras', '_test.keras')
    test_cls_path = BEST_CLS_PATH.replace('.keras', '_test.keras')
    
    # Try to save and delete
    G.save(test_gen_path)
    C.save(test_cls_path)
    
    assert os.path.exists(test_gen_path), "Generator save failed"
    assert os.path.exists(test_cls_path), "Classifier save failed"
    
    # Clean up
    os.remove(test_gen_path)
    os.remove(test_cls_path)
    
    test_status(
        "File Paths for Saving",
        True,
        f"Can save to: {os.path.dirname(BEST_GEN_PATH) or 'current directory'}"
    )
except Exception as e:
    test_status("File Paths for Saving", False, str(e))

# ==========================================================
# TEST SUMMARY
# ==========================================================
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70 + "\n")

total_tests = len(test_results)
passed_tests = sum(1 for _, passed in test_results if passed)
failed_tests = total_tests - passed_tests

print(f"Total Tests: {total_tests}")
print(f"✅ Passed: {passed_tests}")
print(f"❌ Failed: {failed_tests}")
print()

if failed_tests > 0:
    print("⚠️  FAILED TESTS:")
    for name, passed in test_results:
        if not passed:
            print(f"   - {name}")
    print()
    print("❌ FIX THESE ISSUES BEFORE TRAINING!")
else:
    print("🎉 ALL TESTS PASSED!")
    print("✅ Ready to start training!")

print("\n" + "="*70)

# Optional: Print configuration summary
if failed_tests == 0:
    print("\nCONFIGURATION SUMMARY:")
    print("-" * 70)
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Total Epochs: {GAN_EPOCHS}")
    print(f"Unfreeze Epoch: {UNFREEZE_EPOCH}")
    print(f"Steps per Epoch: {STEPS_PER_EPOCH}")
    print(f"Generator LR: {LR_G}")
    print(f"Discriminator LR: {LR_D}")
    print(f"Classifier Fine-tune LR: {LR_C_FINE}")
    print(f"\nDataset:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val: {len(val_paths)} images")
    print(f"  Test: {len(test_paths)} images")
    print(f"  Classes: {NUM_CLASSES} ({', '.join(class_names)})")
    print(f"\nModel Parameters:")
    print(f"  Generator: {G.count_params():,}")
    print(f"  Discriminator: {D.count_params():,}")
    print(f"  Classifier: {C.count_params():,}")
    print("="*70)