"""
models.py  –  Generator (Medical ResSE-UNet), Discriminator, and ViT Classifier.
"""
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, LeakyReLU,
    BatchNormalization, Activation, Concatenate,
    Add, Dense, Multiply, GlobalAveragePooling2D,
    Reshape, Lambda, Dropout, Softmax, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from transformers import TFViTModel

from config import IMG_SIZE, NUM_CLASSES, VIT_MODEL_NAME

try:
    from tensorflow_addons.layers import InstanceNormalization
    NormLayer = InstanceNormalization
    print("Using InstanceNormalization")
except ImportError:
    NormLayer = BatchNormalization
    print("InstanceNormalization not available, using BatchNormalization")

# ==========================================================
# BUILD IMPROVED GAN MODELS
# ==========================================================

# Prefer InstanceNorm if available (better for GANs)
try:
    from tensorflow_addons.layers import InstanceNormalization
    NormLayer = InstanceNormalization
    print("✅ Using InstanceNormalization")
except:
    NormLayer = BatchNormalization
    print("⚠️ InstanceNormalization not found, using BatchNormalization")

# Residual SE Block
def res_se_block(x_in, filters, kernel_size=3, reduction=16, norm=NormLayer):
    shortcut = x_in

    x = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x_in)
    x = norm()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = norm()(x)

    # SE block
    se = GlobalAveragePooling2D()(x)
    se = Dense(max(filters // reduction, 4), activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    x = Multiply()([x, se])

    in_channels = int(shortcut.shape[-1])
    if in_channels != filters:
        shortcut = Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = norm()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Attention Gate
def attention_gate(skip, gating, inter_channels):
    theta_x = Conv2D(inter_channels, 1, strides=1, padding='same')(skip)
    phi_g = Conv2D(inter_channels, 1, strides=1, padding='same')(gating)
    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi = Conv2D(1, 1, padding='same', activation='sigmoid')(f)
    return Multiply()([skip, psi])

# Down Block
def down_block(x, filters, kernel_size=4, strides=2, norm=NormLayer):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = norm()(x)
    x = LeakyReLU(0.2)(x)
    x = res_se_block(x, filters, norm=norm)
    return x

# Up Block with Attention
def up_block(x, skip, filters, kernel_size=4, strides=2, norm=NormLayer, use_dropout=False, attention=True):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = norm()(x)
    x = Activation('relu')(x)
    x = res_se_block(x, filters, norm=norm)

    if use_dropout:
        x = Dropout(0.3)(x)

    if attention:
        refined_skip = attention_gate(skip, x, inter_channels=max(filters//2, 16))
        x = Concatenate()([x, refined_skip])
    else:
        x = Concatenate()([x, skip])

    return x

# Build Medical ResSE-UNet Generator
def build_generator(img_shape=(224,224,3), base_filters=64, use_attention=True, use_dropout_in_decoder=True):
    inputs = Input(img_shape)

    # Encoder
    e1 = down_block(inputs, base_filters)
    e2 = down_block(e1, base_filters*2)
    e3 = down_block(e2, base_filters*4)
    e4 = down_block(e3, base_filters*8)

    # Bottleneck
    b = Conv2D(base_filters*8, 4, strides=2, padding='same', use_bias=False)(e4)
    b = NormLayer()(b)
    b = Activation('relu')(b)
    b = res_se_block(b, base_filters*8, norm=NormLayer)

    # Decoder
    d1 = up_block(b, e4, base_filters*8, use_dropout=use_dropout_in_decoder, attention=use_attention)
    d2 = up_block(d1, e3, base_filters*4, use_dropout=use_dropout_in_decoder, attention=use_attention)
    d3 = up_block(d2, e2, base_filters*2, use_dropout=use_dropout_in_decoder, attention=use_attention)
    d4 = up_block(d3, e1, base_filters, use_dropout=use_dropout_in_decoder, attention=use_attention)

    outputs = Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')(d4)

    model = Model(inputs, outputs, name="Med_ResSE_UNet_Generator")
    return model

# Self-Attention Layer
class SelfAttention(Layer):
    def __init__(self, filters):
        super(SelfAttention, self).__init__()
        self.filters = filters
        self.f_conv = Conv2D(filters//8, 1, padding='same')
        self.g_conv = Conv2D(filters//8, 1, padding='same')
        self.h_conv = Conv2D(filters, 1, padding='same')
        self.v_conv = Conv2D(filters, 1, padding='same')
        self.softmax = Softmax(axis=-1)

    def call(self, x):
        batch_size, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        f = self.f_conv(x)
        g = self.g_conv(x)
        h_ = self.h_conv(x)

        f_flat = tf.reshape(f, [batch_size, -1, self.filters//8])
        g_flat = tf.reshape(g, [batch_size, -1, self.filters//8])
        h_flat = tf.reshape(h_, [batch_size, -1, self.filters])

        s = tf.matmul(g_flat, f_flat, transpose_b=True)
        beta = self.softmax(s)

        o = tf.matmul(beta, h_flat)
        o = tf.reshape(o, [batch_size, h, w, c])
        o = self.v_conv(o)
        return Add()([x, o])

# Residual block for Discriminator
def residual_block(x, filters, stride=1):
    shortcut = x
    x = Conv2D(filters, 4, strides=stride, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters, 4, strides=1, padding='same')(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
    x = Add()([x, shortcut])
    x = LeakyReLU(0.2)(x)
    return x

# Build Attention Discriminator
def build_discriminator(img_shape=(224,224,3)):
    inp = Input(img_shape)
    x = Conv2D(64, 4, strides=2, padding='same')(inp)
    x = LeakyReLU(0.2)(x)
    x = residual_block(x, 128, stride=2)
    x = Dropout(0.3)(x)
    x = residual_block(x, 256, stride=2)
    x = SelfAttention(256)(x)
    x = Dropout(0.3)(x)
    x = residual_block(x, 512, stride=2)
    x = Dropout(0.3)(x)
    x = Conv2D(1, 4, padding='same')(x)
    outputs = Activation('sigmoid')(x)
    return Model(inp, outputs, name="Attention_Discriminator")

# ==========================================================
# BUILD ViT CLASSIFIER
# ==========================================================
print("\n" + "="*60)
print("BUILDING ViT CLASSIFIER FROM SCRATCH")
print("="*60 + "\n")

from transformers import TFViTModel, ViTConfig

print("Attempting to load ViT (try TF weights, then convert from PyTorch)...")
try:
    # Prefer native TF weights if present
    vit_base = TFViTModel.from_pretrained(VIT_MODEL_NAME, from_pt=False)
    print("✅ Loaded native TensorFlow ViT weights")
except Exception as e_tf:
    print("Native TF weights not available or failed. Trying to load PyTorch weights and convert to TF...")
    print("Error (TF attempt):", e_tf)
    try:
        # This converts PyTorch weights -> TF. Requires `torch` installed.
        vit_base = TFViTModel.from_pretrained(VIT_MODEL_NAME, from_pt=True)
        print("✅ Loaded PyTorch weights and converted to TF (from_pt=True)")
    except Exception as e_pt:
        print("Failed to load PyTorch weights too:", e_pt)
        print("Falling back to building ViT from config (random init).")
        config = ViTConfig.from_pretrained(VIT_MODEL_NAME)
        vit_base = TFViTModel(config)  # random init
        # build (call) once as you did before to instantiate variables
        dummy = tf.zeros((1, 3, IMG_SIZE, IMG_SIZE))
        _ = vit_base(pixel_values=dummy, training=False)
        print("⚠️ ViT created from config (randomly initialized).")


# Freeze ViT initially
for layer in vit_base.layers:
    layer.trainable = False

print("✅ ViT base model loaded and frozen")

# ViT forward function
def vit_forward(x):
    """
    Forward pass through ViT
    Input: (B, H, W, C) - already preprocessed by vit_processor
    Output: (B, 768) - CLS token features
    """
    # Transpose to (B, C, H, W)
    x = tf.transpose(x, [0, 3, 1, 2])
    
    # Forward pass
    outputs = vit_base(pixel_values=x, training=False)
    return outputs.last_hidden_state[:, 0, :]

# Build classifier model
print("Building ViT classifier head...")

