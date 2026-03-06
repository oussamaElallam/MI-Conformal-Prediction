"""
Model architectures for BSPC experiments.

Existing:
  - Lightweight CNN (paper primary model)
  - ResNet1D (paper baseline)

New (Reviewer 2, Point 2):
  - Depthwise Separable CNN (DSC-1D) — efficient alternative
  - Temporal Convolutional Network (TCN-1D) — dilated causal convolutions
  - MiniResNet1D — smaller residual network for edge comparison

Ablation variants (Reviewer 2, Point 3):
  - lightweight_cnn_ablation(kernel_size, n_filters_1, n_filters_2, n_blocks)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, GlobalAveragePooling1D,
    BatchNormalization, Activation, Add, Dropout,
    SeparableConv1D, LayerNormalization
)
from tensorflow.keras.metrics import AUC


def _compile(model):
    """Standard compilation for all models."""
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    return model


# ─────────────────────────────────────────────
# Paper models (unchanged)
# ─────────────────────────────────────────────

def create_lightweight_cnn(input_shape):
    """Paper primary model: ~50K params, 2 conv blocks + FC."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 5, activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return _compile(model)


def create_resnet1d(input_shape):
    """Paper baseline: ~200K params, 4 residual blocks."""
    def res_block(x, filters, kernel_size, stride=1, downsample=False):
        shortcut = x
        y = Conv1D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)(y)
        y = BatchNormalization()(y)
        if downsample or shortcut.shape[-1] != filters:
            shortcut = Conv1D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
            shortcut = BatchNormalization()(shortcut)
        out = Add()([shortcut, y])
        out = Activation('relu')(out)
        return out

    inputs = Input(shape=input_shape)
    x = Conv1D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = res_block(x, 64, 3)
    x = res_block(x, 64, 3)
    x = res_block(x, 128, 3, stride=2, downsample=True)
    x = res_block(x, 128, 3)
    x = res_block(x, 256, 3, stride=2, downsample=True)
    x = res_block(x, 256, 3)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return _compile(model)


# ─────────────────────────────────────────────
# New lightweight architectures (R2-2)
# ─────────────────────────────────────────────

def create_depthwise_separable_cnn(input_shape):
    """
    Depthwise Separable 1D CNN — replaces standard convolutions with
    depthwise separable ones for parameter efficiency.
    Comparable capacity to Lightweight CNN with fewer parameters.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 5, padding='same', use_bias=False),  # initial standard conv
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(2),

        SeparableConv1D(64, 5, padding='same', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(2),

        SeparableConv1D(128, 5, padding='same', use_bias=False),
        BatchNormalization(),
        Activation('relu'),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    return _compile(model)


def create_tcn(input_shape):
    """
    Temporal Convolutional Network (TCN-1D) — uses dilated causal
    convolutions for capturing long-range temporal dependencies
    in ECG signals. Lightweight variant suitable for edge deployment.
    """
    inputs = Input(shape=input_shape)

    # TCN blocks with increasing dilation
    x = inputs
    n_filters = 32
    for dilation_rate in [1, 2, 4, 8]:
        residual = x
        # Dilated causal conv block
        y = Conv1D(n_filters, 3, padding='causal', dilation_rate=dilation_rate, use_bias=False)(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Dropout(0.1)(y)
        y = Conv1D(n_filters, 3, padding='causal', dilation_rate=dilation_rate, use_bias=False)(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        # Residual connection
        if residual.shape[-1] != n_filters:
            residual = Conv1D(n_filters, 1, padding='same', use_bias=False)(residual)
        x = Add()([residual, y])

    # Second stage with more filters
    n_filters = 64
    for dilation_rate in [1, 2, 4]:
        residual = x
        y = Conv1D(n_filters, 3, padding='causal', dilation_rate=dilation_rate, use_bias=False)(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Dropout(0.1)(y)
        y = Conv1D(n_filters, 3, padding='causal', dilation_rate=dilation_rate, use_bias=False)(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        if residual.shape[-1] != n_filters:
            residual = Conv1D(n_filters, 1, padding='same', use_bias=False)(residual)
        x = Add()([residual, y])

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return _compile(model)


def create_mini_resnet(input_shape):
    """
    MiniResNet1D — smaller residual network (2 blocks instead of 6),
    designed to be comparable in size to the Lightweight CNN while
    using residual connections and batch normalization.
    """
    def res_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        y = Conv1D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv1D(filters, kernel_size, padding='same', use_bias=False)(y)
        y = BatchNormalization()(y)
        if shortcut.shape[-1] != filters or stride > 1:
            shortcut = Conv1D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
            shortcut = BatchNormalization()(shortcut)
        return Activation('relu')(Add()([shortcut, y]))

    inputs = Input(shape=input_shape)
    x = Conv1D(32, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2, padding='same')(x)

    x = res_block(x, 32)
    x = res_block(x, 64, stride=2)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return _compile(model)


# ─────────────────────────────────────────────
# Ablation factory (R2-3)
# ─────────────────────────────────────────────

def create_lightweight_cnn_ablation(input_shape,
                                     kernel_size=5,
                                     n_filters_1=32,
                                     n_filters_2=64,
                                     n_blocks=2,
                                     fc_units=64):
    """
    Parameterized Lightweight CNN for architectural sensitivity analysis.

    Args:
        kernel_size:  Convolutional kernel size (default 5, vary: 3, 7, 9)
        n_filters_1:  Filters in first conv block (default 32)
        n_filters_2:  Filters in second conv block (default 64)
        n_blocks:     Number of conv blocks (1, 2, or 3)
        fc_units:     Dense layer units (default 64)
    """
    layers = [Input(shape=input_shape)]

    filter_schedule = [n_filters_1, n_filters_2] + [n_filters_2 * 2] * (n_blocks - 2)
    for i in range(n_blocks):
        n_filt = filter_schedule[i] if i < len(filter_schedule) else n_filters_2
        layers.append(Conv1D(n_filt, kernel_size, activation='relu'))
        layers.append(MaxPooling1D(2))

    layers.extend([
        Flatten(),
        Dense(fc_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model = Sequential(layers)
    return _compile(model)


# ─────────────────────────────────────────────
# Registry for convenience
# ─────────────────────────────────────────────

MODEL_REGISTRY = {
    'LightweightCNN': create_lightweight_cnn,
    'ResNet1D': create_resnet1d,
    'DSC_CNN': create_depthwise_separable_cnn,
    'TCN': create_tcn,
    'MiniResNet': create_mini_resnet,
}
