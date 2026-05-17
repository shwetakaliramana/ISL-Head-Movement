"""
src/ml/bilstm_model.py
─────────────────────────────────────────────────────────────────────────────
BiLSTM gesture classifier — fixed for TILT_LEFT/TILT_RIGHT discrimination.

WHY THE ORIGINAL MODEL FAILED
──────────────────────────────
Three compounding problems:

1. Symmetric normalisation destroyed roll sign (fixed in feature_engineering.py
   via explicit roll_sign and roll_abs channels).

2. L2 weight decay + high dropout penalised the small sustained DC offset
   of the roll feature, causing the model to ignore it.

3. The LSTM hidden state at t=0 starts at zero.  For a sustained TILT gesture
   (30 frames of constant roll=-25°) the LSTM sees the same input every step
   and its output barely changes from the initial state — effectively a very
   weak signal for the dense classifier.

THE FIX
───────
Architecture changes:
  a. Separate 1-D conv "tilt extractor" branch: Conv1D(32, 1) applied only
     to channels [roll_norm, roll_sign, roll_abs].  A 1×1 conv can learn
     "roll_sign == -1 → fire for TILT_LEFT" trivially, bypassing the LSTM.
     Its output is concatenated with the BiLSTM output before the dense head.

  b. No L2 decay on the tilt branch (it needs to hold a sustained signal).
     L2 is kept only on the LSTM gates where it prevents overfitting on
     NOD/SHAKE oscillation patterns.

  c. Class weights passed to model.fit(): TILT classes get weight 2.0 so
     their loss contribution matches the higher NOD/SHAKE signal strength.

  d. Focal loss optional wrapper: reduces confidence on easy STATIC samples
     so the gradient focuses on the harder TILT classes.

Input shape:  (batch, 30, 8)
Output shape: (batch, 5)    — softmax probabilities over CLASSES
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

NUM_CLASSES  = cfg.model.num_classes    # 5
LSTM_UNITS   = cfg.model.lstm_units     # [128, 64]
DENSE_UNITS  = cfg.model.dense_units    # 64
DROPOUT      = cfg.model.dropout        # 0.3
N_FEAT       = 8                        # must match feature_engineering.N_FEAT

# Indices of the roll-related channels in the feature vector
ROLL_CHANNELS = [2, 6, 7]   # roll_norm, roll_sign, roll_abs


def build_model(
    window:      int = 30,
    n_feat:      int = N_FEAT,
    num_classes: int = NUM_CLASSES,
    lstm_units:  list[int] = None,
    dense_units: int = DENSE_UNITS,
    dropout:     float = DROPOUT,
) -> keras.Model:
    """
    Build the dual-branch BiLSTM + Conv tilt extractor model.

    Architecture:
        Input (30, 8)
            │
            ├── [all 8 features] ──► BiLSTM(128, return_sequences=True)
            │                       ► Dropout(0.3)
            │                       ► BiLSTM(64)
            │                       ► Dropout(0.3)
            │                       ► [128-dim LSTM repr]
            │
            └── [channels 2,6,7] ── Conv1D(32, kernel=1, relu)
                                   GlobalAvgPool1D
                                   ► [32-dim tilt repr]
            │
            Concatenate([128, 32]) = 160-dim
            │
            Dense(64, relu)
            Dropout(0.2)
            Dense(5, softmax)
    """
    if lstm_units is None:
        lstm_units = LSTM_UNITS

    inp = keras.Input(shape=(window, n_feat), name="angle_sequence")

    # ── Branch 1: BiLSTM on all features ──────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units[0],
            return_sequences=True,
            dropout=dropout * 0.5,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        name="bilstm_1",
    )(inp)
    x = layers.Dropout(dropout, name="drop_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units[1],
            return_sequences=False,
            dropout=dropout * 0.5,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        name="bilstm_2",
    )(x)
    x = layers.Dropout(dropout, name="drop_2")(x)   # shape: (batch, 128)

    # ── Branch 2: Conv tilt extractor on roll channels only ───────────────────
    roll_inp = layers.Lambda(
        lambda t: tf.gather(t, ROLL_CHANNELS, axis=2),
        name="roll_channel_slice",
    )(inp)                                            # (batch, 30, 3)

    t = layers.Conv1D(
        32, kernel_size=1, activation="relu",
        name="tilt_conv",
    )(roll_inp)                                       # (batch, 30, 32)

    t = layers.Conv1D(
        32, kernel_size=3, activation="relu",
        padding="same", name="tilt_conv2",
    )(t)                                              # (batch, 30, 32)

    t = layers.GlobalAveragePooling1D(name="tilt_gap")(t)   # (batch, 32)

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = layers.Concatenate(name="merge")([x, t])  # (batch, 160)

    d = layers.Dense(dense_units, activation="relu",
                     name="dense_1")(merged)
    d = layers.Dropout(dropout * 0.67, name="drop_3")(d)
    out = layers.Dense(num_classes, activation="softmax",
                       name="output")(d)

    model = keras.Model(inputs=inp, outputs=out, name="ISL_BiLSTM_TiltFix")
    log.info("Model built: %d params", model.count_params())
    return model


def get_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """
    Compute class weights so each class contributes equally to the loss.
    TILT classes get an additional 2× boost to overcome their weaker signal.

    Args:
        y_train: integer label array for the training split.
    Returns:
        dict mapping class_index → weight for keras model.fit(class_weight=...).
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}

    # TILT_LEFT=2, TILT_RIGHT=3 — extra boost
    tilt_indices = [2, 3]
    for idx in tilt_indices:
        if idx in cw:
            cw[idx] *= 2.0
            log.info("TILT class %d weight boosted to %.3f", idx, cw[idx])

    log.info("Class weights: %s", cw)
    return cw


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal loss — down-weights easy examples (STATIC) so gradient focuses
    on hard ones (TILT_LEFT, TILT_RIGHT).

    Use as: model.compile(loss=focal_loss(gamma=2.0))
    """
    def loss_fn(y_true, y_pred):
        y_pred   = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce       = -tf.reduce_sum(
            tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), 5), tf.float32)
            * tf.math.log(y_pred),
            axis=-1,
        )
        pt       = tf.reduce_sum(
            tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), 5), tf.float32)
            * y_pred,
            axis=-1,
        )
        focal_w  = alpha * tf.pow(1.0 - pt, gamma)
        return tf.reduce_mean(focal_w * ce)

    return loss_fn
