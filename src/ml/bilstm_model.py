"""
src/ml/bilstm_model.py
BiLSTM model definition for ISL head movement classification.
"""

from __future__ import annotations


def build_bilstm(window: int = 30,
                 n_features: int = 8,
                 n_classes: int = 5,
                 lstm_units: tuple[int, int] = (64, 32),
                 dense_units: int = 64,
                 dropout_lstm: float = 0.3,
                 dropout_dense: float = 0.2,
                 learning_rate: float = 1e-3):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units[0], return_sequences=True)
    )(inputs)
    x = tf.keras.layers.Dropout(dropout_lstm)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units[1])
    )(x)
    x = tf.keras.layers.Dropout(dropout_lstm)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_dense)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="bilstm_gesture_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
"""
src/ml/bilstm_model.py
BiLSTM model definition for ISL head movement classification.

Architecture:
    Input (30, 8)
    → BiLSTM(64)  + dropout(0.3)
    → BiLSTM(32)  + dropout(0.3)
    → Dense(64, relu) + dropout(0.2)
    → Dense(5, softmax)

~180K parameters — fast inference (<15ms on CPU).
"""

import numpy as np

# ── lazy TF import so the file is importable without GPU ──────────────────────
def _tf():
    import tensorflow as tf
    return tf

def _keras():
    import tensorflow as tf
    return tf.keras


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss for imbalanced classification (optional, not used by default)."""
    import tensorflow as tf
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        ce_loss = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        focal_loss_val = alpha * focal_weight * ce_loss
        return tf.reduce_mean(tf.reduce_sum(focal_loss_val, axis=-1))
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_bilstm(window: int = 30,
                 n_features: int = 8,
                 n_classes: int = 5,
                 lstm_units: tuple = (64, 32),
                 dense_units: int = 64,
                 dropout_lstm: float = 0.3,
                 dropout_dense: float = 0.2,
                 learning_rate: float = 1e-3) -> "tf.keras.Model":
    """
    Simple BiLSTM without roll-gating (removes Lambda layer issues).

    Architecture:
        Input (30, 8)
        → BiLSTM(64) + dropout(0.3)
        → BiLSTM(32) + dropout(0.3)
        → Dense(64, relu) + dropout(0.2)
        → Dense(5, softmax)
    """
    tf = _tf()
    keras = _keras()

    inputs = keras.Input(shape=(window, n_features), name="angle_sequence")

    # ── BiLSTM stack ─────────────────────────────────────────────────────────
    x = inputs
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(units,
                              return_sequences=return_seq,
                              kernel_regularizer=keras.regularizers.l2(1e-4)),
            name=f"bilstm_{i+1}"
        )(x)
        x = keras.layers.Dropout(dropout_lstm, name=f"drop_lstm_{i+1}")(x)

    # ── Dense layers ─────────────────────────────────────────────────────────
    x = keras.layers.Dense(dense_units, activation="relu",
                           name="dense_hidden")(x)
    x = keras.layers.Dropout(dropout_dense, name="drop_dense")(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax",
                                 name="output")(x)

    model = keras.Model(inputs, outputs, name="ISL_BiLSTM_Simple")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest baseline (for ablation in evaluate_full_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_rf_features(X: np.ndarray) -> np.ndarray:
    """
    Flatten (N, window, n_feat) into hand-crafted (N, n_rf_feat) for RF.
    Features: per-channel mean, std, min, max, range = 5 stats × 8 channels = 40.
    """
    mean  = X.mean(axis=1)          # (N, 8)
    std   = X.std(axis=1)           # (N, 8)
    mn    = X.min(axis=1)           # (N, 8)
    mx    = X.max(axis=1)           # (N, 8)
    rng   = mx - mn                 # (N, 8)
    return np.concatenate([mean, std, mn, mx, rng], axis=1)  # (N, 40)


def build_rf_baseline(random_state: int = 42):
    """Return an unfitted RandomForestClassifier with tuned hyperparameters."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility shims for older scripts expecting different names
# ─────────────────────────────────────────────────────────────────────────────
def build_model(*args, **kwargs):
    """Legacy alias: calls `build_bilstm` to preserve old imports."""
    return build_bilstm(*args, **kwargs)


def get_class_weights(y: np.ndarray, n_classes: int = 5) -> dict:
    """Compute balanced class weights as a dict {int: float}.

    Mirrors the behaviour used in `scripts/train_bilstm.py`.
    """
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight("balanced",
                                   classes=np.arange(n_classes),
                                   y=y)
    return {i: float(w) for i, w in enumerate(weights)}


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_bilstm()
    model.summary()
    dummy = np.zeros((4, 30, 8), dtype=np.float32)
    out   = model.predict(dummy, verbose=0)
    print(f"Output shape: {out.shape}")   # (4, 5)
    print(f"Sum (should be ~1): {out.sum(axis=1)}")