"""
scripts/deep_diagnostic.py
Answers three questions in one run:

  Q1. What does the trained model predict for TILT_LEFT samples?
      -> If it always predicts one class, the model never saw TILT_LEFT
         during training (train/test split stratification failed).

  Q2. What are the model's output probabilities for TILT_LEFT samples?
      -> If TILT_LEFT prob is always < 0.1, the output neuron is dead.

  Q3. What does the model predict for a hand-crafted perfect TILT_LEFT?
      -> Bypasses all data pipeline issues. If this also fails -> model bug.
         If this succeeds -> the test split has wrong labels.

Usage:
    python scripts/deep_diagnostic.py \
        --data data/raw/synthetic_dataset.csv \
        --model models/bilstm_best.keras \
        --stats models/normaliser_stats.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ml.feature_engineering import CLASSES, DatasetBuilder, N_FEAT, WINDOW

TILT_L = CLASSES.index("TILT_LEFT")
TILT_R = CLASSES.index("TILT_RIGHT")


def split_dataset(X, y, val_frac=0.15, test_frac=0.15, seed=42):
    from sklearn.model_selection import train_test_split

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=seed
    )
    val_size = val_frac / (1 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_size, stratify=y_tv, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def q1_q2(model, X_test, y_test):
    print("\n" + "=" * 60)
    print("Q1+Q2 -- Model predictions on actual test split")
    print("=" * 60)

    print("\n  Test split class counts:")
    for i, cls in enumerate(CLASSES):
        n = int((y_test == i).sum())
        print(f"    index {i} -> {cls:<14}  n={n}")

    tl_mask = y_test == TILT_L
    if tl_mask.sum() == 0:
        print("\n  [FAIL] TILT_LEFT has zero samples in the test split.")
        print("     The split or label mapping is broken.")
        return False

    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)

    print(f"\n  TILT_LEFT samples in test split: {int(tl_mask.sum())}")
    print("\n  Predicted class distribution for true TILT_LEFT samples:")
    for i, cls in enumerate(CLASSES):
        n = int((preds[tl_mask] == i).sum())
        bar = "#" * min(n, 40)
        print(f"    -> {cls:<14}  {n:3d}  {bar}")

    print("\n  Mean output probabilities for true TILT_LEFT samples:")
    mean_probs = probs[tl_mask].mean(axis=0)
    for i, cls in enumerate(CLASSES):
        marker = " <- TILT_LEFT neuron" if i == TILT_L else ""
        print(f"    P({cls:<14}) = {mean_probs[i]:.4f}{marker}")

    if mean_probs[TILT_L] < 0.05:
        print(f"\n  [FAIL] TILT_LEFT output neuron is essentially dead (mean prob={mean_probs[TILT_L]:.4f})")
        print("     The model did not learn to activate this class.")
        return False

    print(f"\n  [PASS] TILT_LEFT neuron is alive (mean prob={mean_probs[TILT_L]:.4f})")
    return True


def q3_perfect_sample(model, builder):
    print("\n" + "=" * 60)
    print("Q3 -- Perfect synthetic TILT_LEFT fed directly to model")
    print("     (bypasses CSV, feature_engineering, split -- pure model test)")
    print("=" * 60)

    roll = np.full(WINDOW, -25.0, dtype=np.float32)
    yaw = np.zeros(WINDOW, dtype=np.float32)
    pitch = np.zeros(WINDOW, dtype=np.float32)
    dyaw = np.zeros(WINDOW, dtype=np.float32)
    dpitch = np.zeros(WINDOW, dtype=np.float32)
    droll = np.zeros(WINDOW, dtype=np.float32)

    roll_sign = np.full(WINDOW, -1.0, dtype=np.float32)
    roll_raw = np.full(WINDOW, -25.0, dtype=np.float32)

    feat6 = np.stack([yaw, pitch, roll, dyaw, dpitch, droll], axis=1)
    feat6_norm = (feat6 - builder._mean) / builder._std

    if N_FEAT == 8:
        feat = np.concatenate(
            [
                feat6_norm,
                roll_sign[:, np.newaxis],
                roll_raw[:, np.newaxis],
            ],
            axis=1,
        )
    elif N_FEAT == 7:
        feat = np.concatenate([feat6_norm, roll_sign[:, np.newaxis]], axis=1)
    else:
        feat = feat6_norm

    x_in = feat[np.newaxis].astype(np.float32)

    print(f"\n  Input shape fed to model: {x_in.shape}")
    print(f"  Feature 2 (roll norm):  {x_in[0, 0, 2]:.4f}  (should be strongly negative)")
    if N_FEAT >= 8:
        print(f"  Feature 6 (roll_sign):  {x_in[0, 0, 6]:.4f}  (should be -1.0)")
        print(f"  Feature 7 (roll_raw):   {x_in[0, 0, 7]:.4f}  (should be -25.0)")

    probs = model.predict(x_in, verbose=0)[0]
    pred = CLASSES[np.argmax(probs)]

    print("\n  Model output probabilities:")
    for i, cls in enumerate(CLASSES):
        bar = "#" * min(int(probs[i] * 40), 40)
        marker = " <-" if i == TILT_L else ""
        print(f"    P({cls:<14}) = {probs[i]:.4f}  {bar}{marker}")

    print(f"\n  Predicted: {pred}")

    if pred == "TILT_LEFT":
        print("  [PASS] Model does recognise perfect TILT_LEFT")
        print("     -> Problem is likely in test data labels or split, not the model weights.")
        return True

    print(f"  [FAIL] Model predicts {pred} even for a textbook TILT_LEFT")
    print("     -> The model weights are wrong, or TILT_LEFT was not learned.")
    return False


def q4_print_mapping(builder):
    print("\n" + "=" * 60)
    print("Q4 -- LabelEncoder class mapping (what the model was trained with)")
    print("=" * 60)
    print(f"\n  N_FEAT = {N_FEAT}")
    print("\n  CLASSES list order (index -> label):")
    for i, cls in enumerate(builder.classes):
        match = "[PASS]" if cls == CLASSES[i] else f"[FAIL] expected {CLASSES[i]}"
        print(f"    {i} -> {cls:<14}  {match}")

    if list(builder.classes) != CLASSES:
        print("\n  [FAIL] Mismatch between saved encoder and current CLASSES list.")
        print(f"     Saved:   {list(builder.classes)}")
        print(f"     Current: {CLASSES}")
        return False

    print("\n  [PASS] Label mapping is consistent")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stats", required=True)
    args = parser.parse_args()

    import tensorflow as tf

    print(f"[INFO] Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model, safe_mode=False)
    print(f"[INFO] Model input shape:  {model.input_shape}")
    print(f"[INFO] Model output shape: {model.output_shape}")

    expected_feat = model.input_shape[-1]
    if expected_feat != N_FEAT:
        print(
            f"\n  [FAIL] Shape mismatch: model expects {expected_feat} features but N_FEAT={N_FEAT}"
        )
        print("     Delete the model artifact and retrain with the current feature pipeline.")
        sys.exit(1)

    builder = DatasetBuilder.load_stats(args.stats)
    q4_print_mapping(builder)

    builder2 = DatasetBuilder()
    X, y = builder2.fit_transform(args.data)
    _, _, X_te, _, _, y_te = split_dataset(X, y, seed=42)

    q1_ok = q1_q2(model, X_te, y_te)
    q3_ok = q3_perfect_sample(model, builder)

    print("\n" + "=" * 60)
    print("SUMMARY -- what to do next")
    print("=" * 60)
    if q3_ok:
        print("  Q3 passed: the model recognises a textbook TILT_LEFT.")
        print("  The remaining issue is likely in the split or label mapping.")
    else:
        print("  Q3 failed: the model itself is not learning TILT_LEFT.")
        print("  Check training-split counts and model capacity before retraining.")

    if q1_ok and q3_ok:
        print("  Next: inspect train/test split class counts in the training script.")


if __name__ == "__main__":
    main()
