"""
scripts/comprehensive_test.py
Compute F1 scores and measure live inference latency on the newly trained model.
"""

import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.feature_engineering import DatasetBuilder

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  COMPREHENSIVE TEST: F1 Scores + Latency Measurement")
print("="*70)

# Load dataset and builder
print("\n[1] Loading dataset...")
builder = DatasetBuilder()
X, y = builder.fit_transform("data/raw/synthetic_dataset.csv")
print(f"    Loaded {X.shape[0]} samples, {X.shape[1]} features, {X.shape[2]} frames")

# Split: 70% train, 15% val, 15% test
n_total = len(X)
n_train = int(0.70 * n_total)
n_val   = int(0.15 * n_total)

X_train = X[:n_train]
y_train = y[:n_train]

X_val   = X[n_train:n_train+n_val]
y_val   = y[n_train:n_train+n_val]

X_test  = X[n_train+n_val:]
y_test  = y[n_train+n_val:]

print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Load model
print("\n[2] Loading model...")
model = tf.keras.models.load_model("models/bilstm_best.keras")
print(f"    Model loaded: {model.name}")

# Get normalizer stats
print("\n[3] Loading normalizer stats...")
stats = np.load("models/normaliser_stats.npz")
mean = stats["mean"]
std = stats["std"]
print(f"    Mean shape: {mean.shape}")
print(f"    Std shape: {std.shape}")

# Normalize
X_test_norm = (X_test - mean) / (std + 1e-8)

# ─────────────────────────────────────────────────────────────────────────────
# Compute predictions and F1 scores
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4] Computing predictions (test set: {} samples)...".format(len(X_test)))
y_pred = model.predict(X_test_norm, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_labels)
macro_f1 = f1_score(y_test, y_pred_labels, average="macro")
weighted_f1 = f1_score(y_test, y_pred_labels, average="weighted")

print(f"\n[5] Results:")
print(f"    Accuracy:    {accuracy:.4f}")
print(f"    Macro F1:    {macro_f1:.4f} {'✓' if macro_f1 >= 0.92 else '✗ (target: ≥0.92)'}")
print(f"    Weighted F1: {weighted_f1:.4f}")

# Per-class
print(f"\n[6] Per-class F1 scores:")
classes = builder.le.classes_
for i, cls_name in enumerate(classes):
    mask = y_test == i
    if mask.sum() == 0:
        continue
    y_test_i = (y_test == i).astype(int)
    y_pred_i = (y_pred_labels == i).astype(int)
    f1_i = f1_score(y_test_i, y_pred_i)
    support_i = mask.sum()
    print(f"    {cls_name:12s}  F1={f1_i:.4f}  support={support_i}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)
print(f"\n[7] Confusion Matrix:")
print("Predicted:    " + "  ".join(f"{c:>6s}" for c in classes))
for i, cls_name in enumerate(classes):
    print(f"True {cls_name:6s}:  " + "  ".join(f"{cm[i, j]:>6d}" for j in range(len(classes))))

# ─────────────────────────────────────────────────────────────────────────────
# Latency measurement (Keras)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[8] Latency measurement (Keras, 100 inferences):")
latencies = []
for _ in range(100):
    sample = X_test_norm[0:1]  # single batch
    t0 = time.perf_counter()
    _ = model.predict(sample, verbose=0)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)  # ms

latencies = np.array(latencies)
mean_latency = latencies.mean()
min_latency  = latencies.min()
max_latency  = latencies.max()
p95_latency  = np.percentile(latencies, 95)

print(f"    Mean: {mean_latency:.2f} ms  {'✓' if mean_latency < 15 else '✗ (target: <15ms)'}")
print(f"    Min:  {min_latency:.2f} ms")
print(f"    Max:  {max_latency:.2f} ms")
print(f"    P95:  {p95_latency:.2f} ms")

print("\n" + "="*70)
print("  TEST COMPLETE")
print("="*70 + "\n")
