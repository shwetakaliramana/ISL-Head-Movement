#!/usr/bin/env python3
"""Analyze feature separability for all classes, focusing on TILT_LEFT."""

import numpy as np
import pandas as pd
from src.ml.feature_engineering import DatasetBuilder

# Load raw dataset
df = pd.read_csv('data/raw/synthetic_dataset.csv')

# Separate by class
classes = ['NOD', 'SHAKE', 'TILT_LEFT', 'TILT_RIGHT', 'STATIC']
class_samples = {cls: df[df['label'] == cls] for cls in classes}

# Build features
builder = DatasetBuilder()
X, y = builder.fit_transform('data/raw/synthetic_dataset.csv')

print(f"Total samples: {len(X)}")
print(f"Feature shape per sample: {X[0].shape}")
print(f"Number of features: {X.shape[2]}")
print()

# Analyze feature distributions per class
for cls_idx, cls_name in enumerate(classes):
    mask = y == cls_idx
    X_cls = X[mask]
    print(f"\n{'='*60}")
    print(f"Class: {cls_name} (label={cls_idx}, n={mask.sum()})")
    print(f"{'='*60}")
    
    # Flatten all frames for this class
    X_cls_flat = X_cls.reshape(-1, X_cls.shape[2])  # (n_samples*30, n_features)
    
    # Show mean, std, min, max for each feature
    feature_names = ['yaw', 'pitch', 'roll', 'dyaw', 'dpitch', 'droll', 'roll_sign_smoothed', 'roll_mean']
    
    for feat_idx in range(X_cls.shape[2]):
        feat_vals = X_cls_flat[:, feat_idx]
        print(f"  Feature {feat_idx} ({feature_names[feat_idx] if feat_idx < len(feature_names) else '?'}):")
        print(f"    Mean: {feat_vals.mean():8.3f}  Std: {feat_vals.std():8.3f}  Min: {feat_vals.min():8.3f}  Max: {feat_vals.max():8.3f}")

# Check roll statistics specifically for each class
print(f"\n{'='*60}")
print("ROLL COLUMN FOCUS (Feature 2, raw roll angle)")
print(f"{'='*60}")
for cls_idx, cls_name in enumerate(classes):
    mask = y == cls_idx
    X_cls = X[mask]
    roll_vals = X_cls[:, :, 2].flatten()  # All frames, roll column
    print(f"{cls_name:12s}: mean={roll_vals.mean():7.2f}  std={roll_vals.std():7.2f}  min={roll_vals.min():7.2f}  max={roll_vals.max():7.2f}")

# Check roll_mean (feature 7)
print(f"\n{'='*60}")
print("ROLL_MEAN COLUMN FOCUS (Feature 7, per-sample broadcast)")
print(f"{'='*60}")
for cls_idx, cls_name in enumerate(classes):
    mask = y == cls_idx
    X_cls = X[mask]
    roll_mean_vals = X_cls[:, 0, 7]  # First frame is enough (all frames same for broadcast)
    print(f"{cls_name:12s}: mean={roll_mean_vals.mean():7.2f}  std={roll_mean_vals.std():7.2f}  min={roll_mean_vals.min():7.2f}  max={roll_mean_vals.max():7.2f}")

# Check raw CSV roll values for TILT_LEFT
print(f"\n{'='*60}")
print("RAW CSV ROLL VALUES - TILT_LEFT class only")
print(f"{'='*60}")
tilt_left_csv = class_samples['TILT_LEFT']
print(f"Number of samples: {len(tilt_left_csv)}")
print(f"Roll column stats:")
roll_raw = tilt_left_csv['roll'].values
print(f"  Mean: {roll_raw.mean():.2f}")
print(f"  Std:  {roll_raw.std():.2f}")
print(f"  Min:  {roll_raw.min():.2f}")
print(f"  Max:  {roll_raw.max():.2f}")
print(f"  Negative roll count: {(roll_raw < 0).sum()}")
print(f"  Positive roll count: {(roll_raw >= 0).sum()}")
print(f"  Values < -12: {(roll_raw < -12).sum()}")
print(f"  Values >= 12: {(roll_raw >= 12).sum()}")

# Compare with NOD (should have roll ~ 0)
print(f"\nRaw CSV roll values - NOD class (for comparison):")
nod_csv = class_samples['NOD']
roll_nod = nod_csv['roll'].values
print(f"  Mean: {roll_nod.mean():.2f}")
print(f"  Std:  {roll_nod.std():.2f}")
print(f"  Min:  {roll_nod.min():.2f}")
print(f"  Max:  {roll_nod.max():.2f}")
print(f"  Values in (-12, 12): {((roll_nod >= -12) & (roll_nod < 12)).sum()}")
