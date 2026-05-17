#!/usr/bin/env python3
"""Debug: trace raw TILT_LEFT generation without normalization."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

# Load raw CSV (before normalization)
df = pd.read_csv('data/raw/synthetic_dataset.csv')

# Get TILT_LEFT samples (only first 5 for inspection)
tilt_left_samples = df[df['label'] == 'TILT_LEFT']['sample_id'].unique()[:5]

for sample_id in tilt_left_samples:
    sample_df = df[df['sample_id'] == sample_id]
    roll_vals = sample_df['roll'].values
    print(f"\nSample {sample_id} (TILT_LEFT):")
    print(f"  Roll values (all 30 frames): {roll_vals}")
    print(f"  Min: {roll_vals.min():.2f}, Max: {roll_vals.max():.2f}, Mean: {roll_vals.mean():.2f}")
    print(f"  Values > -12: {(roll_vals > -12).sum()}")
    print(f"  Values <= -12: {(roll_vals <= -12).sum()}")

# Also check distribution stats for all TILT_LEFT
print(f"\n{'='*60}")
print("ALL TILT_LEFT SAMPLES DISTRIBUTION:")
print(f"{'='*60}")
all_tilt_left = df[df['label'] == 'TILT_LEFT']
roll_all = all_tilt_left['roll'].values

# Histogram
bins = [-40, -30, -20, -12, 0, 10]
hist, _ = np.histogram(roll_all, bins=bins)
print(f"Roll distribution (histogram):")
for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
    print(f"  [{low:3d}, {high:3d}): {hist[i]:5d} samples ({100*hist[i]/len(roll_all):5.1f}%)")

print(f"\nTotal TILT_LEFT frames: {len(roll_all)}")
print(f"Frames with roll > -12: {(roll_all > -12).sum()} ({100*(roll_all > -12).sum()/len(roll_all):.1f}%)")

# Check a few samples that have values > -12
print(f"\n{'='*60}")
print("SAMPLES WITH ROLL > -12 (SHOULD BE NONE!):")
print(f"{'='*60}")
bad_samples = all_tilt_left[all_tilt_left['roll'] > -12]['sample_id'].unique()
print(f"Number of samples with at least one frame > -12: {len(bad_samples)}")
print(f"Example samples: {bad_samples[:3]}")

for sid in bad_samples[:3]:
    sample_df = df[df['sample_id'] == sid]
    roll_vals = sample_df['roll'].values
    bad_count = (roll_vals > -12).sum()
    print(f"  Sample {sid}: {bad_count}/30 frames > -12, range [{roll_vals.min():.2f}, {roll_vals.max():.2f}]")
