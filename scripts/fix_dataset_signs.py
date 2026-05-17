#!/usr/bin/env python3
"""Post-process the CSV to enforce tilt sign constraints."""

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data/raw/synthetic_dataset.csv')

print("Before post-processing:")
tilt_left = df[df['label'] == 'TILT_LEFT']
print(f"TILT_LEFT roll: min={tilt_left['roll'].min():.2f}, max={tilt_left['roll'].max():.2f}")
print(f"  Values > -12: {(tilt_left['roll'] > -12).sum()}/{len(tilt_left)}")

tilt_right = df[df['label'] == 'TILT_RIGHT']
print(f"TILT_RIGHT roll: min={tilt_right['roll'].min():.2f}, max={tilt_right['roll'].max():.2f}")
print(f"  Values < 12: {(tilt_right['roll'] < 12).sum()}/{len(tilt_right)}")

# Enforce constraints
df.loc[df['label'] == 'TILT_LEFT', 'roll'] = np.minimum(df.loc[df['label'] == 'TILT_LEFT', 'roll'], -12.0)
df.loc[df['label'] == 'TILT_RIGHT', 'roll'] = np.maximum(df.loc[df['label'] == 'TILT_RIGHT', 'roll'], 12.0)

print("\nAfter post-processing:")
tilt_left = df[df['label'] == 'TILT_LEFT']
print(f"TILT_LEFT roll: min={tilt_left['roll'].min():.2f}, max={tilt_left['roll'].max():.2f}")
print(f"  Values > -12: {(tilt_left['roll'] > -12).sum()}/{len(tilt_left)}")

tilt_right = df[df['label'] == 'TILT_RIGHT']
print(f"TILT_RIGHT roll: min={tilt_right['roll'].min():.2f}, max={tilt_right['roll'].max():.2f}")
print(f"  Values < 12: {(tilt_right['roll'] < 12).sum()}/{len(tilt_right)}")

# Save
df.to_csv('data/raw/synthetic_dataset.csv', index=False)
print("\nSaved → data/raw/synthetic_dataset.csv")
