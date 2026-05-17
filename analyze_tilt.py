import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# Load raw CSV
df = pd.read_csv("data/raw/synthetic_dataset.csv")
print(f"[DATA] Total rows: {len(df)}")
print(f"[DATA] Columns: {df.columns.tolist()}")

# Group by label and sample_id
print("\n[TILT_LEFT] Analyzing...")
tilt_left_samples = df[df["label"] == "TILT_LEFT"]["sample_id"].unique()[:5]  # First 5 samples

for sample_id in tilt_left_samples:
    sample_data = df[df["sample_id"] == sample_id]
    yaw = sample_data["yaw"].values
    pitch = sample_data["pitch"].values
    roll = sample_data["roll"].values
    
    print(f"\n  Sample {sample_id}:")
    print(f"    Yaw:   mean={yaw.mean():+.1f}°, std={yaw.std():.1f}°, range=[{yaw.min():+.1f}, {yaw.max():+.1f}]")
    print(f"    Pitch: mean={pitch.mean():+.1f}°, std={pitch.std():.1f}°, range=[{pitch.min():+.1f}, {pitch.max():+.1f}]")
    print(f"    Roll:  mean={roll.mean():+.1f}°, std={roll.std():.1f}°, range=[{roll.min():+.1f}, {roll.max():+.1f}]")

print("\n[SHAKE] Analyzing...")
shake_samples = df[df["label"] == "SHAKE"]["sample_id"].unique()[:5]

for sample_id in shake_samples:
    sample_data = df[df["sample_id"] == sample_id]
    yaw = sample_data["yaw"].values
    pitch = sample_data["pitch"].values
    roll = sample_data["roll"].values
    
    print(f"\n  Sample {sample_id}:")
    print(f"    Yaw:   mean={yaw.mean():+.1f}°, std={yaw.std():.1f}°, range=[{yaw.min():+.1f}, {yaw.max():+.1f}]")
    print(f"    Pitch: mean={pitch.mean():+.1f}°, std={pitch.std():.1f}°, range=[{pitch.min():+.1f}, {pitch.max():+.1f}]")
    print(f"    Roll:  mean={roll.mean():+.1f}°, std={roll.std():.1f}°, range=[{roll.min():+.1f}, {roll.max():+.1f}]")
