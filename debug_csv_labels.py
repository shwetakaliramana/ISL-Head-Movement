"""
Debug: Check what's actually in the CSV before/after DatasetBuilder reads it.
"""
import sys, tempfile, numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from src.ml.feature_engineering import WINDOW

def make_perfect_csv(path, n=5):
    """Create tiny CSV to inspect."""
    rows = []
    sid = 0

    # TILT_LEFT: roll = -20
    for _ in range(n):
        for i in range(WINDOW):
            rows.append([sid, i, 0.0, 0.0, -20.0, 0.0, 0.0, 0.0, "TILT_LEFT"])
        sid += 1

    # TILT_RIGHT: roll = +20
    for _ in range(n):
        for i in range(WINDOW):
            rows.append([sid, i, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, "TILT_RIGHT"])
        sid += 1

    df = pd.DataFrame(rows, columns=["sample_id", "frame", "yaw", "pitch", "roll", "dyaw", "dpitch", "droll", "label"])
    df.to_csv(path, index=False)

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
    csv_path = Path(f.name)

make_perfect_csv(csv_path, n=5)

# Read the raw CSV
print("=== RAW CSV CONTENTS ===")
df_raw = pd.read_csv(csv_path)
print(df_raw.head(60))
print(f"\nUnique labels in CSV: {df_raw['label'].unique()}")
print(f"Label value counts:\n{df_raw['label'].value_counts()}")

# Now process through DatasetBuilder
from src.ml.feature_engineering import DatasetBuilder

print("\n=== PROCESSING WITH DatasetBuilder ===")
b = DatasetBuilder()
X, y = b.fit_transform(csv_path)

print(f"\nAfter DatasetBuilder:")
print(f"  X shape: {X.shape}")
print(f"  y: {y}")
print(f"  y unique: {np.unique(y)}")

# Check the label encoding mapping
from src.ml.feature_engineering import CLASSES
print(f"\nCLASSES order: {CLASSES}")
for i, cls in enumerate(CLASSES):
    mask = (y == i)
    n_samples = mask.sum()
    if n_samples > 0:
        sample_rolls = X[mask, 0, 2]  # first frame, feature 2 (roll_norm)
        print(f"  Index {i} ({cls:<14}): n={n_samples}, sample roll_norm={sample_rolls[:3]} ...")

csv_path.unlink(missing_ok=True)
