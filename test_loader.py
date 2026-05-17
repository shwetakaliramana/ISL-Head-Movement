import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")

from src.ml.feature_engineering import DatasetBuilder, N_FEAT

print(f"[TEST] N_FEAT constant = {N_FEAT}")

builder = DatasetBuilder()
X, y = builder.fit_transform("data/raw/synthetic_dataset.csv")
print(f"[TEST] X shape: {X.shape}")
print(f"[TEST] Expected: (4000, 30, 7), Got: {X.shape}")

if X.shape[2] == 7:
    print("✓ Dataset has 7 features as expected")
else:
    print(f"✗ ERROR: Dataset has {X.shape[2]} features, expected 7")
    
print(f"[TEST] Normalizer: mean shape {builder._mean.shape}, std shape {builder._std.shape}")
