import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")

from src.ml.feature_engineering import DatasetBuilder

CLASSES = ["NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"]

# Load data
builder = DatasetBuilder(classes=CLASSES)
X, y = builder.fit_transform("data/raw/synthetic_dataset.csv")
print(f"[DIAG] Loaded: X={X.shape}, y={y.shape}")
print(f"[DIAG] Y distribution: {np.bincount(y)}")

# Split like train_bilstm.py does
val_frac = 0.15
test_frac = 0.15
seed = 42

X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=test_frac, stratify=y, random_state=seed)
val_size = val_frac / (1 - test_frac)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=val_size, stratify=y_tv, random_state=seed)

print(f"\n[SPLIT] Train: {X_train.shape}, y_dist={np.bincount(y_train)}")
print(f"[SPLIT] Val: {X_val.shape}, y_dist={np.bincount(y_val)}")
print(f"[SPLIT] Test: {X_test.shape}, y_dist={np.bincount(y_test)}")

# Check data
print(f"\n[DATA] X_train min/max: {X_train.min():.3f} / {X_train.max():.3f}")
print(f"[DATA] X_val min/max: {X_val.min():.3f} / {X_val.max():.3f}")
print(f"[DATA] X_test min/max: {X_test.min():.3f} / {X_test.max():.3f}")

# Check for NaN/Inf
print(f"\n[NAN CHECK]")
print(f"X_train NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")
print(f"X_val NaN: {np.isnan(X_val).sum()}, Inf: {np.isinf(X_val).sum()}")
print(f"X_test NaN: {np.isnan(X_test).sum()}, Inf: {np.isinf(X_test).sum()}")
