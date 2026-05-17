import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")

from src.ml.feature_engineering import DatasetBuilder
from src.ml.bilstm_model import build_bilstm
import tensorflow as tf

CLASSES = ["NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"]

# Load data
builder = DatasetBuilder(classes=CLASSES)
X, y = builder.fit_transform("data/raw/synthetic_dataset.csv")

# Use tiny subset to test overfitting
X_tiny = X[:100]  # Just 100 samples
y_tiny = y[:100]

print(f"[TEST] Tiny dataset: X={X_tiny.shape}, y={y_tiny.shape}")
print(f"[TEST] Y distribution: {np.bincount(y_tiny)}")

# Build and train model
model = build_bilstm()
print(f"[MODEL] Built with input shape (30, 7)")

history = model.fit(
    X_tiny, y_tiny,
    epochs=10,
    batch_size=16,
    verbose=1,
)

print(f"\n[RESULT] Final accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"[RESULT] Should be close to 1.0 if model can learn")

# Test prediction shape
pred = model.predict(X_tiny[:5], verbose=0)
print(f"\n[PRED] Prediction shape: {pred.shape}")
print(f"[PRED] Expected: (5, 5) for 5 samples, 5 classes")
