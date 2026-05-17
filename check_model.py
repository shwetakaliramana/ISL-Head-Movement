import tensorflow as tf
import sys
from pathlib import Path

sys.path.insert(0, ".")

model = tf.keras.models.load_model("models/bilstm_best.keras")
model.summary()

print(f"\n[MODEL] First layer input shape: {model.layers[0].output_shape}")
print(f"[MODEL] Expected: (None, 30, 7)")
