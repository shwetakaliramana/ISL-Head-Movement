from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.feature_engineering import DatasetBuilder

CLASSES = ["NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"]


def split_dataset(X, y, val_frac=0.15, test_frac=0.15, seed=42):
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=seed
    )
    val_size = val_frac / (1 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_size, stratify=y_tv, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true, y_pred, classes):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    per_class = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i].sum()),
        }

    accuracy = float(np.diag(cm).sum()) / len(y_true)
    macro_f1 = float(np.mean([per_class[c]["f1"] for c in classes]))
    weighted_f1 = float(
        np.average([per_class[c]["f1"] for c in classes], weights=[per_class[c]["support"] for c in classes])
    )
    return accuracy, macro_f1, weighted_f1, per_class, cm


def main() -> None:
    builder = DatasetBuilder(classes=CLASSES)
    X, y = builder.fit_transform("data/raw/synthetic_dataset.csv")
    _, _, X_te, _, _, y_te = split_dataset(X, y, seed=42)

    model = tf.keras.models.load_model("models/bilstm_best.keras", safe_mode=False)
    y_prob = model.predict(X_te, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy, macro_f1, weighted_f1, per_class, cm = compute_metrics(y_te, y_pred, CLASSES)

    print(f"accuracy={accuracy:.4f}")
    print(f"macro_f1={macro_f1:.4f}")
    print(f"weighted_f1={weighted_f1:.4f}")
    for cls in CLASSES:
        print(f"{cls} f1={per_class[cls]['f1']:.4f} support={per_class[cls]['support']}")
    print("cm=")
    print(cm)


if __name__ == "__main__":
    main()
