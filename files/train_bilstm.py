"""
scripts/train_bilstm.py
─────────────────────────────────────────────────────────────────────────────
Training script for the fixed BiLSTM gesture classifier.

Key changes vs the broken version:
  1. Uses new feature_engineering.py (8 features, roll_sign + roll_abs)
  2. Passes class_weight to model.fit() (TILT classes get 2× boost)
  3. Uses focal_loss (optional, enabled with --focal flag)
  4. Prints per-class distribution in ALL splits before training
     so you immediately catch any missing-class bug
  5. Saves normaliser_stats.npz BEFORE training so it's always consistent
     with the model that was actually trained
  6. TiltRecallCallback: monitors TILT_LEFT and TILT_RIGHT F1 separately
     and saves the best checkpoint by TILT F1, not just val_accuracy

Usage:
  # Standard training (recommended first run)
  python scripts/train_bilstm.py --data data/raw/synthetic_dataset.csv

  # With focal loss
  python scripts/train_bilstm.py --data data/raw/synthetic_dataset.csv --focal

  # Full options
  python scripts/train_bilstm.py \
      --data  data/raw/synthetic_dataset.csv \
      --out   models/checkpoints \
      --stats models/normaliser_stats.npz \
      --epochs 150 \
      --lr 3e-4 \
      --focal
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.bilstm_model import build_model, get_class_weights, focal_loss
from src.ml.feature_engineering import DatasetBuilder, CLASSES
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TILT-aware callback
# ─────────────────────────────────────────────────────────────────────────────

class TiltRecallCallback(tf.keras.callbacks.Callback):
    """
    After each epoch, compute per-class F1 on the validation set.
    Saves the model when the MEAN of TILT_LEFT and TILT_RIGHT F1 improves.
    Also stops early if TILT F1 hasn't improved for `patience` epochs.
    """

    def __init__(
        self,
        val_data:   tuple[np.ndarray, np.ndarray],
        save_path:  str,
        patience:   int = 15,
    ) -> None:
        super().__init__()
        self._X_val, self._y_val = val_data
        self._save_path  = save_path
        self._patience   = patience
        self._best_tilt  = 0.0
        self._wait       = 0
        self._tilt_idx   = [CLASSES.index("TILT_LEFT"), CLASSES.index("TILT_RIGHT")]

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        probs = self.model.predict(self._X_val, verbose=0)
        preds = np.argmax(probs, axis=1)

        report = classification_report(
            self._y_val, preds,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        tilt_f1s = [report.get(cls, {}).get("f1-score", 0.0)
                    for cls in ["TILT_LEFT", "TILT_RIGHT"]]
        mean_tilt_f1 = float(np.mean(tilt_f1s))

        log.info(
            "Epoch %3d | val_acc=%.4f | "
            "TILT_L F1=%.4f  TILT_R F1=%.4f  mean=%.4f",
            epoch + 1,
            logs.get("val_accuracy", 0),
            tilt_f1s[0], tilt_f1s[1], mean_tilt_f1,
        )

        if mean_tilt_f1 > self._best_tilt:
            self._best_tilt = mean_tilt_f1
            self._wait      = 0
            self.model.save(self._save_path)
            log.info("  ✅ New best TILT F1=%.4f — model saved.", mean_tilt_f1)
        else:
            self._wait += 1
            if self._wait >= self._patience:
                log.info("  Early stop: TILT F1 no improvement for %d epochs.", self._patience)
                self.model.stop_training = True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path  = str(out_dir / "bilstm_best.keras")
    stats_path = Path(args.stats)

    # ── Build dataset ─────────────────────────────────────────────────────────
    log.info("Building dataset from: %s", args.data)
    builder = DatasetBuilder()
    X, y    = builder.fit_transform(args.data, fit_stats=True)

    # CRITICAL: save stats BEFORE any split so they're always consistent
    builder.save_stats(stats_path)

    # ── Verify all classes present ────────────────────────────────────────────
    unique, counts = np.unique(y, return_counts=True)
    log.info("Full dataset class distribution:")
    for idx, cnt in zip(unique, counts):
        log.info("  [%d] %-14s  %d", idx, CLASSES[idx], cnt)

    missing = set(range(len(CLASSES))) - set(unique.tolist())
    if missing:
        missing_names = [CLASSES[i] for i in missing]
        log.error("MISSING CLASSES IN DATASET: %s", missing_names)
        log.error("Add more recordings for these classes and rebuild the dataset.")
        sys.exit(1)

    # ── Stratified splits (train 70 / val 15 / test 15) ──────────────────────
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=42)

    log.info("Split sizes — train: %d  val: %d  test: %d",
             len(X_tr), len(X_va), len(X_te))

    # CRITICAL CHECK: print class counts in TRAINING split
    log.info("Training split class distribution:")
    for idx in range(len(CLASSES)):
        n = int((y_tr == idx).sum())
        if n == 0:
            log.error("  [%d] %-14s  ZERO SAMPLES IN TRAINING — retrain will fail!", idx, CLASSES[idx])
        else:
            log.info("  [%d] %-14s  %d", idx, CLASSES[idx], n)

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weight = get_class_weights(y_tr)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model()
    model.summary(print_fn=log.info)

    loss = focal_loss(gamma=2.0) if args.focal else "sparse_categorical_crossentropy"
    log.info("Loss function: %s", "focal_loss(gamma=2.0)" if args.focal else "sparse_categorical_crossentropy")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss,
        metrics=["accuracy"],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        TiltRecallCallback(
            val_data=(X_va, y_va),
            save_path=best_path,
            patience=args.patience,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(out_dir / "tb_logs"),
            histogram_freq=0,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training — epochs=%d  lr=%.4f  focal=%s",
             args.epochs, args.lr, args.focal)

    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=args.epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Final evaluation on test set ──────────────────────────────────────────
    log.info("Loading best checkpoint: %s", best_path)
    best_model = tf.keras.models.load_model(best_path, safe_mode=False)

    probs = best_model.predict(X_te, verbose=0)
    preds = np.argmax(probs, axis=1)

    report = classification_report(
        y_te, preds,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
    log.info("\n%s", report)

    # Summary: flag any class with F1 < 0.80
    report_dict = classification_report(
        y_te, preds,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    for cls in CLASSES:
        f1  = report_dict.get(cls, {}).get("f1-score", 0.0)
        rec = report_dict.get(cls, {}).get("recall",   0.0)
        ok  = "✅" if f1 >= 0.80 else "❌"
        print(f"  {ok} {cls:<14}  F1={f1:.4f}  Recall={rec:.4f}")
    print(f"\n  Macro F1: {report_dict['macro avg']['f1-score']:.4f}")
    print("="*60)
    print(f"  Best model: {best_path}")
    print(f"  Stats:      {stats_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL BiLSTM Gesture Classifier")
    parser.add_argument("--data",    required=True,
                        help="Path to angles CSV (data/raw/synthetic_dataset.csv)")
    parser.add_argument("--out",     default="models/checkpoints",
                        help="Output directory for model checkpoints")
    parser.add_argument("--stats",   default="models/normaliser_stats.npz",
                        help="Path to save normaliser stats")
    parser.add_argument("--epochs",  type=int,   default=150)
    parser.add_argument("--lr",      type=float, default=3e-4,
                        help="Learning rate (3e-4 recommended after fixing features)")
    parser.add_argument("--patience",type=int,   default=20,
                        help="Early stopping patience on TILT F1")
    parser.add_argument("--focal",   action="store_true",
                        help="Use focal loss instead of cross-entropy")
    args = parser.parse_args()
    main(args)
