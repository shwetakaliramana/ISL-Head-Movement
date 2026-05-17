"""
src/ml/feature_engineering.py
─────────────────────────────────────────────────────────────────────────────
Feature engineering for the BiLSTM gesture classifier.

ROOT CAUSE OF TILT_LEFT/TILT_RIGHT COLLAPSE
────────────────────────────────────────────
Standard z-score normalisation centres roll at zero and scales by std.
If the dataset has roughly equal TILT_LEFT and TILT_RIGHT samples, the
mean roll ≈ 0 and std ≈ large.  After normalisation:
    TILT_LEFT  roll=-25° → normalised ≈ -1.0
    TILT_RIGHT roll=+25° → normalised ≈ +1.0

These ARE distinguishable — BUT only if the LSTM can propagate the sign
of a slowly-varying feature across 30 time steps.  A BiLSTM with dropout
and weight decay can wash out a sustained DC offset, treating both as
"roll is active" rather than "roll is negative / positive".

THE FIX  (three complementary changes)
───────────────────────────────────────
1. Explicit sign feature  — add roll_sign = sign(raw_roll) as a separate
   binary channel {-1, 0, +1}.  This is NOT normalised.  The model gets
   an unambiguous polarity signal on every frame.

2. Asymmetric per-feature scaling  — normalise each feature by its own
   mean and std computed ONLY from training data, then clamp to [-3, 3].
   Prevents dominant features (yaw oscillation in SHAKE) from drowning
   out smaller sustained offsets (roll in TILT).

3. Raw roll magnitude channel  — add |roll| / 45.0  (normalised 0–1).
   Combined with sign, the model has both direction and magnitude without
   relying on the LSTM to reconstruct them from a normalised value.

Final feature vector per frame (N_FEAT = 8):
    [yaw_norm, pitch_norm, roll_norm,
     dyaw_norm, dpitch_norm, droll_norm,
     roll_sign,                           ← NEW: {-1, 0, +1}, not normalised
     roll_abs_norm]                        ← NEW: |roll|/45, range 0–1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW  = 30          # frames per sample (must match config window_frames)
N_FEAT  = 8           # feature dimension per frame
STEP    = 5           # sliding window stride (frames)
CLASSES = ["NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"]

_BASE_COLS  = ["yaw", "pitch", "roll", "dyaw", "dpitch", "droll"]
_ROLL_MAX   = 45.0     # degrees — for abs normalisation
_CLIP_SIGMA = 3.0      # clip normalised values to ±3σ


class DatasetBuilder:
    """
    Converts a raw angles CSV into (X, y) arrays ready for LSTM training.

    Fit on training data only, then transform train/val/test with saved stats.

    Args:
        window: number of frames per sample.
        step:   sliding window stride.
    """

    def __init__(self, window: int = WINDOW, step: int = STEP) -> None:
        self._window = window
        self._step   = step
        self._mean:   Optional[np.ndarray] = None  # shape (6,)
        self._std:    Optional[np.ndarray] = None  # shape (6,)
        self._le      = LabelEncoder()
        self._le.fit(CLASSES)
        self.classes  = list(self._le.classes_)

    # ── Fit + transform ────────────────────────────────────────────────────────

    def fit_transform(
        self,
        csv_path:   str | Path,
        fit_stats:  bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load CSV, fit normalisation stats (if fit_stats=True), build windows.

        Returns:
            X: float32 array of shape (N_samples, window, N_FEAT)
            y: int32   array of shape (N_samples,)
        """
        df = self._load_csv(csv_path)
        if fit_stats:
            self._fit_normaliser(df)
        return self._build_windows(df)

    def transform(
        self,
        csv_path: str | Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform without refitting — use for val/test sets."""
        if self._mean is None:
            raise RuntimeError("Call fit_transform() first or load stats with load_stats().")
        df = self._load_csv(csv_path)
        return self._build_windows(df)

    # ── Normalisation stats I/O ────────────────────────────────────────────────

    def save_stats(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path),
                 mean=self._mean,
                 std=self._std,
                 classes=np.array(self.classes))
        log.info("Normaliser stats saved: %s", path)

    @classmethod
    def load_stats(cls, path: str | Path) -> "DatasetBuilder":
        """Load saved stats and return a ready-to-use DatasetBuilder."""
        data  = np.load(str(path), allow_pickle=True)
        obj   = cls()
        obj._mean = data["mean"]
        obj._std  = data["std"]

        # Backwards compat: older .npz files may not have 'classes'
        if "classes" in data:
            obj._le.fit(list(data["classes"]))
            obj.classes = list(data["classes"])
        else:
            log.warning("normaliser_stats.npz has no 'classes' key — using default CLASSES list.")
            obj._le.fit(CLASSES)
            obj.classes = CLASSES

        log.info("Normaliser stats loaded from %s", path)
        return obj

    # ── Real-time inference helper ─────────────────────────────────────────────

    def transform_window(self, feat6: np.ndarray) -> np.ndarray:
        """
        Normalise and augment a single (window, 6) array for live inference.

        Args:
            feat6: raw [yaw, pitch, roll, dyaw, dpitch, droll] shape (30, 6).
        Returns:
            float32 array shape (1, 30, 8) ready for model.predict().
        """
        if self._mean is None:
            raise RuntimeError("Stats not loaded.")
        norm  = np.clip((feat6 - self._mean) / self._std, -_CLIP_SIGMA, _CLIP_SIGMA)
        roll  = feat6[:, 2]                                    # raw roll, shape (30,)
        rsign = np.sign(roll).astype(np.float32)               # {-1, 0, 1}
        rabs  = (np.abs(roll) / _ROLL_MAX).clip(0.0, 1.0).astype(np.float32)

        feat = np.concatenate([
            norm,
            rsign[:, np.newaxis],
            rabs[:, np.newaxis],
        ], axis=1).astype(np.float32)                          # (30, 8)

        return feat[np.newaxis]                                 # (1, 30, 8)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = [c for c in _BASE_COLS + ["label"] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        df = df[df["label"].isin(CLASSES)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows with valid labels in {path}")
        log.info("Loaded %d rows from %s", len(df), path)
        return df

    def _fit_normaliser(self, df: pd.DataFrame) -> None:
        """Compute mean and std over base 6 features from df."""
        vals       = df[_BASE_COLS].values.astype(np.float32)
        self._mean = vals.mean(axis=0)
        self._std  = vals.std(axis=0)
        # Prevent division by zero on near-constant features
        self._std  = np.where(self._std < 1e-6, 1.0, self._std)
        log.info("Normaliser fitted:  mean=%s", np.round(self._mean, 3))
        log.info("                    std =%s", np.round(self._std,  3))

        # Sanity check: roll mean should be near 0 if dataset is balanced
        if abs(self._mean[2]) > 5.0:
            log.warning(
                "Roll mean=%.2f is large — dataset may be unbalanced "
                "(more TILT_LEFT or TILT_RIGHT than the other). "
                "Consider collecting more of the minority class.",
                self._mean[2],
            )

    def _build_windows(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a window over each session and build (X, y) pairs."""
        # ── Group by session / label run to avoid window straddling two gestures
        # Use label change as session boundary (each contiguous run is one gesture block)
        df["_run"] = (df["label"] != df["label"].shift()).cumsum()

        X_list: list[np.ndarray] = []
        y_list: list[int]        = []

        for (run_id, label), group in df.groupby(["_run", "label"]):
            vals = group[_BASE_COLS].values.astype(np.float32)  # (T, 6)
            if len(vals) < self._window:
                continue

            # Normalise base 6 features
            norm = np.clip(
                (vals - self._mean) / self._std,
                -_CLIP_SIGMA, _CLIP_SIGMA,
            )

            # ── Augmentation channels ──────────────────────────────────────────
            raw_roll  = vals[:, 2]
            roll_sign = np.sign(raw_roll).astype(np.float32)          # {-1, 0, 1}
            roll_abs  = (np.abs(raw_roll) / _ROLL_MAX).clip(0.0, 1.0) # 0–1

            feat = np.concatenate([
                norm,
                roll_sign[:, np.newaxis],
                roll_abs[:, np.newaxis],
            ], axis=1).astype(np.float32)                              # (T, 8)

            y_int = int(self._le.transform([label])[0])

            # Sliding windows
            for start in range(0, len(feat) - self._window + 1, self._step):
                X_list.append(feat[start:start + self._window])
                y_list.append(y_int)

        if not X_list:
            raise RuntimeError("No windows built — check CSV labels and window size.")

        X = np.stack(X_list).astype(np.float32)   # (N, 30, 8)
        y = np.array(y_list, dtype=np.int32)

        log.info("Built %d windows of shape %s", len(X), X.shape[1:])
        _log_class_dist(y, self.classes)
        return X, y


# ── Utility ────────────────────────────────────────────────────────────────────

def _log_class_dist(y: np.ndarray, classes: list[str]) -> None:
    total = len(y)
    log.info("Class distribution in built dataset:")
    for i, cls in enumerate(classes):
        n   = int((y == i).sum())
        pct = 100 * n / total if total > 0 else 0
        log.info("  [%d] %-14s  %4d  (%.1f%%)", i, cls, n, pct)
