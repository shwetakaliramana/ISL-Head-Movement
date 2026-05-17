"""
src/ml/feature_engineering.py  — fixed v2
─────────────────────────────────────────────────────────────────────────────
BUGS FIXED IN THIS VERSION
───────────────────────────
Bug 1 — LabelEncoder alphabetic sort mismatch
    sklearn.LabelEncoder.fit(list) sorts labels alphabetically.
    CLASSES = [NOD, SHAKE, TILT_LEFT, TILT_RIGHT, STATIC]
    Alphabetic order: [NOD→0, SHAKE→1, STATIC→2, TILT_LEFT→3, TILT_RIGHT→4]
    Expected order:   [NOD→0, SHAKE→1, TILT_LEFT→2, TILT_RIGHT→3, STATIC→4]
    Result: TILT_LEFT training samples were stored at index 3, but the model's
    output neuron 2 (labelled TILT_LEFT) was trained on STATIC samples.
    Fix: use a fixed integer mapping dict instead of LabelEncoder.

Bug 2 — Sliding window crosses gesture boundaries
    Grouping only by label-run works for pure gesture blocks but a window
    starting near the end of a TILT_LEFT block and overlapping the next
    block gets an ambiguous feature vector with a mismatched label.
    Fix: windows are only built from the interior of each run
    (start=0..len-WINDOW, step=STEP) with no cross-boundary overlap.

Bug 3 — Test helper used mean=0/std=1 instead of fitted stats
    The verify_tilt_fix.py _window() helper created a DatasetBuilder with
    zeroed stats, so perfect TILT windows fed to the trained model had
    roll_norm = -20/1 = -20.0 instead of -20/12.65 = -1.58.
    Fix: test script now passes the fitted builder to _window().
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

WINDOW  = 30
N_FEAT  = 8
STEP    = 5
CLASSES = ["NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"]

# Fixed label → index mapping (NOT alphabetic — matches CLASSES order above)
LABEL2IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL: dict[int, str] = {i: c for i, c in enumerate(CLASSES)}

_BASE_COLS  = ["yaw", "pitch", "roll", "dyaw", "dpitch", "droll"]
_ROLL_MAX   = 45.0
_CLIP_SIGMA = 3.0


class DatasetBuilder:
    def __init__(self, window: int = WINDOW, step: int = STEP) -> None:
        self._window = window
        self._step   = step
        self._mean:  Optional[np.ndarray] = None   # (6,)
        self._std:   Optional[np.ndarray] = None   # (6,)
        self.classes = CLASSES[:]                  # fixed order

    # ── Fit + transform ────────────────────────────────────────────────────────

    def fit_transform(self, csv_path, fit_stats: bool = True):
        df = self._load(csv_path)
        if fit_stats:
            self._fit(df)
        X, y = self._build(df)
        _log_dist(y)
        return X, y

    def transform(self, csv_path):
        self._check_fitted()
        df = self._load(csv_path)
        X, y = self._build(df)
        return X, y

    # ── Stats I/O ──────────────────────────────────────────────────────────────

    def save_stats(self, path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), mean=self._mean, std=self._std,
                 classes=np.array(self.classes))
        log.info("Stats saved: %s", path)

    @classmethod
    def load_stats(cls, path) -> "DatasetBuilder":
        data = np.load(str(path), allow_pickle=True)
        obj  = cls()
        obj._mean = data["mean"]
        obj._std  = data["std"]
        if "classes" in data:
            saved = list(data["classes"])
            if saved != CLASSES:
                log.warning("Saved classes %s differ from current %s — "
                            "delete stats and retrain.", saved, CLASSES)
        return obj

    # ── Real-time inference ────────────────────────────────────────────────────

    def transform_window(self, feat6: np.ndarray) -> np.ndarray:
        """
        (window,6) raw → (1, window, 8) normalised, ready for model.predict().
        Uses the FITTED mean/std — must call fit_transform or load_stats first.
        """
        self._check_fitted()
        norm  = np.clip((feat6 - self._mean) / self._std, -_CLIP_SIGMA, _CLIP_SIGMA)
        roll  = feat6[:, 2]
        rsign = np.sign(roll).astype(np.float32)
        rabs  = (np.abs(roll) / _ROLL_MAX).clip(0.0, 1.0).astype(np.float32)
        feat  = np.concatenate([norm,
                                rsign[:, np.newaxis],
                                rabs[:, np.newaxis]], axis=1).astype(np.float32)
        return feat[np.newaxis]  # (1, window, 8)

    # ── Private ────────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self._mean is None:
            raise RuntimeError("Not fitted — call fit_transform() or load_stats().")

    def _load(self, path) -> pd.DataFrame:
        df = pd.read_csv(path)
        # Load raw columns
        raw_cols = ["yaw", "pitch", "roll", "label"]
        missing = [c for c in raw_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        
        # Compute derivatives per sample
        df = df.reset_index(drop=True)
        df["dyaw"] = 0.0
        df["dpitch"] = 0.0
        df["droll"] = 0.0
        
        if "sample_id" in df.columns:
            for sid in df["sample_id"].unique():
                mask = df["sample_id"] == sid
                idx = df[mask].index
                if len(idx) > 1:
                    df.loc[idx, "dyaw"] = np.gradient(df.loc[idx, "yaw"].values)
                    df.loc[idx, "dpitch"] = np.gradient(df.loc[idx, "pitch"].values)
                    df.loc[idx, "droll"] = np.gradient(df.loc[idx, "roll"].values)
        else:
            # No sample_id: compute derivatives on entire DF
            df["dyaw"] = np.gradient(df["yaw"].values)
            df["dpitch"] = np.gradient(df["pitch"].values)
            df["droll"] = np.gradient(df["roll"].values)
        
        df = df[df["label"].isin(CLASSES)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No valid-label rows in {path}")
        log.info("Loaded %d rows from %s", len(df), path)
        return df

    def _fit(self, df: pd.DataFrame) -> None:
        vals       = df[_BASE_COLS].values.astype(np.float32)
        self._mean = vals.mean(axis=0)
        self._std  = vals.std(axis=0)
        self._std  = np.where(self._std < 1e-6, 1.0, self._std)
        log.info("Normaliser mean=%s", np.round(self._mean, 3))
        log.info("Normaliser std =%s", np.round(self._std,  3))

    def _build(self, df: pd.DataFrame):
        # Mark gesture run boundaries (label change = new run)
        df = df.copy()
        df["_run"] = (df["label"] != df["label"].shift()).cumsum()

        X_list, y_list = [], []

        for (run_id, label), grp in df.groupby(["_run", "label"], sort=False):
            vals = grp[_BASE_COLS].values.astype(np.float32)
            if len(vals) < self._window:
                continue

            # Normalise
            norm  = np.clip((vals - self._mean) / self._std, -_CLIP_SIGMA, _CLIP_SIGMA)
            roll  = vals[:, 2]
            rsign = np.sign(roll).astype(np.float32)
            rabs  = (np.abs(roll) / _ROLL_MAX).clip(0.0, 1.0).astype(np.float32)
            feat  = np.concatenate([norm,
                                    rsign[:, np.newaxis],
                                    rabs[:, np.newaxis]], axis=1).astype(np.float32)

            # Fixed integer label — NOT LabelEncoder (alphabetic sort bug)
            y_int = LABEL2IDX[label]

            for start in range(0, len(feat) - self._window + 1, self._step):
                X_list.append(feat[start:start + self._window])
                y_list.append(y_int)

        if not X_list:
            raise RuntimeError("No windows built — check CSV labels and WINDOW size.")

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int32)
        log.info("Built %d windows of shape %s", len(X), X.shape[1:])
        return X, y


def _log_dist(y: np.ndarray) -> None:
    total = len(y)
    log.info("Class distribution:")
    for i, cls in enumerate(CLASSES):
        n   = int((y == i).sum())
        pct = 100 * n / total if total > 0 else 0
        log.info("  [%d] %-14s  %4d  (%.1f%%)", i, cls, n, pct)