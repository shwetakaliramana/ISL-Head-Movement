from __future__ import annotations

from pathlib import Path

import numpy as np


def build_approximate_intrinsics(
    frame_width: int,
    frame_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an approximate camera matrix from frame dimensions."""
    fl = float(frame_width)
    cx = frame_width / 2.0
    cy = frame_height / 2.0

    K = np.array(
        [
            [fl, 0.0, cx],
            [0.0, fl, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    D = np.zeros((4, 1), dtype=np.float64)
    return K, D


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load camera intrinsics and distortion coefficients from an .npz file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration file not found: {p}")
    data = np.load(str(p))
    return data["camera_matrix"], data["dist_coeffs"]
