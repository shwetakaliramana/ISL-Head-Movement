"""
src/pipeline/calibration.py
─────────────────────────────────────────────────────────────────────────────
Camera calibration helpers for the pose estimator.

Two modes:

1. APPROXIMATE  (default, zero setup)
   Builds an intrinsic matrix from the frame dimensions alone.
   Focal length = frame width.  Good enough for real-time head tracking.

2. CHECKERBOARD  (optional, higher accuracy)
   Collects frames showing a printed checkerboard, runs
   cv2.calibrateCamera, and saves the result to configs/camera_calib.npz.
   Run once per camera; load the saved file on subsequent runs.

Usage — approximate (used automatically by HeadPoseEstimator):
    from src.pipeline.calibration import build_approximate_intrinsics
    K, D = build_approximate_intrinsics(1280, 720)

Usage — checkerboard calibration (run once, save, reload):
    from src.pipeline.calibration import CheckerboardCalibrator
    cal = CheckerboardCalibrator(cols=9, rows=6, square_mm=25.0)
    K, D, rms = cal.run_interactive()   # opens webcam, press SPACE to capture
    cal.save("configs/camera_calib.npz")

Usage — load saved calibration:
    from src.pipeline.calibration import load_calibration
    K, D = load_calibration("configs/camera_calib.npz")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Approximate intrinsics ─────────────────────────────────────────────────────

def build_approximate_intrinsics(
    frame_width: int,
    frame_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a camera matrix using frame dimensions (no calibration required).

    The focal-length-equals-width heuristic gives ≈ 60° horizontal FOV,
    which matches most laptop/desktop webcams within ~5°.

    Returns:
        (camera_matrix 3×3, dist_coeffs 4×1 zeros)
    """
    fl = float(frame_width)
    cx = frame_width  / 2.0
    cy = frame_height / 2.0

    K = np.array([
        [fl, 0,  cx],
        [0,  fl, cy],
        [0,  0,  1 ],
    ], dtype=np.float64)

    D = np.zeros((4, 1), dtype=np.float64)
    return K, D


# ── Load saved calibration ─────────────────────────────────────────────────────

def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load camera_matrix and dist_coeffs from a saved .npz file.

    Raises:
        FileNotFoundError if the file does not exist.
        KeyError if the file is missing expected arrays.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            "Run CheckerboardCalibrator.run_interactive() to create it."
        )
    data = np.load(str(path))
    K = data["camera_matrix"]
    D = data["dist_coeffs"]
    log.info("Loaded calibration from %s  (RMS=%.4f)", path, float(data.get("rms", -1)))
    return K, D


# ── Checkerboard calibrator ────────────────────────────────────────────────────

class CheckerboardCalibrator:
    """
    Interactive camera calibration using a printed checkerboard target.

    Args:
        cols:       number of inner corners along the short axis (e.g. 9).
        rows:       number of inner corners along the long axis  (e.g. 6).
        square_mm:  physical size of one square in millimetres (e.g. 25.0).
        min_frames: minimum accepted captures before calibration runs.
    """

    def __init__(
        self,
        cols: int = 9,
        rows: int = 6,
        square_mm: float = 25.0,
        min_frames: int = 15,
    ) -> None:
        self._cols = cols
        self._rows = rows
        self._square_mm = square_mm
        self._min_frames = min_frames
        self._pattern_size = (cols, rows)

        # Prepare object points (0,0,0), (1,0,0), ..., scaled to mm
        objp = np.zeros((cols * rows, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= square_mm
        self._objp = objp

        self._obj_points: list[np.ndarray] = []
        self._img_points: list[np.ndarray] = []
        self._frame_size: tuple[int, int] | None = None

        # Results
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs:   Optional[np.ndarray] = None
        self.rms:           float = -1.0

    def run_interactive(self, device_id: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Open the webcam and interactively collect calibration frames.

        Controls:
            SPACE  → capture current frame if corners detected
            q      → quit and run calibration (requires >= min_frames)
            r      → reset all captured frames

        Returns:
            (camera_matrix, dist_coeffs, rms_reprojection_error)
        """
        cap = cv2.VideoCapture(device_id)
        log.info(
            "Calibration mode  |  SPACE=capture  r=reset  q=calibrate & quit"
        )
        log.info("Target: %dx%d inner corners, %.1fmm squares", self._cols, self._rows, self._square_mm)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, self._pattern_size, None)

            display = frame.copy()
            if found:
                # Refine corners to sub-pixel accuracy
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, self._pattern_size, corners_refined, found)

            status = f"Captured: {len(self._obj_points)} / {self._min_frames}"
            color  = (0, 255, 120) if found else (0, 100, 220)
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, "SPACE=capture  r=reset  q=calibrate",
                        (10, display.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow("Camera Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and found:
                self._obj_points.append(self._objp)
                self._img_points.append(corners_refined)
                self._frame_size = gray.shape[::-1]
                log.info("Frame captured. Total: %d", len(self._obj_points))
            elif key == ord("r"):
                self._obj_points.clear()
                self._img_points.clear()
                log.info("Reset — all captured frames cleared.")
            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(self._obj_points) < self._min_frames:
            raise RuntimeError(
                f"Not enough frames for calibration "
                f"({len(self._obj_points)} < {self._min_frames}). "
                "Capture more views and try again."
            )

        return self._calibrate()

    def _calibrate(self) -> tuple[np.ndarray, np.ndarray, float]:
        log.info("Running calibration on %d frames …", len(self._obj_points))
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            self._obj_points,
            self._img_points,
            self._frame_size,
            None,
            None,
        )
        self.camera_matrix = K
        self.dist_coeffs   = D
        self.rms           = rms
        log.info("Calibration done. RMS reprojection error = %.4f px", rms)
        if rms > 1.0:
            log.warning("RMS > 1.0 — consider recapturing with better coverage.")
        return K, D, rms

    def save(self, path: str | Path = "configs/camera_calib.npz") -> None:
        """Save calibration to a .npz file."""
        if self.camera_matrix is None:
            raise RuntimeError("No calibration to save — run run_interactive() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            rms=np.array(self.rms),
        )
        log.info("Calibration saved to %s", path)