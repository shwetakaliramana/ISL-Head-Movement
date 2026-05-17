from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.pipeline.calibration import build_approximate_intrinsics
from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PoseResult:
    yaw: float
    pitch: float
    roll: float
    rotation_vec: np.ndarray
    translation_vec: np.ndarray


class HeadPoseEstimator:
    """Estimate head pose (yaw, pitch, roll) from 2D face anchor points."""

    # Generic 3D face model points in millimeters.
    _MODEL_POINTS = np.array(
        [
            [0.0, 0.0, 0.0],        # nose_tip
            [0.0, -63.6, -12.5],    # chin
            [-43.3, 32.7, -26.0],   # left_eye_corner
            [43.3, 32.7, -26.0],    # right_eye_corner
            [-28.9, -28.9, -24.1],  # left_mouth_corner
            [28.9, -28.9, -24.1],   # right_mouth_corner
        ],
        dtype=np.float64,
    )

    _NAMES = [
        "nose_tip",
        "chin",
        "left_eye_corner",
        "right_eye_corner",
        "left_mouth_corner",
        "right_mouth_corner",
    ]

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self._K, self._D = build_approximate_intrinsics(frame_width, frame_height)
        self._alpha = float(getattr(cfg.pose_estimation, "smoothing_alpha", 0.4))
        self._prev_angles: tuple[float, float, float] | None = None

    @staticmethod
    def _rotation_matrix_to_euler_degrees(R: np.ndarray) -> tuple[float, float, float]:
        # ZYX convention: roll (Z), yaw (Y), pitch (X)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0.0

        return (
            float(np.degrees(yaw)),
            float(np.degrees(pitch)),
            float(np.degrees(roll)),
        )

    def estimate(self, anchor_px: dict[str, tuple[int, int]]) -> PoseResult | None:
        """Estimate pose from anchor pixel coordinates returned by FaceMeshDetector."""
        if any(name not in anchor_px for name in self._NAMES):
            return None

        image_points = np.array([anchor_px[name] for name in self._NAMES], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            self._MODEL_POINTS,
            image_points,
            self._K,
            self._D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = self._rotation_matrix_to_euler_degrees(R)

        if self._prev_angles is not None:
            py, pp, pr = self._prev_angles
            a = self._alpha
            yaw = a * yaw + (1.0 - a) * py
            pitch = a * pitch + (1.0 - a) * pp
            roll = a * roll + (1.0 - a) * pr

        self._prev_angles = (yaw, pitch, roll)
        return PoseResult(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            rotation_vec=rvec,
            translation_vec=tvec,
        )

    def reset_smoothing(self) -> None:
        self._prev_angles = None
