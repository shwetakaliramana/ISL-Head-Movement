"""
src/pipeline/face_mesh.py
MediaPipe FaceMesh wrapper.

Produces:
  - normalised 3D landmarks (x, y, z) for all 468/478 points
  - pixel-space 2D landmarks for the configured anchor points
  - a drawing spec for optional mesh overlay

Usage:
    from src.pipeline.face_mesh import FaceMeshDetector

    detector = FaceMeshDetector()
    with detector:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)
        if result:
            print(result.anchor_px)   # dict of anchor_name -> (x, y) in pixels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import cfg
from logger import get_logger

log = get_logger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# ── Named anchor indices from config ──────────────────────────────────────────
ANCHOR_INDICES: dict[str, int] = {
    "nose_tip":          cfg.landmarks.nose_tip,
    "chin":              cfg.landmarks.chin,
    "left_eye_corner":   cfg.landmarks.left_eye_corner,
    "right_eye_corner":  cfg.landmarks.right_eye_corner,
    "left_mouth_corner": cfg.landmarks.left_mouth_corner,
    "right_mouth_corner": cfg.landmarks.right_mouth_corner,
}


@dataclass
class FaceMeshResult:
    """All outputs from a single frame detection."""

    # Raw normalised landmarks (list of (x, y, z), normalised 0-1)
    landmarks_norm: list[tuple[float, float, float]] = field(default_factory=list)

    # Pixel coords for anchor points  {name: (px_x, px_y)}
    anchor_px: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Pixel coords for ALL landmarks  [(px_x, px_y), ...]
    all_px: list[tuple[int, int]] = field(default_factory=list)

    # 3D coords for anchor points (MediaPipe's normalised z, depth relative to face size)
    anchor_3d: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return len(self.landmarks_norm) > 0


class FaceMeshDetector:
    """
    Wraps mp.solutions.face_mesh.FaceMesh with config-driven settings.
    Use as a context manager to ensure proper resource cleanup.
    """

    def __init__(self) -> None:
        self._solution = None
        self._last_result: Optional[FaceMeshResult] = None

    def _resolve_model_path(self) -> Path:
        project_root = Path(__file__).resolve().parent.parent
        model_dir = project_root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "face_landmarker.task"
        if not model_path.exists():
            log.info("Downloading FaceLandmarker model to %s", model_path)
            urlretrieve(MODEL_URL, model_path)
        return model_path

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "FaceMeshDetector":
        mp_cfg = cfg.mediapipe
        model_path = self._resolve_model_path()

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=mp_cfg.max_num_faces,
            min_face_detection_confidence=mp_cfg.min_detection_confidence,
            min_face_presence_confidence=mp_cfg.min_tracking_confidence,
            min_tracking_confidence=mp_cfg.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._solution = vision.FaceLandmarker.create_from_options(options)
        log.info(
            "FaceLandmarker initialised (max_faces=%d)",
            mp_cfg.max_num_faces,
        )
        return self

    def __exit__(self, *_) -> None:
        if self._solution is not None:
            self._solution.close()
            log.info("FaceLandmarker closed.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        rgb_frame: np.ndarray,
    ) -> Optional[FaceMeshResult]:
        """
        Run face mesh on an RGB frame.

        Args:
            rgb_frame: HxWx3 uint8 RGB image.
        Returns:
            FaceMeshResult if a face is found, else None.
        """
        if self._solution is None:
            raise RuntimeError("FaceMeshDetector not initialised — use as a context manager.")

        h, w = rgb_frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mp_result = self._solution.detect(mp_image)

        if not mp_result.face_landmarks:
            self._last_result = None
            return None

        # Use first (and usually only) detected face
        lms = mp_result.face_landmarks[0]

        # Build normalised landmark list
        landmarks_norm = [(lm.x, lm.y, lm.z) for lm in lms]

        # Pixel-space for all landmarks
        all_px = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        # Anchor pixel coords and 3D coords
        anchor_px: dict[str, tuple[int, int]] = {}
        anchor_3d: dict[str, tuple[float, float, float]] = {}
        for name, idx in ANCHOR_INDICES.items():
            lm = lms[idx]
            anchor_px[name] = (int(lm.x * w), int(lm.y * h))
            anchor_3d[name] = (lm.x, lm.y, lm.z)

        result = FaceMeshResult(
            landmarks_norm=landmarks_norm,
            anchor_px=anchor_px,
            all_px=all_px,
            anchor_3d=anchor_3d,
        )
        self._last_result = result
        return result

    def draw_mesh(
        self,
        bgr_frame: np.ndarray,
        rgb_frame: np.ndarray,
        draw_full_mesh: bool = False,
    ) -> np.ndarray:
        """
        Draw landmarks / mesh onto bgr_frame (in-place) and return it.

        Args:
            bgr_frame:       original BGR frame to draw on.
            rgb_frame:       corresponding RGB frame for MediaPipe re-inference.
            draw_full_mesh:  if True, draws all 468 connections (slow; for debug).
        Returns:
            annotated BGR frame.
        """
        if self._last_result is None:
            return bgr_frame

        if draw_full_mesh:
            for px, py in self._last_result.all_px:
                cv2.circle(bgr_frame, (px, py), 1, (120, 120, 120), -1)

        return bgr_frame

    def draw_anchors(
        self,
        bgr_frame: np.ndarray,
        result: FaceMeshResult,
        radius: int = 4,
        color: tuple[int, int, int] = (0, 255, 120),
    ) -> np.ndarray:
        """
        Draw the 6 anchor points as labelled circles.

        Args:
            bgr_frame: frame to annotate in-place.
            result:    FaceMeshResult from process().
        Returns:
            annotated BGR frame.
        """
        for name, (px, py) in result.anchor_px.items():
            cv2.circle(bgr_frame, (px, py), radius, color, -1)
            cv2.putText(
                bgr_frame,
                name[:4],
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
        return bgr_frame
