"""
tests/test_phase1.py
Unit tests for Phase 1 — config loading, camera wrapper, face mesh result,
and optical flow tracker.  These run without a real webcam or GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─── Config tests ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_loads_without_error(self):
        from src.utils.config import cfg
        assert cfg is not None

    def test_dot_access(self):
        from src.utils.config import cfg
        assert isinstance(cfg.camera.fps, int)

    def test_landmark_indices_valid(self):
        from src.utils.config import cfg
        lm = cfg.landmarks
        for name in ["nose_tip", "chin", "left_eye_corner", "right_eye_corner"]:
            idx = getattr(lm, name)
            assert 0 <= idx < 480, f"{name} index {idx} out of range"

    def test_model_classes_count(self):
        from src.utils.config import cfg
        assert cfg.model.num_classes == len(cfg.model.classes)


# ─── Camera wrapper tests (no real device) ────────────────────────────────────

class TestCamera:
    def test_instantiates(self):
        from src.utils.camera import Camera
        cam = Camera(device_id=99)   # non-existent device, just test construction
        assert cam is not None
        assert not cam.is_open

    def test_frame_size_property(self):
        from src.utils.camera import Camera
        from src.utils.config import cfg
        cam = Camera()
        assert cam.frame_size == (cfg.camera.width, cfg.camera.height)


# ─── FaceMeshResult tests ─────────────────────────────────────────────────────

class TestFaceMeshResult:
    def test_found_false_when_empty(self):
        from src.pipeline.face_mesh import FaceMeshResult
        r = FaceMeshResult()
        assert not r.found

    def test_found_true_with_landmarks(self):
        from src.pipeline.face_mesh import FaceMeshResult
        r = FaceMeshResult(landmarks_norm=[(0.5, 0.5, 0.0)] * 468)
        assert r.found


# ─── Optical flow tests ───────────────────────────────────────────────────────

class TestOpticalFlowTracker:
    def _make_gray(self, h: int = 480, w: int = 640) -> np.ndarray:
        return (np.random.rand(h, w) * 255).astype(np.uint8)

    def test_returns_flow_result(self):
        from src.pipeline.optical_flow import OpticalFlowTracker, FlowResult
        tracker = OpticalFlowTracker()
        g1 = self._make_gray()
        g2 = self._make_gray()
        anchors = {"nose_tip": (320, 240), "chin": (320, 380)}
        result = tracker.compute(g1, g2, anchors)
        assert isinstance(result, FlowResult)

    def test_all_anchors_present_in_velocities(self):
        from src.pipeline.optical_flow import OpticalFlowTracker
        tracker = OpticalFlowTracker()
        g1 = self._make_gray()
        g2 = g1.copy()   # identical frames → zero flow
        anchors = {"nose_tip": (320, 240), "chin": (320, 380), "left_eye_corner": (260, 200)}
        result = tracker.compute(g1, g2, anchors)
        for name in anchors:
            assert name in result.velocities

    def test_zero_flow_on_identical_frames(self):
        from src.pipeline.optical_flow import OpticalFlowTracker
        tracker = OpticalFlowTracker()
        # Use a structured frame (not pure noise) so LK has features to track
        g = np.zeros((480, 640), dtype=np.uint8)
        cv2_available = True
        try:
            import cv2
            g = cv2.GaussianBlur(
                (np.random.rand(480, 640) * 255).astype(np.uint8), (21, 21), 0
            )
        except ImportError:
            cv2_available = False

        if cv2_available:
            anchors = {"nose_tip": (320, 240)}
            result = tracker.compute(g, g.copy(), anchors)
            dx, dy = result.velocities["nose_tip"]
            assert abs(dx) < 2.0 and abs(dy) < 2.0, "Expected near-zero flow on identical frames"

    def test_empty_anchors_returns_default(self):
        from src.pipeline.optical_flow import OpticalFlowTracker, FlowResult
        tracker = OpticalFlowTracker()
        g = self._make_gray()
        result = tracker.compute(g, g, {})
        assert isinstance(result, FlowResult)
        assert result.mean_magnitude == 0.0


# ─── Drawing utilities ────────────────────────────────────────────────────────

class TestDrawing:
    def test_draw_hud_returns_same_shape(self):
        from src.utils.drawing import draw_hud
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = draw_hud(frame, yaw=10.0, pitch=-5.0, roll=2.0, fps=30.0, label="NOD", confidence=0.9)
        assert out.shape == frame.shape

    def test_put_gesture_banner(self):
        from src.utils.drawing import put_gesture_banner
        import numpy as np
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = put_gesture_banner(frame, "NOD")
        assert out.shape == frame.shape
