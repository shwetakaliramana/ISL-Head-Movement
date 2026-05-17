"""
src/utils/camera.py
Thin wrapper around cv2.VideoCapture that applies config settings
and exposes a context-manager interface.

Usage:
    from src.utils.camera import Camera

    with Camera() as cam:
        for frame in cam:
            # frame is already flipped (if cfg.camera.flip_horizontal)
            # and in BGR uint8 — ready for MediaPipe or display
            process(frame)
"""

from __future__ import annotations

import cv2
import numpy as np

from config import cfg
from logger import get_logger

log = get_logger(__name__)


class Camera:
    """
    Config-driven webcam wrapper.

    Attributes:
        cap (cv2.VideoCapture): the underlying capture object.
        width, height, fps: actual values set on the device.
    """

    def __init__(self, device_id: int | None = None) -> None:
        self._device_id = device_id if device_id is not None else cfg.camera.device_id
        self.cap: cv2.VideoCapture | None = None
        self.width = cfg.camera.width
        self.height = cfg.camera.height
        self.fps = cfg.camera.fps

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ── Iterator ───────────────────────────────────────────────────────────────

    def __iter__(self):
        """Yield frames one by one until 'q' is pressed or capture fails."""
        while True:
            frame = self.read()
            if frame is None:
                log.warning("Empty frame received — stopping iteration.")
                break
            yield frame

    # ── Public API ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the capture device and apply resolution / FPS settings."""
        self.cap = cv2.VideoCapture(self._device_id)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {self._device_id}. "
                "Check device_id in configs/config.yaml."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read back actual values (device may not honour the request exactly)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        log.info(
            "Camera %d opened: %dx%d @ %.1f fps",
            self._device_id, self.width, self.height, self.fps,
        )

    def read(self) -> np.ndarray | None:
        """
        Read and optionally flip one frame.

        Returns:
            BGR uint8 ndarray, or None on failure.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        if cfg.camera.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return frame

    def release(self) -> None:
        """Release the capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            log.info("Camera %d released.", self._device_id)

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return self.width, self.height
