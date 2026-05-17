from __future__ import annotations

from collections import deque

import numpy as np

from src.utils.config import cfg


class AngleBuffer:
    """Fixed-size rolling buffer for yaw, pitch, and roll signals."""

    def __init__(self, maxlen: int | None = None) -> None:
        window = int(getattr(cfg.classification, "window_frames", 30))
        pose_window = int(getattr(cfg.pose_estimation, "angle_buffer_size", window))
        self.maxlen = int(maxlen or max(window, pose_window))
        self._yaw: deque[float] = deque(maxlen=self.maxlen)
        self._pitch: deque[float] = deque(maxlen=self.maxlen)
        self._roll: deque[float] = deque(maxlen=self.maxlen)

    def push(self, yaw: float, pitch: float, roll: float) -> None:
        self._yaw.append(float(yaw))
        self._pitch.append(float(pitch))
        self._roll.append(float(roll))

    def clear(self) -> None:
        self._yaw.clear()
        self._pitch.clear()
        self._roll.clear()

    @property
    def size(self) -> int:
        return len(self._yaw)

    @property
    def is_full(self) -> bool:
        return self.size >= self.maxlen

    @property
    def latest(self) -> tuple[float, float, float] | None:
        if self.size == 0:
            return None
        return (self._yaw[-1], self._pitch[-1], self._roll[-1])

    @property
    def yaw_arr(self) -> np.ndarray:
        return np.asarray(self._yaw, dtype=np.float32)

    @property
    def pitch_arr(self) -> np.ndarray:
        return np.asarray(self._pitch, dtype=np.float32)

    @property
    def roll_arr(self) -> np.ndarray:
        return np.asarray(self._roll, dtype=np.float32)

    @property
    def yaw_range(self) -> float:
        arr = self.yaw_arr
        return 0.0 if arr.size == 0 else float(arr.max() - arr.min())

    @property
    def pitch_range(self) -> float:
        arr = self.pitch_arr
        return 0.0 if arr.size == 0 else float(arr.max() - arr.min())

    @property
    def roll_range(self) -> float:
        arr = self.roll_arr
        return 0.0 if arr.size == 0 else float(arr.max() - arr.min())

    @property
    def yaw_variance(self) -> float:
        arr = self.yaw_arr
        return 0.0 if arr.size < 2 else float(np.var(arr))

    @property
    def pitch_variance(self) -> float:
        arr = self.pitch_arr
        return 0.0 if arr.size < 2 else float(np.var(arr))

    @property
    def roll_variance(self) -> float:
        arr = self.roll_arr
        return 0.0 if arr.size < 2 else float(np.var(arr))

    @staticmethod
    def _zero_crossings(values: np.ndarray) -> int:
        if values.size < 3:
            return 0
        arr = values.copy()
        arr[np.abs(arr) < 1e-5] = 0.0
        signs = np.sign(arr)
        # Fill zero sign entries with previous non-zero sign to avoid false crossings.
        for i in range(1, signs.size):
            if signs[i] == 0.0:
                signs[i] = signs[i - 1]
        return int(np.sum(signs[1:] * signs[:-1] < 0))

    @property
    def yaw_zero_crossings(self) -> int:
        return self._zero_crossings(self.yaw_arr)

    @property
    def pitch_zero_crossings(self) -> int:
        return self._zero_crossings(self.pitch_arr)

    @property
    def roll_zero_crossings(self) -> int:
        return self._zero_crossings(self.roll_arr)
