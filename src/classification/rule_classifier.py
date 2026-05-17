"""
src/classification/rule_classifier.py
─────────────────────────────────────────────────────────────────────────────
Rule-based head movement classifier.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


class RuleClassifier:
    """Rule-based head movement classifier."""

    def __init__(self) -> None:
        log.debug("RuleClassifier initialised")

    def classify(self, angles: list[dict[str, float]]) -> dict[str, Any]:
        """
        Classify a sequence of angle measurements.

        Args:
            angles: List of dicts with keys 'yaw', 'pitch', 'roll'

        Returns:
            Dict with 'label', 'confidence', and 'features' keys
        """
        if len(angles) < 2:
            return {"label": "STATIC", "confidence": 1.0, "features": {}}

        yaws = np.array([a.get("yaw", 0.0) for a in angles])
        pitches = np.array([a.get("pitch", 0.0) for a in angles])
        rolls = np.array([a.get("roll", 0.0) for a in angles])

        # Compute features
        yaw_std = float(np.std(yaws))
        pitch_std = float(np.std(pitches))
        roll_std = float(np.std(rolls))
        yaw_range = float(np.max(yaws) - np.min(yaws))
        pitch_range = float(np.max(pitches) - np.min(pitches))
        roll_range = float(np.max(rolls) - np.min(rolls))

        # Zero crossings
        yaw_crossings = self._count_zero_crossings(yaws)
        pitch_crossings = self._count_zero_crossings(pitches)

        features = {
            "pitch_std": pitch_std,
            "yaw_std": yaw_std,
            "roll_std": roll_std,
            "pitch_zero_crossings": pitch_crossings,
            "yaw_zero_crossings": yaw_crossings,
            "pitch_range": pitch_range,
            "yaw_range": yaw_range,
            "roll_range": roll_range,
        }

        # Simple classification logic
        label = "STATIC"
        confidence = 1.0

        if pitch_range > 10.0 and pitch_crossings >= 2 and roll_std < 3.0:
            label = "NOD"
            confidence = min(0.95, 0.5 + pitch_range / 30.0)
        elif yaw_range > 10.0 and yaw_crossings >= 2 and pitch_std < 3.0:
            label = "SHAKE"
            confidence = min(0.95, 0.5 + yaw_range / 30.0)
        elif roll_range > 15.0 and np.mean(rolls) < -5.0:
            label = "TILT_LEFT"
            confidence = 0.8
        elif roll_range > 15.0 and np.mean(rolls) > 5.0:
            label = "TILT_RIGHT"
            confidence = 0.8
        else:
            if yaw_std < 2.0 and pitch_std < 2.0 and roll_std < 2.0:
                confidence = 0.95
            else:
                confidence = max(0.5, 1.0 - (yaw_std + pitch_std + roll_std) / 10.0)

        return {"label": label, "confidence": float(confidence), "features": features}

    @staticmethod
    def _count_zero_crossings(signal: np.ndarray) -> int:
        """Count zero crossings in a signal."""
        if len(signal) < 2:
            return 0
        crossings = np.sum(np.diff(np.sign(signal - np.mean(signal))) != 0)
        return int(crossings)
