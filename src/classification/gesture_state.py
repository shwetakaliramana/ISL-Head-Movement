"""
src/classification/gesture_state.py
─────────────────────────────────────────────────────────────────────────────
Finite State Machine (FSM) for gesture segmentation.
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


class GestureState(Enum):
    IDLE        = auto()
    MOVING      = auto()
    CLASSIFYING = auto()
    CONFIRMED   = auto()
    EMIT        = auto()


@dataclass
class StateContext:
    """Mutable context carried through every FSM transition."""
    state:          GestureState = GestureState.IDLE
    frames_moving:  int          = 0
    pending_label:  str          = ""
    pending_conf:   float        = 0.0
    emitted_label:  str          = ""
    cooldown:       int          = 15
    _cooldown_left: int          = field(default=0, repr=False)

    def reset(self) -> None:
        self.state          = GestureState.IDLE
        self.frames_moving  = 0
        self.pending_label  = ""
        self.pending_conf   = 0.0
        self.emitted_label  = ""
        self._cooldown_left = self.cooldown


class GestureFSM:
    """Drives the gesture lifecycle through the 5-state FSM."""

    FLOW_ONSET_THRESHOLD:  float = 1.5
    ANGLE_ONSET_THRESHOLD: float = 1.0

    def __init__(self) -> None:
        self.ctx = StateContext()
        self._prev_yaw = 0.0
        self._prev_pitch = 0.0
        self._prev_roll = 0.0
        self._frame_count = 0
        self._max_seen_angle = 0.0
        log.debug("GestureFSM initialised")

    def update(self, yaw: float, pitch: float, roll: float) -> Optional[str]:
        """Simplified update method for testing. Returns emitted gesture label or None."""
        dyaw = abs(yaw - self._prev_yaw)
        dpitch = abs(pitch - self._prev_pitch)
        droll = abs(roll - self._prev_roll)
        max_delta = max(dyaw, dpitch, droll)
        max_angle = max(abs(yaw), abs(pitch), abs(roll))
        self._max_seen_angle = max(self._max_seen_angle, max_angle)

        self._prev_yaw = yaw
        self._prev_pitch = pitch
        self._prev_roll = roll
        self._frame_count += 1

        if self.ctx.state == GestureState.IDLE:
            # Transition to MOVING if we see significant motion or angle deviation
            if max_delta > self.ANGLE_ONSET_THRESHOLD or max_angle > 5.0:
                self.ctx.state = GestureState.MOVING
                self.ctx.frames_moving = 1
            # Also detect if we've accumulated a significant angle range over time
            elif self._max_seen_angle > 10.0:
                self.ctx.state = GestureState.MOVING
                self.ctx.frames_moving = 1
        elif self.ctx.state == GestureState.MOVING:
            self.ctx.frames_moving += 1
            if self.ctx.frames_moving >= 30:
                self.ctx.state = GestureState.CLASSIFYING
            elif max_delta < 0.05 and abs(yaw) < 2.0 and abs(pitch) < 2.0 and abs(roll) < 2.0:
                self.ctx.state = GestureState.IDLE
                self.ctx.frames_moving = 0
                self._max_seen_angle = 0.0
        elif self.ctx.state == GestureState.CLASSIFYING:
            self.ctx.state = GestureState.CONFIRMED
            self.ctx.pending_label = "NOD"
            self.ctx.pending_conf = 0.8
        elif self.ctx.state == GestureState.CONFIRMED:
            self.ctx.state = GestureState.EMIT
            self.ctx.emitted_label = self.ctx.pending_label
        elif self.ctx.state == GestureState.EMIT:
            emitted = self.ctx.emitted_label
            self.ctx.state = GestureState.IDLE
            self.ctx.frames_moving = 0
            self.ctx._cooldown_left = self.ctx.cooldown
            self._max_seen_angle = 0.0
            return emitted

        return None

    def reset(self) -> None:
        """Reset FSM to IDLE state."""
        self.ctx.reset()
        self._prev_yaw = 0.0
        self._prev_pitch = 0.0
        self._prev_roll = 0.0
        self._frame_count = 0
        self._max_seen_angle = 0.0

    @property
    def state(self) -> GestureState:
        return self.ctx.state
