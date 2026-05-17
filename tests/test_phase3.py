"""
tests/test_phase3.py
Unit tests for Phase 3: GestureStateMachine + RuleClassifier

Run with:
    pytest tests/test_phase3.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.classification.gesture_state import GestureFSM, GestureState
from src.classification.rule_classifier import RuleClassifier


# ──────────────────────────────────────────────
# Helpers — synthetic angle sequence generators
# ──────────────────────────────────────────────

def make_flat_angles(n=30, yaw=0.0, pitch=0.0, roll=0.0):
    """Static / IDLE sequence — all angles constant."""
    return [{"yaw": yaw, "pitch": pitch, "roll": roll} for _ in range(n)]


def make_nod_angles(n=30, amplitude=15.0):
    """Pitch oscillation simulating a nod (2 full cycles over n frames)."""
    t = np.arange(n) * 4 * np.pi / n  # Avoids endpoint (2*pi*2) which is a sine zero
    pitches = amplitude * np.sin(t)
    return [{"yaw": 0.0, "pitch": float(p), "roll": 0.0} for p in pitches]


def make_shake_angles(n=30, amplitude=15.0):
    """Yaw oscillation simulating a head shake (2 full cycles over n frames)."""
    t = np.arange(n) * 4 * np.pi / n  # Avoids endpoint (2*pi*2) which is a sine zero
    yaws = amplitude * np.sin(t)
    return [{"yaw": float(y), "pitch": 0.0, "roll": 0.0} for y in yaws]


def make_tilt_left_angles(n=30, magnitude=20.0):
    """Sustained left tilt — roll goes negative and stays there."""
    ramp = np.clip(np.linspace(0, -magnitude, n), -magnitude, 0)
    return [{"yaw": 0.0, "pitch": 0.0, "roll": float(r)} for r in ramp]


def make_tilt_right_angles(n=30, magnitude=20.0):
    """Sustained right tilt — roll goes positive and stays there."""
    ramp = np.clip(np.linspace(0, magnitude, n), 0, magnitude)
    return [{"yaw": 0.0, "pitch": 0.0, "roll": float(r)} for r in ramp]


def make_noisy_flat(n=30, noise_std=1.5):
    """Flat angles with low-level noise — should still classify as STATIC."""
    rng = np.random.default_rng(42)
    return [
        {
            "yaw":   float(rng.normal(0, noise_std)),
            "pitch": float(rng.normal(0, noise_std)),
            "roll":  float(rng.normal(0, noise_std)),
        }
        for _ in range(n)
    ]


# ──────────────────────────────────────────────
# GestureStateMachine tests
# ──────────────────────────────────────────────

class TestGestureFSM:

    def test_initial_state_is_idle(self):
        fsm = GestureFSM()
        assert fsm.ctx.state == GestureState.IDLE

    def test_transitions_to_moving_on_motion(self):
        """FSM should leave IDLE when angle deltas exceed threshold."""
        fsm = GestureFSM()
        angles = make_nod_angles(n=5)
        for a in angles:
            fsm.update(a["yaw"], a["pitch"], a["roll"])
        # After a few frames of real motion the FSM must have left IDLE
        assert fsm.ctx.state != GestureState.IDLE

    def test_returns_to_idle_on_flat(self):
        """After motion stops, FSM should eventually settle back to IDLE."""
        fsm = GestureFSM()
        # Feed motion
        for a in make_nod_angles(n=30):
            fsm.update(a["yaw"], a["pitch"], a["roll"])
        # Feed stillness
        for a in make_flat_angles(n=60):
            fsm.update(a["yaw"], a["pitch"], a["roll"])
        assert fsm.ctx.state == GestureState.IDLE

    def test_emit_fires_at_most_once_per_gesture(self):
        """EMIT state must not re-fire on the same gesture window."""
        fsm = GestureFSM()
        emit_count = 0
        for a in make_nod_angles(n=60):
            result = fsm.update(a["yaw"], a["pitch"], a["roll"])
            if result is not None:
                emit_count += 1
        # A single gesture sequence must produce exactly one emit
        assert emit_count <= 1

    def test_state_enum_values_exist(self):
        expected = {"IDLE", "MOVING", "CLASSIFYING", "CONFIRMED", "EMIT"}
        actual = {s.name for s in GestureState}
        assert expected.issubset(actual)

    def test_reset_restores_idle(self):
        fsm = GestureFSM()
        for a in make_nod_angles(n=30):
            fsm.update(a["yaw"], a["pitch"], a["roll"])
        fsm.reset()
        assert fsm.ctx.state == GestureState.IDLE


# ──────────────────────────────────────────────
# RuleClassifier tests
# ──────────────────────────────────────────────

class TestRuleClassifier:

    @pytest.fixture
    def clf(self):
        return RuleClassifier()

    # --- NOD ---
    def test_classifies_nod(self, clf):
        angles = make_nod_angles(n=30, amplitude=18.0)
        result = clf.classify(angles)
        assert result["label"] == "NOD", f"Expected NOD, got {result['label']}"

    def test_nod_confidence_high(self, clf):
        angles = make_nod_angles(n=30, amplitude=20.0)
        result = clf.classify(angles)
        assert result["confidence"] >= 0.6, (
            f"Confidence too low for clear nod: {result['confidence']:.2f}"
        )

    # --- SHAKE ---
    def test_classifies_shake(self, clf):
        angles = make_shake_angles(n=30, amplitude=18.0)
        result = clf.classify(angles)
        assert result["label"] == "SHAKE", f"Expected SHAKE, got {result['label']}"

    def test_shake_confidence_high(self, clf):
        angles = make_shake_angles(n=30, amplitude=20.0)
        result = clf.classify(angles)
        assert result["confidence"] >= 0.6

    # --- TILT ---
    def test_classifies_tilt_left(self, clf):
        angles = make_tilt_left_angles(n=30, magnitude=22.0)
        result = clf.classify(angles)
        assert result["label"] == "TILT_LEFT", f"Expected TILT_LEFT, got {result['label']}"

    def test_classifies_tilt_right(self, clf):
        angles = make_tilt_right_angles(n=30, magnitude=22.0)
        result = clf.classify(angles)
        assert result["label"] == "TILT_RIGHT", f"Expected TILT_RIGHT, got {result['label']}"

    # --- STATIC ---
    def test_classifies_static_on_flat(self, clf):
        angles = make_flat_angles(n=30)
        result = clf.classify(angles)
        assert result["label"] == "STATIC", f"Expected STATIC, got {result['label']}"

    def test_classifies_static_on_noisy_flat(self, clf):
        angles = make_noisy_flat(n=30, noise_std=1.2)
        result = clf.classify(angles)
        assert result["label"] == "STATIC", (
            f"Low-noise input should be STATIC, got {result['label']}"
        )

    # --- Output schema ---
    def test_output_has_required_keys(self, clf):
        angles = make_nod_angles()
        result = clf.classify(angles)
        for key in ("label", "confidence", "features"):
            assert key in result, f"Missing key '{key}' in classifier output"

    def test_confidence_in_unit_range(self, clf):
        for gen in [make_nod_angles, make_shake_angles,
                    make_tilt_left_angles, make_flat_angles]:
            result = clf.classify(gen())
            assert 0.0 <= result["confidence"] <= 1.0, (
                f"Confidence out of [0,1]: {result['confidence']}"
            )

    def test_label_is_valid_class(self, clf):
        valid = {"NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"}
        for gen in [make_nod_angles, make_shake_angles,
                    make_tilt_left_angles, make_tilt_right_angles, make_flat_angles]:
            result = clf.classify(gen())
            assert result["label"] in valid, f"Unknown label: {result['label']}"

    def test_empty_window_returns_static(self, clf):
        result = clf.classify([])
        assert result["label"] == "STATIC"

    def test_single_frame_returns_static(self, clf):
        result = clf.classify([{"yaw": 5.0, "pitch": 3.0, "roll": 1.0}])
        assert result["label"] == "STATIC"

    # --- Feature vector ---
    def test_features_dict_has_expected_keys(self, clf):
        angles = make_nod_angles()
        result = clf.classify(angles)
        expected_keys = {
            "pitch_std", "yaw_std", "roll_std",
            "pitch_zero_crossings", "yaw_zero_crossings",
            "pitch_range", "yaw_range", "roll_range"
        }
        actual_keys = set(result["features"].keys())
        assert expected_keys.issubset(actual_keys), (
            f"Missing feature keys: {expected_keys - actual_keys}"
        )

    # --- Boundary / ambiguous inputs ---
    def test_mixed_motion_does_not_crash(self, clf):
        """High noise on all axes — classifier must not throw, label must be valid."""
        rng = np.random.default_rng(0)
        angles = [
            {"yaw": float(rng.normal(0, 10)),
             "pitch": float(rng.normal(0, 10)),
             "roll": float(rng.normal(0, 10))}
            for _ in range(30)
        ]
        result = clf.classify(angles)
        assert result["label"] in {"NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"}

    def test_large_amplitude_nod_still_nod(self, clf):
        angles = make_nod_angles(n=30, amplitude=45.0)
        result = clf.classify(angles)
        assert result["label"] == "NOD"

    def test_large_amplitude_shake_still_shake(self, clf):
        angles = make_shake_angles(n=30, amplitude=45.0)
        result = clf.classify(angles)
        assert result["label"] == "SHAKE"