"""
scripts/verify_phase3.py
─────────────────────────────────────────────────────────────────────────────
Phase 3 verification script — Rule-Based Classifier + FSM.

What it shows:
  • Everything from Phase 2 (axes, HUD, strip chart, gauges)
  • FSM state badge (IDLE / MOVING / CLASSIFYING / CONFIRMED / EMIT)
  • Per-class confidence bar chart panel (right side)
  • Large gesture banner when a gesture is EMITTED
  • Running gesture log (last 8 emitted gestures with timestamps)
  • Console prints FSM transitions and all emissions

Controls:
  q  quit
  a  toggle 3D axes
  c  toggle strip chart
  g  toggle gauges
  r  reset FSM + smoothing + history
  s  save annotated frame

Run:
  python scripts/verify_phase3.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.classification.gesture_state import GestureFSM
from src.classification.rule_classifier import RuleClassifier
from src.pipeline.angle_buffer import AngleBuffer
from src.pipeline.calibration import build_approximate_intrinsics
from src.pipeline.face_mesh import FaceMeshDetector
from src.pipeline.optical_flow import OpticalFlowTracker
from src.pipeline.pose_estimator import HeadPoseEstimator
from src.utils.camera import Camera
from src.utils.config import cfg
from src.utils.drawing import (
    draw_angle_history,
    draw_axes,
    draw_hud,
    draw_three_gauges,
    put_gesture_banner,
)
from src.utils.logger import get_logger

log = get_logger(__name__)
SAMPLES_DIR = Path("data/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_LEN  = 90
GESTURE_LOG_LEN = 8

# FSM state badge colours  (BGR)
STATE_COLOURS = {
    "IDLE":        (100, 100, 100),
    "MOVING":      (0,   200, 255),
    "CLASSIFYING": (0,   165, 255),
    "CONFIRMED":   (50,  220,  50),
    "EMIT":        (0,   255, 120),
}

# Per-class bar colours
CLASS_COLOURS = {
    "NOD":        (0,   255, 120),
    "SHAKE":      (0,   220, 255),
    "TILT_LEFT":  (220, 140,   0),
    "TILT_RIGHT": (30,  165, 255),
    "STATIC":     (110, 110, 110),
}


def _draw_confidence_panel(
    frame:   np.ndarray,
    scores:  dict[str, float],
    x:       int,
    y:       int,
    w:       int = 200,
) -> None:
    """Right-side panel: horizontal bar for each class confidence."""
    bar_h  = 14
    gap    = 6
    ph     = len(scores) * (bar_h + gap) + 24
    cv2.rectangle(frame, (x - 6, y - 18), (x + w + 6, y + ph),
                  (25, 25, 25), -1)
    cv2.putText(frame, "Confidence", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    for i, (cls, conf) in enumerate(scores.items()):
        by  = y + i * (bar_h + gap)
        col = CLASS_COLOURS.get(cls, (180, 180, 180))
        # Track
        cv2.rectangle(frame, (x, by), (x + w, by + bar_h), (55, 55, 55), -1)
        # Fill
        fill = int(conf * w)
        if fill > 0:
            cv2.rectangle(frame, (x, by), (x + fill, by + bar_h), col, -1)
        # Label + value
        cv2.putText(frame, f"{cls:<11} {conf:.2f}",
                    (x + 3, by + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (210, 210, 210), 1, cv2.LINE_AA)


def _draw_fsm_badge(frame: np.ndarray, state_name: str, x: int, y: int) -> None:
    col  = STATE_COLOURS.get(state_name, (150, 150, 150))
    text = f"FSM: {state_name}"
    sz, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x - 4, y - 16), (x + sz[0] + 6, y + 4), (25, 25, 25), -1)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)


def _draw_gesture_log(
    frame:   np.ndarray,
    log_:    deque,
    x:       int,
    y:       int,
) -> None:
    cv2.putText(frame, "Recent:", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1, cv2.LINE_AA)
    for i, (ts, lbl) in enumerate(reversed(list(log_))):
        age   = time.time() - ts
        alpha = max(0.3, 1.0 - age / 8.0)   # fade older entries
        col   = tuple(int(c * alpha) for c in CLASS_COLOURS.get(lbl, (200, 200, 200)))
        cv2.putText(frame, f"{lbl}  {age:.1f}s ago",
                    (x, y + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1, cv2.LINE_AA)


def main() -> None:
    log.info("=== Phase 3 Verification — Rule-Based Classifier ===")
    log.info("Controls: q=quit  a=axes  c=chart  g=gauges  r=reset  s=save")

    show_axes   = True
    show_chart  = True
    show_gauges = True

    yaw_hist:   deque[float] = deque(maxlen=HISTORY_LEN)
    pitch_hist: deque[float] = deque(maxlen=HISTORY_LEN)
    roll_hist:  deque[float] = deque(maxlen=HISTORY_LEN)
    gesture_log: deque       = deque(maxlen=GESTURE_LOG_LEN)

    fps_window: deque[float] = deque(maxlen=30)
    prev_gray:  np.ndarray | None = None
    frame_count = 0
    saved_count = 0

    angle_buf  = AngleBuffer()
    fsm        = GestureFSM()
    rule_clf   = RuleClassifier()
    tracker    = OpticalFlowTracker()

    # Persistent banner display (fades after N frames)
    banner_label  = ""
    banner_frames = 0
    BANNER_DURATION = 45   # frames

    with Camera() as cam, FaceMeshDetector() as detector:
        w, h      = cam.frame_size
        estimator = HeadPoseEstimator(frame_width=w, frame_height=h)
        K, D      = build_approximate_intrinsics(w, h)

        conf_panel_x = w - 215
        log_x        = w - 215

        for frame in cam:
            t0 = time.perf_counter()
            frame_count += 1

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Face mesh ─────────────────────────────────────────────────────
            mesh = detector.process(rgb)
            yaw = pitch = roll = 0.0
            flow_mag = angle_delta = 0.0
            all_scores: dict[str, float] = {c: 0.0 for c in cfg.model.classes}

            if mesh and mesh.found:
                detector.draw_mesh(frame, rgb, draw_full_mesh=False)
                detector.draw_anchors(frame, mesh)

                # ── Pose ──────────────────────────────────────────────────────
                pose = estimator.estimate(mesh.anchor_px)
                if pose:
                    yaw, pitch, roll = pose.yaw, pose.pitch, pose.roll
                    angle_buf.push(yaw, pitch, roll)
                    yaw_hist.append(yaw)
                    pitch_hist.append(pitch)
                    roll_hist.append(roll)

                    if show_axes:
                        draw_axes(frame,
                                  rotation_vec=pose.rotation_vec,
                                  translation_vec=pose.translation_vec,
                                  camera_matrix=K, dist_coeffs=D,
                                  nose_px=mesh.anchor_px["nose_tip"])

                # ── Optical flow ──────────────────────────────────────────────
                if prev_gray is not None:
                    flow = tracker.compute(prev_gray, gray, mesh.anchor_px)
                    tracker.draw_flow(frame, mesh.anchor_px, flow)
                    flow_mag = flow.mean_magnitude

                # ── Rule classifier ───────────────────────────────────────────
                rule_result = None
                if angle_buf.is_full:
                    all_scores  = rule_clf.classify_all(angle_buf)
                    rule_result = rule_clf.classify(angle_buf)

                # ── FSM step ──────────────────────────────────────────────────
                prev = angle_buf.latest
                if prev and angle_buf.size >= 2:
                    dy = abs(yaw   - list(angle_buf._yaw)[-2])
                    dp = abs(pitch - list(angle_buf._pitch)[-2])
                    dr = abs(roll  - list(angle_buf._roll)[-2])
                    angle_delta = max(dy, dp, dr)

                emission = fsm.step(
                    flow_magnitude=flow_mag,
                    angle_delta=angle_delta,
                    buffer_full=angle_buf.is_full,
                    rule_result=rule_result,
                )

                if emission:
                    banner_label  = emission
                    banner_frames = BANNER_DURATION
                    gesture_log.append((time.time(), emission))
                    log.info(">>> GESTURE EMITTED: %s <<<", emission)

            else:
                estimator.reset_smoothing()
                fsm.force_reset()

            # ── FPS ───────────────────────────────────────────────────────────
            fps_window.append(time.perf_counter() - t0)
            fps = 1.0 / (sum(fps_window) / len(fps_window))

            # ── HUD overlays ──────────────────────────────────────────────────
            draw_hud(frame, yaw, pitch, roll, fps=fps,
                     label=banner_label if banner_frames > 0 else "",
                     confidence=all_scores.get(banner_label, 0.0) if banner_frames > 0 else 0.0)

            if show_chart and len(yaw_hist) > 2:
                draw_angle_history(frame, yaw_hist, pitch_hist, roll_hist,
                                   x=10, y=200, w=240, h=80)

            if show_gauges:
                draw_three_gauges(frame, yaw, pitch, roll, x0=10, y0=330)

            # FSM state badge
            _draw_fsm_badge(frame, fsm.state_name, x=10, y=h - 35)

            # Confidence panel (right side)
            _draw_confidence_panel(frame, all_scores,
                                   x=conf_panel_x, y=50, w=200)

            # Gesture log (right side, below confidence panel)
            if gesture_log:
                _draw_gesture_log(frame, gesture_log,
                                  x=log_x, y=220)

            # Banner countdown
            if banner_frames > 0:
                put_gesture_banner(frame, banner_label)
                banner_frames -= 1

            cv2.imshow("ISL Head Movement — Phase 3: Rule Classifier", frame)

            prev_gray = gray.copy()

            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"): break
            elif key == ord("a"): show_axes   = not show_axes
            elif key == ord("c"): show_chart  = not show_chart
            elif key == ord("g"): show_gauges = not show_gauges
            elif key == ord("s"):
                p = SAMPLES_DIR / f"phase3_{saved_count:04d}.jpg"
                cv2.imwrite(str(p), frame)
                saved_count += 1
                log.info("Saved: %s", p)
            elif key == ord("r"):
                fsm.force_reset()
                estimator.reset_smoothing()
                angle_buf.clear()
                yaw_hist.clear(); pitch_hist.clear(); roll_hist.clear()
                banner_label = ""; banner_frames = 0
                log.info("Full reset.")

    cv2.destroyAllWindows()
    log.info("Done. %d frames, %d gestures emitted, %d saved.",
             frame_count, len(gesture_log), saved_count)


if __name__ == "__main__":
    main()