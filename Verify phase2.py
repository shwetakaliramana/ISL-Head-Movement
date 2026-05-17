"""
scripts/verify_phase2.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 verification script — Head Pose Estimation.

What it shows on screen:
  • Live webcam feed with MediaPipe face mesh
  • 6 anchor points highlighted
  • 3D X/Y/Z axes projected onto the face (nose-tip origin)
  • Optical flow vectors on anchor points
  • Top-left HUD: Yaw / Pitch / Roll bar gauges + FPS
  • Bottom-left strip chart: angle history over the last 90 frames
  • Bottom-left gauges: three semicircular dials

Console output (every 60 frames):
  Frame NNNN | Yaw=+XX.X  Pitch=+XX.X  Roll=+XX.X | FPS=XX.X

Controls:
  q  → quit
  a  → toggle 3D axes overlay
  g  → toggle circular gauges
  c  → toggle strip chart
  m  → toggle full mesh
  f  → toggle optical flow
  s  → save annotated frame to data/samples/
  r  → reset EMA smoothing (useful after occlusion)

Run from the project root:
  python scripts/verify_phase2.py
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
)
from src.utils.logger import get_logger

log = get_logger(__name__)
SAMPLES_DIR = Path("data/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# History length for the strip chart (frames)
HISTORY_LEN = 90


def main() -> None:
    log.info("=== Phase 2 Verification — Head Pose Estimation ===")
    log.info("Controls: q=quit  a=axes  g=gauges  c=chart  m=mesh  f=flow  s=save  r=reset")

    # ── Toggle states ──────────────────────────────────────────────────────────
    show_axes   = True
    show_gauges = True
    show_chart  = True
    show_mesh   = False
    show_flow   = True

    # ── Angle history deques for strip chart ───────────────────────────────────
    yaw_hist:   deque[float] = deque(maxlen=HISTORY_LEN)
    pitch_hist: deque[float] = deque(maxlen=HISTORY_LEN)
    roll_hist:  deque[float] = deque(maxlen=HISTORY_LEN)

    # ── Misc state ─────────────────────────────────────────────────────────────
    prev_gray:    np.ndarray | None = None
    fps_window:   deque[float]      = deque(maxlen=30)
    frame_count   = 0
    saved_count   = 0
    angle_buf     = AngleBuffer()

    with Camera() as cam, FaceMeshDetector() as detector:
        w, h = cam.frame_size
        estimator = HeadPoseEstimator(frame_width=w, frame_height=h)
        K, D      = build_approximate_intrinsics(w, h)
        tracker   = OpticalFlowTracker()

        log.info("Camera: %dx%d @ %.0f fps", w, h, cam.fps)
        log.info("Pose estimator ready. Smoothing alpha=%.2f",
                 cfg.pose_estimation.smoothing_alpha)

        for frame in cam:
            t0 = time.perf_counter()
            frame_count += 1

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Face mesh ─────────────────────────────────────────────────────
            mesh_result = detector.process(rgb)

            yaw = pitch = roll = 0.0
            pose_ok = False

            if mesh_result and mesh_result.found:

                # Mesh / anchor overlay
                if show_mesh:
                    detector.draw_mesh(frame, rgb, draw_full_mesh=True)
                else:
                    detector.draw_mesh(frame, rgb, draw_full_mesh=False)
                detector.draw_anchors(frame, mesh_result)

                # ── Pose estimation ───────────────────────────────────────────
                pose = estimator.estimate(mesh_result.anchor_px)

                if pose:
                    yaw, pitch, roll = pose.yaw, pose.pitch, pose.roll
                    pose_ok = True

                    angle_buf.push(yaw, pitch, roll)
                    yaw_hist.append(yaw)
                    pitch_hist.append(pitch)
                    roll_hist.append(roll)

                    # 3D axes on face
                    if show_axes:
                        nose_px = mesh_result.anchor_px["nose_tip"]
                        draw_axes(
                            frame,
                            rotation_vec=pose.rotation_vec,
                            translation_vec=pose.translation_vec,
                            camera_matrix=K,
                            dist_coeffs=D,
                            nose_px=nose_px,
                            axis_length=65.0,
                        )

                    # Periodic console log
                    if frame_count % 60 == 0:
                        log.info(
                            "Frame %4d | Yaw=%+6.1f  Pitch=%+6.1f  Roll=%+6.1f | FPS=%.1f",
                            frame_count, yaw, pitch, roll,
                            1.0 / (sum(fps_window) / max(len(fps_window), 1) + 1e-9),
                        )

                # ── Optical flow ──────────────────────────────────────────────
                if prev_gray is not None and show_flow:
                    flow = tracker.compute(prev_gray, gray, mesh_result.anchor_px)
                    tracker.draw_flow(frame, mesh_result.anchor_px, flow)

            else:
                # No face — show warning
                cv2.putText(frame, "No face detected",
                            (w // 2 - 100, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 220), 2, cv2.LINE_AA)
                estimator.reset_smoothing()

            # ── FPS ───────────────────────────────────────────────────────────
            dt = time.perf_counter() - t0
            fps_window.append(dt)
            fps = 1.0 / (sum(fps_window) / len(fps_window))

            # ── Overlays ──────────────────────────────────────────────────────
            draw_hud(frame, yaw, pitch, roll, fps=fps)

            if show_chart and len(yaw_hist) > 2:
                draw_angle_history(
                    frame, yaw_hist, pitch_hist, roll_hist,
                    x=10, y=200, w=240, h=80,
                )

            if show_gauges and pose_ok:
                # Place gauges below the strip chart
                draw_three_gauges(frame, yaw, pitch, roll, x0=10, y0=330)

            # Mode status strip at the bottom
            status = (
                f"[a]xes={'ON' if show_axes else 'OFF'}  "
                f"[g]auges={'ON' if show_gauges else 'OFF'}  "
                f"[c]hart={'ON' if show_chart else 'OFF'}  "
                f"[m]esh={'ON' if show_mesh else 'OFF'}  "
                f"[f]low={'ON' if show_flow else 'OFF'}"
            )
            cv2.putText(frame, status, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1, cv2.LINE_AA)

            # ── Display ───────────────────────────────────────────────────────
            cv2.imshow("ISL Head Movement — Phase 2: Pose Estimation", frame)

            prev_gray = gray.copy()

            # ── Keypress ──────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"): break
            elif key == ord("a"): show_axes   = not show_axes
            elif key == ord("g"): show_gauges = not show_gauges
            elif key == ord("c"): show_chart  = not show_chart
            elif key == ord("m"): show_mesh   = not show_mesh
            elif key == ord("f"): show_flow   = not show_flow
            elif key == ord("r"):
                estimator.reset_smoothing()
                angle_buf.clear()
                yaw_hist.clear(); pitch_hist.clear(); roll_hist.clear()
                log.info("Smoothing + history reset.")
            elif key == ord("s"):
                p = SAMPLES_DIR / f"pose_frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(p), frame)
                saved_count += 1
                log.info("Saved: %s", p)

    cv2.destroyAllWindows()
    log.info("Done. %d frames processed, %d frames saved.", frame_count, saved_count)


if __name__ == "__main__":
    main()