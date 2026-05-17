"""
scripts/verify_phase1.py
─────────────────────────────────────────────────────────────────────────────
Phase 1 verification script.

What it does:
  • Opens your webcam (device_id from configs/config.yaml)
  • Runs MediaPipe FaceMesh → extracts 468 landmarks
  • Highlights the 6 anchor points used for pose estimation
  • Computes Lucas-Kanade optical flow on anchor points
  • Draws flow vectors + minimal HUD (no angles yet — that's Phase 3)
  • Prints a live landmark count and FPS to console

Controls:
  q    quit
  m    toggle full mesh overlay (slow — debug only)
  f    toggle optical flow arrows
  s    save current frame to data/samples/frame_NNNN.jpg

Run from the project root:
  python scripts/verify_phase1.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ── Make src importable when running from project root ────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from face_mesh import FaceMeshDetector
from optical_flow import OpticalFlowTracker
from camera import Camera
from config import cfg
from drawing import put_gesture_banner
from logger import get_logger

log = get_logger(__name__)

SAMPLES_DIR = Path("data/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    log.info("=== Phase 1 Verification ===")
    log.info("Controls: q=quit  m=toggle mesh  f=toggle flow  s=save frame")

    show_mesh = cfg.display.show_mesh
    show_flow = cfg.display.show_flow
    frame_count = 0
    saved_count = 0

    prev_gray: np.ndarray | None = None
    fps_times: list[float] = []

    tracker = OpticalFlowTracker()

    with Camera() as cam, FaceMeshDetector() as detector:
        log.info("Camera: %dx%d @ %.0f fps", cam.width, cam.height, cam.fps)

        for frame in cam:
            t_start = time.perf_counter()
            frame_count += 1

            # ── Convert colour spaces ─────────────────────────────────────────
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Face mesh detection ───────────────────────────────────────────
            result = detector.process(rgb)

            if result and result.found:
                # Draw mesh / contours
                if show_mesh:
                    detector.draw_mesh(frame, rgb, draw_full_mesh=True)
                else:
                    detector.draw_mesh(frame, rgb, draw_full_mesh=False)

                # Draw anchor points
                detector.draw_anchors(frame, result)

                # ── Optical flow ──────────────────────────────────────────────
                if prev_gray is not None and show_flow:
                    flow = tracker.compute(prev_gray, gray, result.anchor_px)
                    tracker.draw_flow(frame, result.anchor_px, flow)

                    # Print flow magnitude to console every 30 frames
                    if frame_count % 30 == 0:
                        log.info(
                            "Frame %4d | landmarks=%d | flow_mag=%.2f px/frame",
                            frame_count,
                            len(result.landmarks_norm),
                            flow.mean_magnitude,
                        )

                # Landmark count overlay
                cv2.putText(
                    frame,
                    f"Landmarks: {len(result.landmarks_norm)}",
                    (cam.width - 180, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 120),
                    1,
                    cv2.LINE_AA,
                )

            else:
                # No face found
                cv2.putText(
                    frame,
                    "No face detected",
                    (cam.width // 2 - 90, cam.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 60, 220),
                    2,
                    cv2.LINE_AA,
                )
                if frame_count % 60 == 0:
                    log.warning("Frame %d: no face detected.", frame_count)

            # ── FPS ───────────────────────────────────────────────────────────
            fps_times.append(time.perf_counter() - t_start)
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Mode indicators
            _mode_text = f"[m]esh={'ON' if show_mesh else 'OFF'}  [f]low={'ON' if show_flow else 'OFF'}"
            cv2.putText(
                frame,
                _mode_text,
                (10, cam.height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )

            # ── Display ───────────────────────────────────────────────────────
            cv2.imshow("ISL Head Movement — Phase 1", frame)

            prev_gray = gray.copy()

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("Quit requested.")
                break
            elif key == ord("m"):
                show_mesh = not show_mesh
                log.info("Mesh overlay: %s", "ON" if show_mesh else "OFF")
            elif key == ord("f"):
                show_flow = not show_flow
                log.info("Optical flow: %s", "ON" if show_flow else "OFF")
            elif key == ord("s"):
                save_path = SAMPLES_DIR / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(save_path), frame)
                saved_count += 1
                log.info("Saved: %s", save_path)

    cv2.destroyAllWindows()
    log.info("Done. Processed %d frames. Saved %d samples.", frame_count, saved_count)


if __name__ == "__main__":
    main()
