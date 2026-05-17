"""
scripts/record_angles.py
─────────────────────────────────────────────────────────────────────────────
Angle data recorder for building the Phase 4 / Phase 5 dataset.

Opens the webcam, runs pose estimation, and writes one row per frame to:
  data/raw/angles_<session_id>.csv

Each row:   timestamp, yaw, pitch, roll, dyaw, dpitch, droll, label

Controls:
  0-4  → set the current gesture label:
            0 = STATIC
            1 = NOD
            2 = SHAKE
            3 = TILT_LEFT
            4 = TILT_RIGHT
  SPACE → toggle recording ON / OFF  (green/red indicator)
  q     → quit and save

Usage:
  python scripts/record_angles.py --session s001
  python scripts/record_angles.py           # auto-generates session ID

Output files:
  data/raw/angles_<session>.csv
  data/annotations/labels_<session>.json   (class counts + metadata)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.pipeline.calibration import build_approximate_intrinsics
from src.pipeline.face_mesh import FaceMeshDetector
from src.pipeline.pose_estimator import HeadPoseEstimator
from src.utils.camera import Camera
from src.utils.config import cfg
from src.utils.drawing import draw_hud
from src.utils.logger import get_logger

log = get_logger(__name__)

RAW_DIR  = Path("data/raw")
ANN_DIR  = Path("data/annotations")
RAW_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = cfg.model.classes          # [NOD, SHAKE, TILT_LEFT, TILT_RIGHT, STATIC]
CLASS_MAP   = {str(i): name for i, name in enumerate(CLASS_NAMES)}
KEY_MAP     = {ord(str(i)): name for i, name in enumerate(CLASS_NAMES)}

CSV_HEADER = ["timestamp", "yaw", "pitch", "roll",
              "dyaw", "dpitch", "droll", "label"]

# Display colours
RED_BGR   = (0,  60, 220)
GREEN_BGR = (0, 255, 120)
GRAY_BGR  = (120, 120, 120)


def _draw_recorder_hud(
    frame: np.ndarray,
    recording: bool,
    current_label: str,
    counts: dict[str, int],
    total: int,
) -> None:
    h, w = frame.shape[:2]

    # Recording indicator (top-right)
    rec_text = "● REC" if recording else "○ PAUSED"
    rec_col  = GREEN_BGR if recording else RED_BGR
    cv2.putText(frame, rec_text, (w - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_col, 2, cv2.LINE_AA)

    # Current label (large, centre-top)
    cv2.putText(frame, f"LABEL: {current_label}",
                (w // 2 - 100, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN_BGR, 2, cv2.LINE_AA)

    # Class counts panel (right side)
    px, py = w - 170, 60
    cv2.rectangle(frame, (px - 5, py - 15),
                  (w - 5, py + len(counts) * 20 + 10),
                  (25, 25, 25), -1)
    for i, (cls, cnt) in enumerate(counts.items()):
        active = cls == current_label
        col    = GREEN_BGR if active else GRAY_BGR
        cv2.putText(frame, f"{i}: {cls:<12} {cnt:>4}",
                    (px, py + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)

    # Total frames recorded
    cv2.putText(frame, f"Total frames: {total}",
                (px, py + len(counts) * 20 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, GRAY_BGR, 1, cv2.LINE_AA)

    # Key guide (bottom)
    guide = "0=STATIC  1=NOD  2=SHAKE  3=TILT_L  4=TILT_R  SPACE=rec  q=quit"
    cv2.putText(frame, guide, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, GRAY_BGR, 1, cv2.LINE_AA)


def main(session: str) -> None:
    csv_path = RAW_DIR  / f"angles_{session}.csv"
    ann_path = ANN_DIR  / f"labels_{session}.json"

    log.info("Session: %s", session)
    log.info("CSV output: %s", csv_path)

    recording     = False
    current_label = "STATIC"
    counts        = {name: 0 for name in CLASS_NAMES}
    rows: list[list] = []

    prev_yaw = prev_pitch = prev_roll = 0.0

    with Camera() as cam, FaceMeshDetector() as detector:
        w, h      = cam.frame_size
        estimator = HeadPoseEstimator(frame_width=w, frame_height=h)
        K, D      = build_approximate_intrinsics(w, h)

        log.info("Ready. Press SPACE to start recording, 0-4 to set label.")

        for frame in cam:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_result = detector.process(rgb)

            yaw = pitch = roll = 0.0

            if mesh_result and mesh_result.found:
                detector.draw_mesh(frame, rgb, draw_full_mesh=False)
                detector.draw_anchors(frame, mesh_result)
                pose = estimator.estimate(mesh_result.anchor_px)
                if pose:
                    yaw, pitch, roll = pose.yaw, pose.pitch, pose.roll

                    if recording:
                        dyaw   = yaw   - prev_yaw
                        dpitch = pitch - prev_pitch
                        droll  = roll  - prev_roll
                        rows.append([
                            time.time(),
                            round(yaw, 4), round(pitch, 4), round(roll, 4),
                            round(dyaw, 4), round(dpitch, 4), round(droll, 4),
                            current_label,
                        ])
                        counts[current_label] += 1

                    prev_yaw, prev_pitch, prev_roll = yaw, pitch, roll

            draw_hud(frame, yaw, pitch, roll)
            _draw_recorder_hud(frame, recording, current_label,
                                counts, sum(counts.values()))

            cv2.imshow("ISL Angle Recorder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording
                log.info("Recording: %s", "ON" if recording else "OFF")
            elif key in KEY_MAP:
                current_label = KEY_MAP[key]
                log.info("Label -> %s", current_label)

    cv2.destroyAllWindows()

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)
        log.info("Saved %d rows to %s", len(rows), csv_path)
    else:
        log.warning("No rows recorded — CSV not written.")

    # ── Save annotation metadata ───────────────────────────────────────────────
    meta = {
        "session":    session,
        "datetime":   datetime.now().isoformat(),
        "total_rows": len(rows),
        "counts":     counts,
        "csv_file":   str(csv_path),
    }
    with open(ann_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Annotation metadata saved: %s", ann_path)
    log.info("Class distribution: %s", counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Angle Data Recorder")
    parser.add_argument(
        "--session", "-s",
        default=datetime.now().strftime("s%Y%m%d_%H%M%S"),
        help="Session ID for output filenames (default: timestamp)",
    )
    args = parser.parse_args()
    main(session=args.session)