"""
src/pipeline/optical_flow.py
Lucas-Kanade sparse optical flow on facial landmark keypoints.

Tracks the 6 anchor points across consecutive frames to obtain:
  - per-point velocity vectors (dx, dy) in pixels/frame
  - mean flow direction and magnitude
  - a visual overlay for debugging

This runs alongside (not instead of) the pose estimator — the flow
vectors provide a fast motion-onset signal for the Phase 4 state machine.

Usage:
    from src.pipeline.optical_flow import OpticalFlowTracker

    tracker = OpticalFlowTracker()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    for frame in stream:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_result = tracker.compute(prev_gray, gray, anchor_px_list)
        prev_gray = gray
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from config import cfg
from logger import get_logger

log = get_logger(__name__)


@dataclass
class FlowResult:
    """Per-frame optical flow output."""

    # {anchor_name: (dx, dy)} velocities in pixels/frame
    velocities: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Mean flow magnitude across all tracked points (pixels/frame)
    mean_magnitude: float = 0.0

    # Mean flow direction in degrees (0° = right, 90° = up)
    mean_direction_deg: float = 0.0

    # Points successfully tracked (others were lost)
    tracked_names: list[str] = field(default_factory=list)


class OpticalFlowTracker:
    """
    Sparse Lucas-Kanade optical flow tracker for facial anchor points.

    Does NOT maintain internal state between frames — the caller passes
    prev_gray and curr_gray explicitly, making it stateless and easy to reset.
    """

    def __init__(self) -> None:
        oc = cfg.optical_flow
        self._lk_params = dict(
            winSize=tuple(oc.win_size),
            maxLevel=oc.max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

    def compute(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        anchor_px: dict[str, tuple[int, int]],
    ) -> FlowResult:
        """
        Compute LK flow for each anchor point.

        Args:
            prev_gray:  grayscale frame at t-1.
            curr_gray:  grayscale frame at t.
            anchor_px:  {name: (px_x, px_y)} from FaceMeshResult.
        Returns:
            FlowResult with per-anchor velocities.
        """
        if not anchor_px:
            return FlowResult()

        names = list(anchor_px.keys())
        p0 = np.array(
            [[list(anchor_px[n])] for n in names],
            dtype=np.float32,
        )  # shape: (N, 1, 2)

        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None, **self._lk_params
        )

        velocities: dict[str, tuple[float, float]] = {}
        tracked: list[str] = []
        flow_vectors: list[tuple[float, float]] = []

        for i, name in enumerate(names):
            if status is not None and status[i, 0] == 1 and p1 is not None:
                x0, y0 = p0[i, 0]
                x1, y1 = p1[i, 0]
                dx, dy = float(x1 - x0), float(y1 - y0)
                velocities[name] = (dx, dy)
                tracked.append(name)
                flow_vectors.append((dx, dy))
            else:
                velocities[name] = (0.0, 0.0)

        # Aggregate stats
        mean_mag = 0.0
        mean_dir = 0.0
        if flow_vectors:
            vecs = np.array(flow_vectors)
            mags = np.linalg.norm(vecs, axis=1)
            mean_mag = float(np.mean(mags))
            mean_dx, mean_dy = vecs.mean(axis=0)
            # Convert to degrees; negate dy because image y-axis is inverted
            mean_dir = float(np.degrees(np.arctan2(-mean_dy, mean_dx)))

        return FlowResult(
            velocities=velocities,
            mean_magnitude=mean_mag,
            mean_direction_deg=mean_dir,
            tracked_names=tracked,
        )

    def draw_flow(
        self,
        bgr_frame: np.ndarray,
        anchor_px: dict[str, tuple[int, int]],
        flow: FlowResult,
        scale: float = 5.0,
        color: tuple[int, int, int] = (0, 200, 255),
    ) -> np.ndarray:
        """
        Draw flow vectors as arrows on bgr_frame (in-place).

        Args:
            scale: multiplier to make small motions visible.
        """
        for name, (dx, dy) in flow.velocities.items():
            if name not in anchor_px:
                continue
            x0, y0 = anchor_px[name]
            x1 = int(x0 + dx * scale)
            y1 = int(y0 + dy * scale)
            cv2.arrowedLine(
                bgr_frame,
                (x0, y0),
                (x1, y1),
                color,
                thickness=2,
                tipLength=0.4,
                line_type=cv2.LINE_AA,
            )
        return bgr_frame
