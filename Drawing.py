"""
src/utils/drawing.py
All OpenCV drawing helpers: HUD panel, angle gauges, gesture label,
FPS counter, and axes overlay.  Pure functions — no state.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
GREEN  = (0,  255, 120)
YELLOW = (0,  220, 255)
RED    = (0,   60, 220)
BLUE   = (220, 140,  0)
WHITE  = (230, 230, 230)
DARK   = (30,  30,  30)
GRAY   = (120, 120, 120)


def draw_hud(
    frame: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    fps: float = 0.0,
    label: str = "",
    confidence: float = 0.0,
) -> np.ndarray:
    """
    Draw a semi-transparent HUD panel in the top-left corner showing:
      - Yaw / Pitch / Roll with bar gauges
      - Current gesture label + confidence
      - FPS counter

    Args:
        frame:      BGR frame to annotate (modified in-place).
        yaw:        head yaw in degrees   (-: left,  +: right).
        pitch:      head pitch in degrees (-: up,    +: down).
        roll:       head roll in degrees  (-: left,  +: right).
        fps:        current frames-per-second.
        label:      classified gesture string (e.g. "NOD").
        confidence: classifier confidence 0.0-1.0.
    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]
    panel_w, panel_h = 240, 175
    x0, y0 = 10, 10

    # Semi-transparent background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def _angle_bar(name: str, val: float, y: int, range_deg: float = 45.0) -> None:
        lx = x0 + 8
        bar_w = panel_w - 90
        bar_h = 10
        bx = x0 + 70
        # Label
        cv2.putText(frame, name, (lx, y + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, GRAY, 1, cv2.LINE_AA)
        # Track
        cv2.rectangle(frame, (bx, y), (bx + bar_w, y + bar_h), (60, 60, 60), -1)
        # Fill (clamped)
        frac = max(-1.0, min(1.0, val / range_deg))
        mid = bx + bar_w // 2
        fill_w = int(abs(frac) * (bar_w // 2))
        if frac >= 0:
            cv2.rectangle(frame, (mid, y), (mid + fill_w, y + bar_h), GREEN, -1)
        else:
            cv2.rectangle(frame, (mid - fill_w, y), (mid, y + bar_h), YELLOW, -1)
        # Centre tick
        cv2.line(frame, (mid, y - 2), (mid, y + bar_h + 2), WHITE, 1)
        # Value text
        cv2.putText(frame, f"{val:+.1f}", (bx + bar_w + 4, y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

    _angle_bar("Yaw",   yaw,   y0 + 18)
    _angle_bar("Pitch", pitch, y0 + 42)
    _angle_bar("Roll",  roll,  y0 + 66)

    # Gesture label
    label_color = GREEN if label else GRAY
    cv2.putText(frame, f"Gesture: {label or '—'}",
                (x0 + 8, y0 + 102), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, label_color, 1, cv2.LINE_AA)

    # Confidence bar
    if confidence > 0:
        bar_total = panel_w - 16
        bar_fill = int(confidence * bar_total)
        cv2.rectangle(frame, (x0 + 8, y0 + 112), (x0 + 8 + bar_total, y0 + 122),
                      (60, 60, 60), -1)
        cv2.rectangle(frame, (x0 + 8, y0 + 112), (x0 + 8 + bar_fill, y0 + 122),
                      GREEN, -1)
        cv2.putText(frame, f"{confidence * 100:.0f}%",
                    (x0 + bar_total + 12, y0 + 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS {fps:.1f}",
                (x0 + 8, y0 + 160), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, GRAY, 1, cv2.LINE_AA)

    return frame


def draw_axes(
    frame: np.ndarray,
    rotation_vec: np.ndarray,
    translation_vec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    nose_px: tuple[int, int],
    axis_length: float = 60.0,
) -> np.ndarray:
    """
    Project and draw 3D X/Y/Z axes onto the face (nose tip origin).

    X = red  (right)
    Y = green (up — negated for image coords)
    Z = blue  (towards camera / out of screen)
    """
    axis_3d = np.float32([
        [axis_length, 0, 0],
        [0, -axis_length, 0],   # negative Y = up in image space
        [0, 0, axis_length],
    ])

    pts, _ = cv2.projectPoints(
        axis_3d, rotation_vec, translation_vec, camera_matrix, dist_coeffs
    )
    pts = pts.reshape(-1, 2).astype(int)

    nose = nose_px
    cv2.arrowedLine(frame, nose, tuple(pts[0]), (0, 0, 220),   2, tipLength=0.2, line_type=cv2.LINE_AA)  # X red
    cv2.arrowedLine(frame, nose, tuple(pts[1]), (0, 220, 0),   2, tipLength=0.2, line_type=cv2.LINE_AA)  # Y green
    cv2.arrowedLine(frame, nose, tuple(pts[2]), (220, 100, 0), 2, tipLength=0.2, line_type=cv2.LINE_AA)  # Z blue

    return frame


def put_gesture_banner(
    frame: np.ndarray,
    label: str,
    color: tuple[int, int, int] = GREEN,
) -> np.ndarray:
    """
    Draw a large gesture label banner at the bottom centre of the frame.
    Used when a confirmed gesture is emitted.
    """
    h, w = frame.shape[:2]
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)[0]
    tx = (w - text_size[0]) // 2
    ty = h - 30

    # Shadow
    cv2.putText(frame, label, (tx + 2, ty + 2),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, DARK, 3, cv2.LINE_AA)
    # Text
    cv2.putText(frame, label, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2, cv2.LINE_AA)

    return frame