"""Streamlit app for ISL head-movement analysis.

The app runs the camera capture and inference loop in the Streamlit main
thread and refreshes `st.empty()` placeholders on every iteration.
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mapping.isl_text_engine import ISLTextEngine
from src.ml.feature_engineering import CLASSES, WINDOW, N_FEAT, DatasetBuilder
from src.pipeline.face_mesh import FaceMeshDetector
from src.pipeline.pose_estimator import HeadPoseEstimator
from src.utils.config import cfg

ROLL_CLIP = 60.0
MODEL_PATH = "models/bilstm_best.keras"
STATS_PATH = "models/normaliser_stats.npz"
GRAMMAR = "configs/isl_grammar.json"
DEBOUNCE_N = 3
INFER_EVERY_N_FRAMES = 5
TILT_ROLL_THRESHOLD = abs(float(cfg.classification.tilt_left.roll_threshold))
TILT_MIN_FRAMES = int(cfg.classification.tilt_left.min_duration_frames)

COLOR_HEX = {
    "NOD": "#00D200",
    "SHAKE": "#0050DC",
    "TILT_LEFT": "#DC8C00",
    "TILT_RIGHT": "#A000DC",
    "STATIC": "#808080",
}


@st.cache_resource
def load_model():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH, safe_mode=False)


@st.cache_resource
def load_builder():
    return DatasetBuilder.load_stats(STATS_PATH)


@st.cache_resource
def load_detector():
    return FaceMeshDetector(), HeadPoseEstimator(frame_width=640, frame_height=480)


def build_feat(angle_buf, builder):
    raw = np.array(angle_buf, dtype=np.float32)
    dyaw = np.gradient(raw[:, 0]).astype(np.float32)
    dpitch = np.gradient(raw[:, 1]).astype(np.float32)
    droll = np.gradient(raw[:, 2]).astype(np.float32)
    feat6 = np.stack([raw[:, 0], raw[:, 1], raw[:, 2], dyaw, dpitch, droll], axis=1)
    return builder.transform_window(feat6)


def gauge(label, val, lo=-45, hi=45):
    pct = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    fill = int(pct * 24)
    bar = "█" * fill + "░" * (24 - fill)
    return f"`{label}` `{bar}` **{val:+.1f}°**"


st.set_page_config(page_title="ISL Head Movement", page_icon="🤟", layout="wide")

with st.sidebar:
    st.title("🤟 ISL Head Analysis")
    st.markdown("---")
    camera_id = st.number_input("Camera index", 0, 4, 0, step=1)
    threshold = st.slider(
        "Confidence threshold",
        0.30,
        0.95,
        0.55,
        0.05,
        help="Lower = more sensitive. Raise if false triggers.",
    )
    start_btn = st.button("▶ Start camera", use_container_width=True)
    stop_btn = st.button("⏹ Stop", use_container_width=True)
    st.markdown("---")
    st.markdown(
        "**Gestures**\n- Nod = YES\n- Shake = NO\n"
        "- Tilt L = ?\n- Tilt R = !\n- Tilt L + Nod = REALLY?"
    )

cam_col, info_col = st.columns([3, 2])

with cam_col:
    st.subheader("Live Feed")
    frame_ph = st.empty()
    status_ph = st.empty()

with info_col:
    st.subheader("Pose angles")
    yaw_ph, pitch_ph, roll_ph = st.empty(), st.empty(), st.empty()
    st.markdown("---")
    st.subheader("Gesture")
    badge_ph = st.empty()
    conf_ph = st.empty()
    lat_ph = st.empty()
    st.markdown("---")
    st.subheader("ISL Text")
    live_ph = st.empty()
    buf_ph = st.empty()
    sent_ph = st.empty()
    hist_ph = st.empty()


if "running" not in st.session_state:
    st.session_state.running = False

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

if not st.session_state.running:
    frame_ph.info("Press **▶ Start camera** in the sidebar.")
    st.stop()


model = load_model()
builder = load_builder()
detector, estimator = load_detector()
engine = ISLTextEngine(grammar_path=GRAMMAR)

cap = cv2.VideoCapture(int(camera_id), cv2.CAP_DSHOW)
if not cap.isOpened():
    st.error(
        f"❌ Cannot open camera {int(camera_id)}.\n\n"
        "**Try these fixes:**\n"
        "1. Change camera index in the sidebar (try 1 or 2)\n"
        "2. Close any other app using the webcam (Teams, Zoom, etc.)\n"
        "3. On Windows: Settings → Privacy → Camera → allow apps\n"
        "4. Restart Streamlit with `streamlit run app/streamlit_app.py`"
    )
    st.stop()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

angle_buf = deque(maxlen=WINDOW)
debounce_buf = deque(maxlen=DEBOUNCE_N)
last_emitted = "STATIC"
gesture = "STATIC"
confidence = 0.0
latency_ms = 0.0
probs = [0.2] * len(CLASSES)
frame_index = 0

status_ph.success("🟢 Camera running — move your head")

try:
    with detector:
        while st.session_state.running:
            frame_index += 1
            ok, frame = cap.read()
            if not ok:
                status_ph.warning("⚠️ Camera read failed — retrying...")
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector.process(rgb)

            yaw = pitch = roll = 0.0
            if res:
                angles = estimator.estimate(res.anchor_px)
                if angles:
                    yaw = angles.yaw
                    pitch = angles.pitch
                    roll = angles.roll
                    angle_buf.append([yaw, pitch, roll])

            push_result = None
            recent_roll = np.array([row[2] for row in angle_buf], dtype=np.float32)
            tilt_override = None
            if recent_roll.size >= TILT_MIN_FRAMES:
                tail = recent_roll[-TILT_MIN_FRAMES:]
                mean_roll = float(np.mean(tail))
                if np.all(tail <= -TILT_ROLL_THRESHOLD):
                    tilt_override = ("TILT_LEFT", min(0.99, 0.70 + abs(mean_roll) / 50.0))
                elif np.all(tail >= TILT_ROLL_THRESHOLD):
                    tilt_override = ("TILT_RIGHT", min(0.99, 0.70 + abs(mean_roll) / 50.0))

            if tilt_override is not None:
                gesture, confidence = tilt_override
                probs = [0.0] * len(CLASSES)
                probs[CLASSES.index(gesture)] = confidence
                latency_ms = 0.0
            elif len(angle_buf) == WINDOW and frame_index % INFER_EVERY_N_FRAMES == 0:
                x_in = build_feat(angle_buf, builder)
                t0 = time.perf_counter()
                p = model.predict(x_in, verbose=0)[0]
                latency_ms = (time.perf_counter() - t0) * 1000
                probs = p.tolist()
                idx = int(np.argmax(p))
                gesture = CLASSES[idx]
                confidence = float(p[idx])

                debounce_buf.append(gesture)
                if (
                    len(debounce_buf) == DEBOUNCE_N
                    and len(set(debounce_buf)) == 1
                    and gesture != last_emitted
                    and confidence >= threshold
                ):
                    push_result = engine.push(gesture)
                    last_emitted = gesture

            tick_result = engine.tick()

            frame_ph.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

            yaw_ph.markdown(gauge("YAW  ", yaw))
            pitch_ph.markdown(gauge("PITCH", pitch))
            roll_ph.markdown(gauge("ROLL ", roll))

            color = COLOR_HEX.get(gesture, "#808080")
            show = gesture if confidence >= threshold else "..."
            badge_ph.markdown(
                f"<h2 style='color:{color};margin:0'>{show}</h2>",
                unsafe_allow_html=True,
            )
            conf_ph.progress(float(confidence), text=f"Confidence: {confidence:.2f}")
            lat_ph.caption(f"Latency: {latency_ms:.1f} ms")

            if push_result and (push_result.sentence or push_result.token):
                live_text = push_result.sentence or push_result.token
            elif tick_result and tick_result.sentence:
                live_text = tick_result.sentence
            else:
                live_text = engine.buffer_preview or "—"

            live_ph.markdown(
                f"<h3 style='color:#FFD166;margin:0'>{live_text}</h3>",
                unsafe_allow_html=True,
            )
            buf_ph.markdown(f"**Buffer:** {engine.buffer_preview or '(empty)'}")
            last = engine.history[-1][1] if engine.history else "—"
            sent_ph.markdown(
                f"<div style='color:#50DC80;font-size:1.1rem'>Latest sentence: {last}</div>",
                unsafe_allow_html=True,
            )
            hist = [s for _, s in engine.history[-5:]]
            hist_ph.markdown("\n".join(f"- {s}" for s in reversed(hist)) or "_No history yet_")
finally:
    cap.release()
    status_ph.info("⚫ Camera stopped.")