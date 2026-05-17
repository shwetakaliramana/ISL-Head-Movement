"""
scripts/verify_phase4.py
Real-time inference demo using the trained BiLSTM model.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.face_mesh import FaceMeshDetector
from src.pipeline.pose_estimator import HeadPoseEstimator
from src.utils.camera import Camera
from src.utils.logger import get_logger
from src.ml.feature_engineering import WINDOW, ROLL_CLIP

log = get_logger(__name__)

N_FEAT = 8  # yaw, pitch, roll, gradients, roll sign, raw roll
CLASSES = ["NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"]
MODEL_PATH = Path("models/bilstm_best.keras")
TFLITE_PATH = Path("models/bilstm_best.tflite")
STATS_PATH = Path("models/normaliser_stats.npz")


def load_normalizer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["mean"].astype(np.float32), data["std"].astype(np.float32)


def build_model_input(buffer: deque[dict[str, float]], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raw = np.array([[d["yaw"], d["pitch"], d["roll"]] for d in buffer], dtype=np.float32)

    dyaw = np.gradient(raw[:, 0]).astype(np.float32)
    dpitch = np.gradient(raw[:, 1]).astype(np.float32)
    droll = np.gradient(raw[:, 2]).astype(np.float32)

    roll_sign = np.sign(raw[:, 2]).astype(np.float32)
    pad = np.pad(roll_sign, (2, 2), mode="edge")
    rs_smooth = np.array([pad[i:i + 5].mean() for i in range(WINDOW)], dtype=np.float32)

    roll_raw = np.clip(raw[:, 2], -ROLL_CLIP, ROLL_CLIP).astype(np.float32)

    feat6 = np.stack([raw[:, 0], raw[:, 1], raw[:, 2], dyaw, dpitch, droll], axis=1)
    feat6 = (feat6 - mean) / std

    features = np.concatenate(
        [feat6, rs_smooth[:, np.newaxis], roll_raw[:, np.newaxis]],
        axis=1,
    )
    return features[np.newaxis, ...].astype(np.float32)


def load_keras_model(path: Path):
    import tensorflow as tf

    model = tf.keras.models.load_model(path)
    model.predict(np.zeros((1, WINDOW, N_FEAT), dtype=np.float32), verbose=0)
    return model


def convert_to_tflite(keras_path: Path, tflite_path: Path) -> Path:
    import tensorflow as tf

    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    return tflite_path


def load_tflite_interpreter(path: Path):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def infer_keras(model, x: np.ndarray) -> np.ndarray:
    return model.predict(x, verbose=0)[0]


def infer_tflite(interpreter, x: np.ndarray) -> np.ndarray:
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    return interpreter.get_tensor(output_index)[0]


def draw_hud(frame: np.ndarray, label: str, confidence: float, latency_ms: float) -> None:
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, h - 110), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"{label} {confidence:.2f}", (20, h - 65), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{latency_ms:.1f} ms", (20, h - 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (180, 220, 255), 2, cv2.LINE_AA)


def run_demo(use_tflite: bool, threshold: float, max_frames: int) -> int:
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Missing normalizer stats: {STATS_PATH}")
    if not use_tflite and not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing Keras model: {MODEL_PATH}")
    if use_tflite and not TFLITE_PATH.exists():
        raise FileNotFoundError(f"Missing TFLite model: {TFLITE_PATH}")

    mean, std = load_normalizer(STATS_PATH)
    model = None
    interpreter = None

    if use_tflite:
        tflite_path = TFLITE_PATH if TFLITE_PATH.exists() else convert_to_tflite(MODEL_PATH, TFLITE_PATH)
        log.info("Using TFLite model: %s", tflite_path)
        try:
            interpreter = load_tflite_interpreter(tflite_path)
        except RuntimeError as exc:
            log.warning("TFLite interpreter unavailable (%s); falling back to Keras.", exc)
            use_tflite = False
            interpreter = None
    else:
        log.info("Using Keras model: %s", MODEL_PATH)
        model = load_keras_model(MODEL_PATH)

    if model is None and not use_tflite:
        model = load_keras_model(MODEL_PATH)

    buffer: deque[dict[str, float]] = deque(maxlen=WINDOW)

    with Camera() as cam, FaceMeshDetector() as detector:
        pose = HeadPoseEstimator(cam.frame_size[0], cam.frame_size[1])
        log.info("Phase 4 demo started (max_frames=%d, tflite=%s)", max_frames, use_tflite)

        frame_count = 0
        empty_reads = 0
        last_label = "STATIC"
        last_conf = 0.0
        display_enabled = True

        while frame_count < max_frames:
            frame = cam.read()
            if frame is None:
                empty_reads += 1
                if empty_reads <= 30:
                    continue
                log.warning("Camera returned no frame repeatedly; stopping.")
                break

            empty_reads = 0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = detector.process(rgb)

            if face is not None:
                pose_result = pose.estimate(face.anchor_px)
                if pose_result is not None:
                    buffer.append({"yaw": pose_result.yaw, "pitch": pose_result.pitch, "roll": pose_result.roll})
                    cv2.putText(frame, f"Y:{pose_result.yaw:+.1f} P:{pose_result.pitch:+.1f} R:{pose_result.roll:+.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2, cv2.LINE_AA)

            if len(buffer) == WINDOW:
                x = build_model_input(buffer, mean, std)
                t0 = time.perf_counter()
                probs = infer_tflite(interpreter, x) if use_tflite else infer_keras(model, x)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                idx = int(np.argmax(probs))
                label = CLASSES[idx]
                confidence = float(probs[idx])
                last_label, last_conf = label, confidence
                if confidence >= threshold:
                    log.info("Prediction: %s (%.3f) | %.1f ms", label, confidence, latency_ms)
                draw_hud(frame, label, confidence, latency_ms)
            else:
                draw_hud(frame, last_label, last_conf, 0.0)

            if display_enabled:
                try:
                    cv2.imshow("Phase 4 Demo", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except cv2.error:
                    display_enabled = False

            frame_count += 1

        if display_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4 live inference demo")
    parser.add_argument("--tflite", action="store_true", help="Use the TFLite model")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold for logging")
    parser.add_argument("--max-frames", type=int, default=120, help="Stop after this many frames")
    args = parser.parse_args()

    try:
        return run_demo(args.tflite, args.threshold, args.max_frames)
    except Exception as exc:
        log.exception("Phase 4 demo failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())