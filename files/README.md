# ISL Head Movement Analysis for Text Conversion

Real-time system to detect and classify head movements in **Indian Sign Language (ISL)** — nods, shakes, and tilts — converting them to text using optical flow, 3D pose estimation, and an LSTM sequence classifier.

---

## Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment setup & project scaffold | ✅ |
| 2 | Face detection & landmark extraction | 🔜 |
| 3 | Head pose estimation (Yaw/Pitch/Roll) | 🔜 |
| 4 | Rule-based movement classification | 🔜 |
| 5 | LSTM sequence classifier | 🔜 |
| 6 | ISL grammar mapping & text engine | 🔜 |
| 7 | Streamlit UI & project polish | 🔜 |

---

## Tech Stack

- **Python 3.10+** · **OpenCV 4.8+** · **MediaPipe 0.10+**
- **NumPy / SciPy** — signal processing & linear algebra
- **TensorFlow/Keras** — BiLSTM model (Phase 5)
- **scikit-learn** — baseline models & metrics
- **Streamlit** — real-time dashboard (Phase 7)

---

## Environment Setup

### 1. Choose your platform

| Platform | Recommendation |
|----------|---------------|
| Windows | WSL2 (Ubuntu 22.04) + VS Code Remote |
| macOS Intel | Native Python 3.10 |
| macOS Apple Silicon | Native Python 3.10 + `tensorflow-macos` |
| Linux | Native — smoothest overall |
| Cloud | Google Colab (Phases 1–3 only; no persistent webcam) |

### 2. Clone & create environment

```bash
# Clone
git clone https://github.com/yourname/isl-head-movement.git
cd isl-head-movement

# Create virtual environment
python3.10 -m venv .venv

# Activate
source .venv/bin/activate        # Linux / macOS / WSL2
# .venv\Scripts\activate         # Windows native CMD
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3)?** After the main install:
> ```bash
> pip uninstall tensorflow
> pip install tensorflow-macos tensorflow-metal
> ```

### 4. Verify the environment

```bash
python scripts/check_env.py
```

Expected output — all lines show `[OK]`.

### 5. Run Phase 1 live demo

```bash
python scripts/verify_phase1.py
```

**Controls:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Toggle full 468-point mesh overlay |
| `f` | Toggle optical flow arrows |
| `s` | Save current frame to `data/samples/` |

---

## Project Structure

```
isl_head_movement/
├── configs/
│   └── config.yaml          # All tunable parameters — edit this first
├── src/
│   ├── pipeline/
│   │   ├── face_mesh.py     # MediaPipe FaceMesh wrapper
│   │   └── optical_flow.py  # Lucas-Kanade flow tracker
│   ├── classification/      # Phases 4–5: rule-based + LSTM
│   ├── mapping/             # Phase 6: ISL grammar → text
│   └── utils/
│       ├── camera.py        # VideoCapture wrapper
│       ├── config.py        # YAML config loader (dot-access)
│       ├── drawing.py       # HUD, axes, banners
│       └── logger.py        # Structured logging
├── data/
│   ├── raw/                 # Original recorded clips (DVC tracked)
│   ├── processed/           # Feature arrays (.npy)
│   ├── annotations/         # Gesture labels (.json)
│   └── samples/             # Saved frames for quick inspection
├── models/
│   ├── checkpoints/         # .h5 / .keras training checkpoints
│   ├── exports/             # .tflite, .onnx for deployment
│   └── baselines/           # Scikit-learn baseline models
├── notebooks/               # EDA and experiment notebooks
├── tests/
│   └── test_phase1.py       # pytest unit tests
├── scripts/
│   ├── check_env.py         # Environment verification
│   └── verify_phase1.py     # Phase 1 live demo
├── docs/
├── requirements.txt
├── pytest.ini
├── setup.cfg
└── README.md
```

---

## Configuration

All pipeline parameters live in `configs/config.yaml`. Key settings:

```yaml
camera:
  device_id: 0        # change if using external webcam
  flip_horizontal: true

mediapipe:
  min_detection_confidence: 0.7
  min_tracking_confidence: 0.6

classification:
  nod:
    pitch_delta_threshold: 8.0   # degrees
  shake:
    yaw_delta_threshold: 8.0
```

No code changes needed for tuning — edit the YAML and re-run.

---

## Running Tests

```bash
pytest tests/test_phase1.py -v
```

---

## Next Phase

Once Phase 1 is verified (all [OK] in check_env, live mesh visible):

→ **Phase 2**: Head pose estimation with `solvePnP` + Euler angle decomposition
→ Run: `python scripts/verify_phase1.py` and confirm you see anchor points + flow vectors

---

## ISL Head Movement Semantics

| Movement | ISL Meaning |
|----------|-------------|
| Nod (pitch oscillation) | Affirmation / Yes |
| Head shake (yaw oscillation) | Negation / No |
| Tilt left | Question marker / Doubt |
| Tilt right | Emphasis / Certainty |
| Static | Neutral / No modifier |

---

## License

MIT — see LICENSE for details.
