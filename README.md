# ISL Head Movement Analysis — Text Conversion System

> Real-time Indian Sign Language head movement detection and text conversion using MediaPipe, BiLSTM, and ISL grammar rules.

---

## Demo

```
Nod twice        →  "DEFINITELY YES"
Tilt left + Nod  →  "REALLY?"
Shake twice      →  "ABSOLUTELY NOT"
Tilt left twice  →  "WHY?"
```

Run the Streamlit dashboard:
```bash
streamlit run app/streamlit_app.py
```

---

## Architecture

```
Webcam
  │
  ▼
MediaPipe FaceMesh (468 landmarks)          ~12 ms
  │
  ▼
solvePnP → Euler angles (Yaw / Pitch / Roll)  ~3 ms
  │
  ▼
30-frame sliding window (6+2 features)
  │
  ▼
BiLSTM classifier  (2-layer, ~180K params)    ~8 ms
  │
  ▼
ISL Grammar Engine  (sequence + modifier)     ~0.2 ms
  │
  ▼
Text output                          TOTAL  ~23 ms
```

**5 gesture classes:** NOD · SHAKE · TILT_LEFT · TILT_RIGHT · STATIC

---

## Project Structure

```
isl-head-movement/
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
├── configs/
│   ├── config.yaml               # global config
│   └── isl_grammar.json          # ISL grammar rules
├── data/
│   └── raw/                      # CSV datasets
├── models/
│   ├── bilstm_best.keras         # trained model
│   ├── bilstm.tflite             # TFLite export
│   └── normaliser_stats.npz      # z-score stats
├── reports/
│   ├── eval_final.json           # full evaluation report
│   └── confusion_final.png       # confusion matrix
├── scripts/
│   ├── generate_synthetic_dataset.py
│   ├── adapt_real_dataset.py
│   ├── train_bilstm.py
│   ├── evaluate_full_pipeline.py
│   ├── verify_phase1.py  …  verify_phase6.py
│   └── diagnose_tilt.py
├── src/
│   ├── classification/
│   │   ├── gesture_state.py      # FSM
│   │   └── rule_classifier.py    # threshold engine
│   ├── mapping/
│   │   └── isl_text_engine.py    # gesture → text
│   ├── ml/
│   │   ├── bilstm_model.py       # model definition
│   │   └── feature_engineering.py
│   ├── pipeline/
│   │   ├── face_mesh.py
│   │   ├── optical_flow.py
│   │   └── pose_estimator.py
│   └── utils/
│       ├── camera.py
│       ├── config.py
│       ├── drawing.py
│       └── logger.py
├── tests/
│   ├── test_phase1.py
│   ├── test_phase3.py
│   └── test_phase6.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone and enter
git clone https://github.com/your-username/isl-head-movement.git
cd isl-head-movement

# 2. Create environment (Python 3.10+)
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python scripts/check_env.py
```

---

## Quick Start

### Option A — Streamlit dashboard (recommended)
```bash
streamlit run app/streamlit_app.py
```
Press **▶ Start** in the sidebar. Move your head to generate ISL text.

### Option B — OpenCV demo (lightweight)
```bash
python scripts/verify_phase6.py
```

### Option C — TFLite (faster inference)
```bash
python scripts/verify_phase6.py --tflite
```

---

## Training From Scratch

```bash
# 1. Generate dataset
python scripts/generate_synthetic_dataset.py \
    --samples-per-class 400 --augment-factor 2 --noise 1.2 --plot

# 2. Train BiLSTM
python scripts/train_bilstm.py \
    --data data/raw/synthetic_dataset.csv \
    --epochs 80 --batch 32 --lr 0.001 --plot

# 3. Full evaluation
python scripts/evaluate_full_pipeline.py \
    --data data/raw/synthetic_dataset.csv --plot
```

---

## Results

| Model | Macro F1 | Accuracy | Latency (p50) |
|---|---|---|---|
| Rule-based | ~0.78 | ~80% | <1 ms |
| Random Forest | ~0.94 | ~94% | ~2 ms |
| **BiLSTM (ours)** | **>0.999** | **>99%** | **~8 ms** |

**End-to-end latency:** ~23 ms (well under 40 ms real-time budget)

Per-class F1 (BiLSTM):

| Class | F1 |
|---|---|
| NOD | >0.999 |
| SHAKE | >0.999 |
| TILT_LEFT | >0.999 |
| TILT_RIGHT | >0.999 |
| STATIC | >0.999 |

---

## ISL Grammar Rules

| Gesture | Meaning |
|---|---|
| NOD | YES |
| SHAKE | NO |
| TILT_LEFT | ? (question marker) |
| TILT_RIGHT | ! (emphasis marker) |
| NOD + NOD | DEFINITELY YES |
| SHAKE + SHAKE | ABSOLUTELY NOT |
| TILT_LEFT + NOD | REALLY? |
| TILT_LEFT + SHAKE | IS THAT NO? |
| TILT_LEFT + TILT_LEFT | WHY? |
| TILT_RIGHT + NOD | YES! |
| TILT_RIGHT + SHAKE | NO! |

Full grammar: `configs/isl_grammar.json`

---

## Running Tests

```bash
pytest tests/ -v
```

Expected: **all tests pass** across phase1, phase3, phase6 suites.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Face detection | MediaPipe FaceMesh (468 landmarks) |
| Pose estimation | OpenCV solvePnP + Rodrigues decomposition |
| Motion tracking | Lucas-Kanade optical flow |
| ML classifier | TensorFlow/Keras BiLSTM |
| Deployment | TFLite (quantised) |
| Dashboard | Streamlit |
| Testing | pytest |

---

## Extending the System

**Add a new gesture class:**
1. Add generator in `generate_synthetic_dataset.py`
2. Add mapping in `configs/isl_grammar.json`
3. Increment `n_classes` in `bilstm_model.py`
4. Retrain

**Use real webcam data:**
```bash
python scripts/record_angles.py --session s001
python scripts/adapt_real_dataset.py --source hopenet --path data/external/
```

**Deploy on mobile:**
```bash
# TFLite model is already exported at models/bilstm.tflite
# Use TensorFlow Lite Android/iOS runtime
```

---

## License

MIT License — see `LICENSE` for details.
