"""
scripts/adapt_real_dataset.py
Adapts publicly available head pose datasets into the ISL project's
CSV format: sample_id, frame, yaw, pitch, roll, label

Supported datasets
──────────────────
1. BIWI Kinect Head Pose  (ETH Zürich)
   Download: https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html
   Format:   .txt files per frame — rotation matrix (3×3) + head centre (x,y,z)

2. 300-W LP / AFLW2000-3D  (Carnegie Mellon)
   Download: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
   Format:   .mat files per image — yaw/pitch/roll stored as pose_para

3. CMU Panoptic (subset)
   Download: http://domedb.perception.cs.cmu.edu/
   Format:   JSON per frame — body25 + face keypoints, head orientation quaternion

4. SynHead / HopeNet public angles CSV
   Download: https://github.com/natanielruiz/deep-head-pose (test set CSV)
   Format:   CSV with columns: filename, yaw, pitch, roll  (single-frame, degrees)

Usage examples
──────────────
# BIWI:
python scripts/adapt_real_dataset.py --source biwi --path data/external/biwi/ --plot

# 300-W LP / AFLW2000:
python scripts/adapt_real_dataset.py --source aflw --path data/external/AFLW2000/ --plot

# HopeNet CSV:
python scripts/adapt_real_dataset.py --source hopenet --path data/external/hopenet_test.csv

# CMU Panoptic:
python scripts/adapt_real_dataset.py --source panoptic --path data/external/panoptic/

Output
──────
data/raw/real_<source>_dataset.csv   — same schema as synthetic_dataset.csv
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification.rule_classifier import RuleClassifier
from src.pipeline.angle_buffer import AngleBuffer

OUTPUT_DIR = Path("data/raw")
WINDOW     = 30    # frames per sample
STEP       = 15    # sliding window step

CLASSES    = ["NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"]

# ─────────────────────────────────────────────────────────────────────────────
# Angle-based auto-labeller  (uses your existing rule classifier)
# ─────────────────────────────────────────────────────────────────────────────

def autolabel_windows(angle_sequence: list[dict],
                      window: int = WINDOW,
                      step:   int = STEP,
                      min_conf: float = 0.55) -> pd.DataFrame:
    """
    Slide a window over a continuous angle sequence,
    run RuleClassifier on each window, return labelled DataFrame.
    """
    clf  = RuleClassifier()
    rows = []
    sid  = 0

    for start in range(0, len(angle_sequence) - window + 1, step):
        chunk = angle_sequence[start: start + window]

        buf = AngleBuffer(maxlen=window)
        for a in chunk:
            buf.push(a["yaw"], a["pitch"], a["roll"])

        result = clf.classify(buf)
        if result is None:
            continue

        label, conf = result

        # Skip low-confidence ambiguous windows
        if conf < min_conf and label != "STATIC":
            continue

        for frame_idx, a in enumerate(chunk):
            rows.append({
                "sample_id":  sid,
                "frame":      frame_idx,
                "yaw":        round(a["yaw"],   4),
                "pitch":      round(a["pitch"], 4),
                "roll":       round(a["roll"],  4),
                "label":      label,
                "confidence": round(conf, 4),
                "source":     a.get("source", "real"),
            })
        sid += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# BIWI adapter
# ─────────────────────────────────────────────────────────────────────────────

def _rot_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Convert 3×3 rotation matrix → (yaw, pitch, roll) in degrees."""
    pitch = float(np.degrees(np.arcsin(-R[2, 0])))
    roll  = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    yaw   = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return yaw, pitch, roll


def load_biwi(path: str) -> list[dict]:
    """
    BIWI stores one .txt per frame.  Each .txt = 6 lines:
      lines 0-2: rotation matrix rows (space-separated)
      line  3:   head centre
    Returns flat list of {yaw, pitch, roll, source} dicts.
    """
    txt_files = sorted(glob.glob(os.path.join(path, "**", "*.txt"),
                                  recursive=True))
    if not txt_files:
        sys.exit(f"[ERROR] No .txt files found in {path}")

    angles = []
    for fp in txt_files:
        try:
            lines = open(fp).read().strip().split("\n")
            if len(lines) < 3:
                continue
            R = np.array([list(map(float, lines[i].split()))
                          for i in range(3)])
            if R.shape != (3, 3):
                continue
            yaw, pitch, roll = _rot_matrix_to_euler(R)
            angles.append({"yaw": yaw, "pitch": pitch,
                            "roll": roll, "source": "biwi"})
        except Exception:
            continue

    print(f"[BIWI] Loaded {len(angles)} frames from {len(txt_files)} files")
    return angles


# ─────────────────────────────────────────────────────────────────────────────
# AFLW2000-3D / 300-W LP adapter
# ─────────────────────────────────────────────────────────────────────────────

def load_aflw(path: str) -> list[dict]:
    """
    AFLW2000 .mat files each contain a pose_para field:
        pose_para[0:3] = pitch, yaw, roll (radians)
    Requires scipy.io.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        sys.exit("[ERROR] scipy required for AFLW: pip install scipy")

    mat_files = sorted(glob.glob(os.path.join(path, "**", "*.mat"),
                                  recursive=True))
    if not mat_files:
        sys.exit(f"[ERROR] No .mat files in {path}")

    angles = []
    for fp in mat_files:
        try:
            mat = loadmat(fp)
            if "Pose_Para" not in mat:
                continue
            pp = mat["Pose_Para"].flatten()
            # AFLW convention: pitch, yaw, roll in radians
            pitch = float(np.degrees(pp[0]))
            yaw   = float(np.degrees(pp[1]))
            roll  = float(np.degrees(pp[2]))
            angles.append({"yaw": yaw, "pitch": pitch,
                            "roll": roll, "source": "aflw"})
        except Exception:
            continue

    print(f"[AFLW] Loaded {len(angles)} frames from {len(mat_files)} .mat files")
    return angles


# ─────────────────────────────────────────────────────────────────────────────
# HopeNet public test CSV adapter
# ─────────────────────────────────────────────────────────────────────────────

def load_hopenet(path: str) -> list[dict]:
    """
    HopeNet test CSV columns: filename, yaw, pitch, roll (degrees, per-image).
    Since there's no temporal ordering, we sort by filename (which is often
    sequential within a video) and treat as a continuous sequence.
    """
    df = pd.read_csv(path)
    required = {"yaw", "pitch", "roll"}

    # Handle slight column name variants
    df.columns = [c.strip().lower() for c in df.columns]

    if not required.issubset(df.columns):
        sys.exit(f"[ERROR] HopeNet CSV must have columns: {required}. "
                 f"Found: {list(df.columns)}")

    # Sort by filename if present (preserves video sequence)
    if "filename" in df.columns:
        df = df.sort_values("filename").reset_index(drop=True)

    angles = [
        {"yaw":   round(float(r.yaw),   3),
         "pitch": round(float(r.pitch), 3),
         "roll":  round(float(r.roll),  3),
         "source": "hopenet"}
        for r in df.itertuples()
    ]
    print(f"[HopeNet] Loaded {len(angles)} frames from {path}")
    return angles


# ─────────────────────────────────────────────────────────────────────────────
# CMU Panoptic adapter
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_euler(q: list) -> tuple[float, float, float]:
    """Quaternion [w, x, y, z] → (yaw, pitch, roll) degrees."""
    w, x, y, z = q
    # Yaw (Z), Pitch (Y), Roll (X)
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    return float(yaw), float(pitch), float(roll)


def load_panoptic(path: str) -> list[dict]:
    """
    CMU Panoptic: each frame is a JSON file with body keypoints.
    We look for 'hdFace3d' or 'bodies' → orientation quaternion.
    """
    json_files = sorted(glob.glob(os.path.join(path, "**", "*.json"),
                                   recursive=True))
    if not json_files:
        sys.exit(f"[ERROR] No JSON files in {path}")

    angles = []
    for fp in json_files:
        try:
            data = json.load(open(fp))
            # Try hdFace3d first
            if "hdFace3d" in data:
                for face in data["hdFace3d"]:
                    q = face.get("orientation", None)
                    if q and len(q) == 4:
                        yaw, pitch, roll = _quat_to_euler(q)
                        angles.append({"yaw": yaw, "pitch": pitch,
                                       "roll": roll, "source": "panoptic"})
            # Fallback: body orientation
            elif "bodies" in data:
                for body in data["bodies"]:
                    q = body.get("orientation", None)
                    if q and len(q) == 4:
                        yaw, pitch, roll = _quat_to_euler(q)
                        angles.append({"yaw": yaw, "pitch": pitch,
                                       "roll": roll, "source": "panoptic"})
        except Exception:
            continue

    print(f"[Panoptic] Loaded {len(angles)} frames from {len(json_files)} files")
    return angles


# ─────────────────────────────────────────────────────────────────────────────
# Stats + plot
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(df: pd.DataFrame, source: str) -> None:
    print(f"\n{'':=<55}")
    print(f"  Real Dataset Adapter — {source.upper()} Summary")
    print(f"{'':=<55}")
    samples = df.drop_duplicates("sample_id")
    counts  = samples["label"].value_counts()
    print(f"  {'Class':<14} {'Samples':>10} {'Frames':>10}")
    print(f"  {'-'*13} {'-'*10} {'-'*10}")
    for label, cnt in counts.items():
        print(f"  {label:<14} {cnt:>10} {cnt * WINDOW:>10}")
    print(f"  {'TOTAL':<14} {len(samples):>10} {len(df):>10}")
    print(f"{'':=<55}\n")


def plot_preview(df: pd.DataFrame, out_dir: Path, source: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels   = df["label"].unique()
    fig, axes = plt.subplots(len(labels), 3, figsize=(14, 3 * len(labels)))
    if len(labels) == 1:
        axes = axes[np.newaxis, :]

    for row_i, label in enumerate(sorted(labels)):
        sub = df[df["label"] == label]
        sid = sub["sample_id"].iloc[0]
        sample = sub[sub["sample_id"] == sid]
        for col_i, (angle, color) in enumerate(
                [("yaw", "#1565C0"), ("pitch", "#2E7D32"), ("roll", "#B71C1C")]):
            ax = axes[row_i, col_i]
            ax.plot(sample["frame"].values, sample[angle].values,
                    color=color, linewidth=2)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_title(f"{label} — {angle}", fontsize=9)
            ax.set_ylabel("degrees"); ax.set_xlabel("frame")
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Real Dataset ({source}) — One Sample Per Class", fontsize=12)
    fig.tight_layout()
    out = out_dir / f"real_{source}_preview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[INFO] Preview saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Balancer — downsample majority classes to avoid skew
# ─────────────────────────────────────────────────────────────────────────────

def balance_dataset(df: pd.DataFrame, max_per_class: int = 600) -> pd.DataFrame:
    samples = df.drop_duplicates("sample_id")[["sample_id", "label"]]
    balanced_ids = []
    for label in df["label"].unique():
        ids = samples[samples["label"] == label]["sample_id"].values
        chosen = ids[:max_per_class]              # already shuffled by autolabel
        balanced_ids.extend(chosen)
    return df[df["sample_id"].isin(balanced_ids)].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

LOADERS = {
    "biwi":     load_biwi,
    "aflw":     load_aflw,
    "hopenet":  load_hopenet,
    "panoptic": load_panoptic,
}


def main():
    parser = argparse.ArgumentParser(
        description="Adapt a public head-pose dataset → ISL project CSV")
    parser.add_argument("--source", required=True,
                        choices=list(LOADERS.keys()),
                        help="Dataset source to adapt")
    parser.add_argument("--path",   required=True,
                        help="Path to raw dataset directory or CSV file")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help=f"Window size in frames (default {WINDOW})")
    parser.add_argument("--step",   type=int, default=STEP,
                        help=f"Sliding window step (default {STEP})")
    parser.add_argument("--max-per-class", type=int, default=600,
                        help="Cap samples per class after balancing (default 600)")
    parser.add_argument("--min-conf", type=float, default=0.55,
                        help="Minimum classifier confidence to keep a window (default 0.55)")
    parser.add_argument("--plot",   action="store_true",
                        help="Save a preview plot")
    parser.add_argument("--output", type=str, default="",
                        help="Override output CSV path")
    args = parser.parse_args()

    # 1. Load raw angles
    loader = LOADERS[args.source]
    angles = loader(args.path)

    if len(angles) < args.window:
        sys.exit(f"[ERROR] Only {len(angles)} frames loaded — need at least {args.window}")

    # 2. Auto-label with rule classifier
    print(f"[INFO] Auto-labelling {len(angles)} frames with sliding window "
          f"(w={args.window}, step={args.step}, min_conf={args.min_conf}) ...")
    df = autolabel_windows(
        angles,
        window=args.window,
        step=args.step,
        min_conf=args.min_conf,
    )

    if df.empty:
        sys.exit("[ERROR] No windows passed the confidence threshold. "
                 "Try lowering --min-conf or check your dataset path.")

    # 3. Balance
    df = balance_dataset(df, max_per_class=args.max_per_class)

    # 4. Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else \
               OUTPUT_DIR / f"real_{args.source}_dataset.csv"
    df.to_csv(out_path, index=False)

    # 5. Report
    print_stats(df, args.source)
    print(f"[INFO] Saved → {out_path}")
    print(f"       {len(df):,} rows | "
          f"{df['sample_id'].nunique():,} samples | "
          f"{df['label'].nunique()} classes\n")

    if args.plot:
        plot_preview(df, OUTPUT_DIR, args.source)

    print("[NEXT] Merge with synthetic data if needed:")
    print("       python scripts/merge_datasets.py\n")
    print("[NEXT] Or evaluate directly:")
    print(f"       python scripts/evaluate_rule_classifier.py "
          f"--data {out_path} --plot --save-report\n")


if __name__ == "__main__":
    main()