"""
scripts/evaluate_rule_classifier.py
Evaluates the Phase 3 rule-based classifier against labeled CSV data
produced by record_angles.py.

Usage:
    python scripts/evaluate_rule_classifier.py --data data/raw/angles_session_s001.csv
    python scripts/evaluate_rule_classifier.py --data data/raw/ --window 30 --step 15
    python scripts/evaluate_rule_classifier.py --data data/raw/ --plot --save-report

Outputs:
    • Console: per-class precision / recall / F1, overall accuracy
    • PNG:      confusion matrix heat-map  (reports/confusion_matrix_phase3.png)
    • JSON:     full metrics report        (reports/eval_phase3.json)
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification.rule_classifier import RuleClassifier

# ── optional matplotlib (skip plot if unavailable) ──────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

CLASSES = ["NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"]
REPORTS_DIR = Path("reports")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_files(path: str) -> pd.DataFrame:
    """Accept a single CSV file or a directory of CSVs."""
    p = Path(path)
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.csv")))
        if not files:
            sys.exit(f"[ERROR] No CSV files found in {path}")
        frames = [pd.read_csv(f) for f in files]
        df = pd.concat(frames, ignore_index=True)
        print(f"[INFO] Loaded {len(files)} file(s) → {len(df)} rows")
    else:
        df = pd.read_csv(p)
        print(f"[INFO] Loaded {p.name} → {len(df)} rows")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required = {"yaw", "pitch", "roll", "label"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] CSV missing columns: {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# Window slicing
# ─────────────────────────────────────────────────────────────────────────────

def slice_windows(df: pd.DataFrame, window: int, step: int):
    """
    Slide a window over the DataFrame.
    Ground-truth label = majority label inside the window.
    Returns list of (angles_list, true_label).
    """
    samples = []
    rows = df.to_dict("records")
    for start in range(0, len(rows) - window + 1, step):
        chunk = rows[start: start + window]
        angles = [{"yaw": r["yaw"], "pitch": r["pitch"], "roll": r["roll"]}
                  for r in chunk]
        labels = [r["label"] for r in chunk]
        # majority vote for ground truth
        from collections import Counter
        true_label = Counter(labels).most_common(1)[0][0]
        samples.append((angles, true_label))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, classes):
    """
    Returns dict with per-class precision/recall/F1 and macro/weighted averages.
    Pure numpy — no sklearn dependency.
    """
    label2idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        ti = label2idx.get(t, -1)
        pi = label2idx.get(p, -1)
        if ti >= 0 and pi >= 0:
            cm[ti, pi] += 1

    per_class = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int(cm[i, :].sum())
        per_class[cls] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   support,
        }

    total = len(y_true)
    correct = int(np.diag(cm).sum())
    accuracy = correct / total if total > 0 else 0.0

    supports = np.array([per_class[c]["support"] for c in classes])
    macro_f1    = float(np.mean([per_class[c]["f1"] for c in classes]))
    weighted_f1 = float(np.average([per_class[c]["f1"] for c in classes],
                                    weights=supports)) if supports.sum() > 0 else 0.0

    return {
        "accuracy":    round(accuracy, 4),
        "macro_f1":    round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class":   per_class,
        "confusion_matrix": cm.tolist(),
        "classes":     classes,
        "total_samples": total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Console report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(metrics: dict) -> None:
    cls_list = metrics["classes"]
    pc = metrics["per_class"]

    header = f"\n{'':=<62}"
    print(header)
    print(f"  Phase 3 Rule Classifier — Evaluation Report")
    print(f"  Total windows: {metrics['total_samples']}   "
          f"Accuracy: {metrics['accuracy']*100:.1f}%   "
          f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"{'':=<62}")
    print(f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*13} {'-'*10} {'-'*10} {'-'*8} {'-'*9}")
    for cls in cls_list:
        m = pc[cls]
        print(f"  {cls:<14} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>8.3f} {m['support']:>9}")
    print(f"{'':=<62}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.3f}")

    # Print confusion matrix to console
    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    cm = np.array(metrics["confusion_matrix"])
    col_w = 10
    print("  " + "".join(f"{c:>{col_w}}" for c in cls_list))
    for i, cls in enumerate(cls_list):
        row = "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(cls_list)))
        print(f"  {cls:<14}{row}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics: dict, save_path: Path) -> None:
    if not HAS_MPL:
        print("[WARN] matplotlib not found — skipping plot")
        return

    cm   = np.array(metrics["confusion_matrix"])
    cls  = metrics["classes"]
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in [
        (axes[0], cm,   "Counts",       "d"),
        (axes[1], norm, "Normalised",   ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls, rotation=30, ha="right")
        ax.set_yticks(range(len(cls))); ax.set_yticklabels(cls)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {title}")
        plt.colorbar(im, ax=ax)
        thresh = data.max() / 2
        for r in range(len(cls)):
            for c in range(len(cls)):
                val = f"{data[r,c]:{fmt}}"
                color = "white" if data[r, c] > thresh else "black"
                ax.text(c, r, val, ha="center", va="center",
                        fontsize=9, color=color)

    # Per-class bar below
    f1s = [metrics["per_class"][c]["f1"] for c in cls]
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    bars = ax2.bar(cls, f1s, color=["#2196F3" if f >= 0.88 else "#FF5722" for f in f1s])
    ax2.axhline(0.88, color="green", linestyle="--", label="Target F1 = 0.88")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Per-Class F1 — Phase 3 Rule Classifier")
    ax2.legend()
    for bar, f1 in zip(bars, f1s):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02, f"{f1:.2f}",
                 ha="center", fontsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    fig2.tight_layout()
    fig2.savefig(save_path.parent / "f1_bars_phase3.png", dpi=150)
    print(f"[INFO] Plots saved → {save_path.parent}/")
    plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 rule classifier")
    parser.add_argument("--data",        required=True,
                        help="Path to labeled CSV file or directory of CSVs")
    parser.add_argument("--window",      type=int, default=30,
                        help="Sliding window size in frames (default 30)")
    parser.add_argument("--step",        type=int, default=15,
                        help="Stride between windows (default 15 = 50%% overlap)")
    parser.add_argument("--plot",        action="store_true",
                        help="Generate and save confusion matrix PNG")
    parser.add_argument("--save-report", action="store_true",
                        help="Save JSON metrics to reports/eval_phase3.json")
    args = parser.parse_args()

    # 1. Load data
    df = load_csv_files(args.data)
    validate_columns(df)

    # 2. Filter to known classes only
    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)
    print(f"[INFO] Class distribution:\n{df['label'].value_counts().to_string()}\n")

    if len(df) < args.window:
        sys.exit(f"[ERROR] Not enough data: {len(df)} rows < window {args.window}")

    # 3. Slice into windows
    samples = slice_windows(df, args.window, args.step)
    print(f"[INFO] Generated {len(samples)} evaluation windows "
          f"(window={args.window}, step={args.step})")

    # 4. Run classifier
    clf = RuleClassifier()
    y_true, y_pred, confidences = [], [], []

    for angles, true_label in samples:
        result = clf.classify(angles)
        y_true.append(true_label)
        y_pred.append(result["label"])
        confidences.append(result["confidence"])

    # 5. Compute metrics
    metrics = compute_metrics(y_true, y_pred, CLASSES)
    metrics["mean_confidence"] = round(float(np.mean(confidences)), 4)

    # 6. Print report
    print_report(metrics)
    print(f"  Mean classifier confidence: {metrics['mean_confidence']:.3f}\n")

    # 7. Check target thresholds
    print("  Target checks (Phase 3 rule-based baseline):")
    all_pass = True
    for cls in CLASSES:
        f1 = metrics["per_class"][cls]["f1"]
        sup = metrics["per_class"][cls]["support"]
        target = 0.75  # rule-based target (LSTM target is 0.88 in Phase 5)
        status = "✅" if f1 >= target or sup == 0 else "⚠️ "
        if f1 < target and sup > 0:
            all_pass = False
        print(f"    {status} {cls:<14} F1={f1:.3f}  (target ≥ {target})")

    if all_pass:
        print("\n  ✅ All classes meet rule-based baseline — ready for Phase 4!\n")
    else:
        print("\n  ⚠️  Some classes below target — review thresholds in "
              "rule_classifier.py or collect more data.\n")

    # 8. Optional: save JSON report
    if args.save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / "eval_phase3.json"
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Report saved → {out}")

    # 9. Optional: plot
    if args.plot:
        plot_confusion_matrix(metrics, REPORTS_DIR / "confusion_matrix_phase3.png")


if __name__ == "__main__":
    main()