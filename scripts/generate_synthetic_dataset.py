"""
scripts/generate_synthetic_dataset.py
Generates a realistic synthetic dataset of head movement angle sequences
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR  = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "synthetic_dataset.csv"
WINDOW      = 30

def gen_nod(rng, noise_std):
    amp    = rng.uniform(10, 25)
    cycles = rng.uniform(1.0, 2.5)
    phase  = rng.uniform(0, 2 * np.pi)
    t      = np.linspace(0, 2 * np.pi * cycles, WINDOW)
    pitch  = amp * np.sin(t + phase)
    yaw    = rng.normal(0, 2, WINDOW)
    roll   = rng.normal(0, 1.5, WINDOW)
    data   = np.stack([yaw, pitch, roll], axis=1)
    data  += rng.normal(0, noise_std, data.shape)
    return data

def gen_shake(rng, noise_std):
    amp    = rng.uniform(10, 25)
    cycles = rng.uniform(1.0, 2.5)
    phase  = rng.uniform(0, 2 * np.pi)
    t      = np.linspace(0, 2 * np.pi * cycles, WINDOW)
    yaw    = amp * np.sin(t + phase)
    pitch  = rng.normal(0, 2, WINDOW)
    roll   = rng.normal(0, 1.5, WINDOW)
    data   = np.stack([yaw, pitch, roll], axis=1)
    data  += rng.normal(0, noise_std, data.shape)
    return data

def gen_tilt_left(rng, noise_std):
    target = -rng.uniform(15, 30)
    ramp   = np.linspace(0, target, 8)
    hold   = np.full(WINDOW - 8, target + rng.normal(0, 1.5))
    roll   = np.concatenate([ramp, hold])
    roll  += rng.normal(0, noise_std, WINDOW)
    yaw    = rng.normal(0, 2, WINDOW)
    pitch  = rng.normal(0, 2, WINDOW)
    return np.stack([yaw, pitch, roll], axis=1)

def gen_tilt_right(rng, noise_std):
    target = rng.uniform(15, 30)
    ramp   = np.linspace(0, target, 8)
    hold   = np.full(WINDOW - 8, target + rng.normal(0, 1.5))
    roll   = np.concatenate([ramp, hold])
    roll  += rng.normal(0, noise_std, WINDOW)
    yaw    = rng.normal(0, 2, WINDOW)
    pitch  = rng.normal(0, 2, WINDOW)
    return np.stack([yaw, pitch, roll], axis=1)

def gen_static(rng, noise_std):
    drift_scale = rng.uniform(0.3, 1.2)
    steps       = rng.normal(0, drift_scale, (WINDOW, 3))
    data        = np.cumsum(steps, axis=0)
    data        = np.clip(data, -6, 6)
    data       += rng.normal(0, noise_std * 0.5, data.shape)
    return data

GENERATORS = {
    "NOD":        gen_nod,
    "SHAKE":      gen_shake,
    "TILT_LEFT":  gen_tilt_left,
    "TILT_RIGHT": gen_tilt_right,
    "STATIC":     gen_static,
}

def augment(data, rng):
    bias = rng.normal(0, 3, 3)
    data = data + bias
    scale = rng.uniform(0.8, 1.2)
    data  = data * scale
    if rng.random() < 0.3:
        idx = rng.integers(1, WINDOW - 1)
        data = np.delete(data, idx, axis=0)
        data = np.insert(data, idx, data[idx], axis=0)
    return data

def build_dataset(samples_per_class, noise_std, seed, augment_factor=2):
    rng    = np.random.default_rng(seed)
    rows   = []
    sid    = 0
    total_per_class = samples_per_class * augment_factor

    for label, gen_fn in GENERATORS.items():
        print(f"  Generating {total_per_class} samples for {label} ...")
        for _ in range(samples_per_class):
            base = gen_fn(rng, noise_std)
            variants = [base] + [augment(base.copy(), rng) for _ in range(augment_factor - 1)]
            for data in variants:
                for frame_idx in range(WINDOW):
                    rows.append({
                        "sample_id": sid,
                        "frame":     frame_idx,
                        "yaw":       round(float(data[frame_idx, 0]), 4),
                        "pitch":     round(float(data[frame_idx, 1]), 4),
                        "roll":      round(float(data[frame_idx, 2]), 4),
                        "label":     label,
                    })
                sid += 1

    df = pd.DataFrame(rows)
    return df

def print_stats(df):
    print(f"\n{'':=<55}")
    print(f"  Synthetic Dataset Summary")
    print(f"{'':=<55}")
    samples_df = df.drop_duplicates("sample_id")
    counts     = samples_df["label"].value_counts()
    print(f"  {'Class':<14} {'Samples':>10} {'Frames':>10}")
    print(f"  {'-'*13} {'-'*10} {'-'*10}")
    for label, cnt in counts.items():
        print(f"  {label:<14} {cnt:>10} {cnt * WINDOW:>10}")
    total_s = len(samples_df)
    total_f = len(df)
    print(f"  {'TOTAL':<14} {total_s:>10} {total_f:>10}")
    print(f"{'':=<55}\n")

def plot_samples(df, save_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed")
        return

    labels = list(GENERATORS.keys())
    fig, axes = plt.subplots(len(labels), 3, figsize=(14, 3 * len(labels)))

    for row_i, label in enumerate(labels):
        sub = df[df["label"] == label]
        sid = sub["sample_id"].iloc[0]
        sample = sub[sub["sample_id"] == sid]

        for col_i, (angle, color) in enumerate([("yaw", "#1565C0"), ("pitch", "#2E7D32"), ("roll", "#B71C1C")]):
            ax = axes[row_i, col_i]
            ax.plot(sample["frame"].values, sample[angle].values, color=color, linewidth=2)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_title(f"{label} — {angle.capitalize()}", fontsize=9)
            ax.set_xlim(0, WINDOW - 1)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = save_dir / "synthetic_dataset_preview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [INFO] Plot saved → {out}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--augment-factor", type=int, default=2)
    parser.add_argument("--noise", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Generating synthetic dataset")
    print(f"       samples_per_class = {args.samples_per_class}")
    print(f"       augment_factor    = {args.augment_factor}")
    print(f"       noise_std         = {args.noise}°\n")

    df = build_dataset(args.samples_per_class, args.noise, args.seed, args.augment_factor)
    print_stats(df)

    df.to_csv(out_path, index=False)
    print(f"[INFO] Dataset saved → {out_path}")
    print(f"       {len(df):,} rows | {df['sample_id'].nunique():,} samples\n")

    if args.plot:
        plot_samples(df, out_path.parent)

if __name__ == "__main__":
    main()
