"""
scripts/diagnose_tilt.py
Definitive diagnostic — answers exactly WHY TILT_LEFT has F1=0.

Runs 4 checks and prints a verdict for each:
  1. Raw data check    — are TILT_LEFT roll values actually negative in the CSV?
  2. Feature check     — after feature engineering, is feature[2] (roll) negative?
  3. Separability test — can a 1-rule classifier (roll_mean < 0) hit F1 > 0.95?
  4. RF probe          — does a depth-3 tree with raw features separate the classes?

If check 3 passes → the data is fine, the problem is in training/normalisation code.
If check 3 fails → the CSV itself is corrupted and must be regenerated.

Usage:
    python scripts/diagnose_tilt.py --data data/raw/synthetic_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASSES    = ["NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"]
TILT_L_IDX = CLASSES.index("TILT_LEFT")
TILT_R_IDX = CLASSES.index("TILT_RIGHT")
WINDOW     = 30


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: Raw CSV
# ─────────────────────────────────────────────────────────────────────────────

def check_raw_csv(df: pd.DataFrame) -> bool:
    print("\n" + "="*60)
    print("CHECK 1 -- Raw CSV roll values")
    print("="*60)

    for label in CLASSES:
        sub  = df[df["label"] == label]["roll"]
        if len(sub) == 0:
            print(f"  {label:<14} ⚠️  NO ROWS FOUND")
            continue
        print(f"  {label:<14}  n={len(sub):5d}  "
              f"mean={sub.mean():+7.2f}°  "
              f"min={sub.min():+7.2f}°  "
              f"max={sub.max():+7.2f}°  "
              f"pct_negative={100*(sub<0).mean():.1f}%")

    tl = df[df["label"] == "TILT_LEFT"]["roll"]
    tr = df[df["label"] == "TILT_RIGHT"]["roll"]

    ok = True
    if len(tl) == 0:
        print("\n  ❌ TILT_LEFT has zero rows in CSV!")
        ok = False
    elif tl.mean() >= 0:
        print(f"\n  [FAIL] TILT_LEFT roll mean is {tl.mean():+.2f} deg -- should be negative!")
        ok = False
    elif tl.max() > -5:
        print(f"\n  [FAIL] TILT_LEFT roll max is {tl.max():+.2f} deg -- too close to zero, augmentation is flipping the sign!")
        ok = False
    else:
        print(f"\n  [PASS] TILT_LEFT roll is firmly negative (mean={tl.mean():+.2f} deg)")

    if len(tr) > 0 and tr.mean() <= 0:
        print(f"  [FAIL] TILT_RIGHT roll mean is {tr.mean():+.2f} deg -- should be positive!")
        ok = False
    elif len(tr) > 0:
        print(f"  [PASS] TILT_RIGHT roll is firmly positive (mean={tr.mean():+.2f} deg)")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: Feature engineering output
# ─────────────────────────────────────────────────────────────────────────────

def check_feature_output(df: pd.DataFrame) -> bool:
    print("\n" + "="*60)
    print("CHECK 2 -- Feature engineering output (what the model actually sees)")
    print("="*60)

    try:
        from src.ml.feature_engineering import DatasetBuilder, N_FEAT
    except ImportError as e:
        print(f"  [FAIL] Cannot import feature_engineering: {e}")
        return False

    builder = DatasetBuilder()
    X, y    = builder.fit_transform(args.data)

    feat_names = ["yaw", "pitch", "roll(norm)", "dyaw", "dpitch", "droll",
                  "roll_sign", "roll_raw"][:X.shape[2]]

    print(f"\n  Feature matrix shape: {X.shape}  (samples, frames, features)")
    print(f"  N_FEAT = {X.shape[2]}  (expected 8)\n")

    if X.shape[2] != 8:
        print(f"  [WARN] N_FEAT={X.shape[2]} but expected 8 -- feature_engineering.py may not have been updated!")

    print(f"  {'Class':<14} {'roll_norm(f2) mean':>20} "
          f"{'roll_raw(f7) mean':>20}  {'roll_raw pct<0':>14}")

    ok = True
    for i, cls in enumerate(CLASSES):
        mask = (y == i)
        if mask.sum() == 0:
            print(f"  {cls:<14}  [WARN] no samples")
            continue
        X_cls = X[mask]                     # (n, 30, 8)
        roll_norm = X_cls[:, :, 2].mean()   # feature 2: normalised roll
        if X.shape[2] >= 8:
            roll_raw  = X_cls[:, :, 7].mean()   # feature 7: raw roll
            pct_neg   = (X_cls[:, :, 7] < 0).mean() * 100
            print(f"  {cls:<14}  {roll_norm:>+20.4f}  {roll_raw:>+20.4f}  {pct_neg:>13.1f}%")
        else:
            print(f"  {cls:<14}  {roll_norm:>+20.4f}  (no feature 7)")

    # Critical check: is TILT_LEFT roll_raw firmly negative?
    tl_mask = (y == TILT_L_IDX)
    tr_mask = (y == TILT_R_IDX)
    if tl_mask.sum() > 0 and X.shape[2] >= 8:
        tl_raw = X[tl_mask, :, 7].mean()
        tr_raw = X[tr_mask, :, 7].mean()
        if tl_raw >= -5:
            print(f"\n  [FAIL] TILT_LEFT roll_raw mean = {tl_raw:+.2f} -- should be << 0. Normalisation is collapsing the sign!")
            ok = False
        else:
            print(f"\n  [PASS] TILT_LEFT roll_raw mean = {tl_raw:+.2f} (clearly negative in feature space)")
        sep = abs(tr_raw - tl_raw)
        print(f"  Separation TILT_R - TILT_L = {sep:.2f} units ({'[GOOD] large' if sep > 10 else '[WARN] small -- may still collapse'})")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: 1-rule separability
# ─────────────────────────────────────────────────────────────────────────────

def check_one_rule_separability(df: pd.DataFrame) -> bool:
    print("\n" + "="*60)
    print("CHECK 3 -- One-rule classifier (roll_mean < 0 -> TILT_LEFT)")
    print("="*60)
    print("  If this fails, the CSV data itself is broken.")
    print("  If this passes, the problem is ONLY in training code.\n")

    # Build per-sample roll mean from raw CSV
    records = []
    for sid, grp in df.groupby("sample_id"):
        roll_mean = grp["roll"].mean()
        label     = grp["label"].iloc[0]
        records.append({"sample_id": sid, "roll_mean": roll_mean, "label": label})
    sdf = pd.DataFrame(records)

    # One-rule: if roll_mean < −8 → TILT_LEFT, elif roll_mean > +8 → TILT_RIGHT
    def one_rule(row):
        if row["roll_mean"] < -8:   return "TILT_LEFT"
        if row["roll_mean"] > +8:   return "TILT_RIGHT"
        return "OTHER"

    sdf["pred"] = sdf.apply(one_rule, axis=1)

    tl = sdf[sdf["label"] == "TILT_LEFT"]
    tr = sdf[sdf["label"] == "TILT_RIGHT"]

    tl_correct = (tl["pred"] == "TILT_LEFT").mean()
    tr_correct = (tr["pred"] == "TILT_RIGHT").mean()

    print(f"  TILT_LEFT  correctly identified by roll_mean<-8: "
          f"{tl_correct*100:.1f}%  (n={len(tl)})")
    print(f"  TILT_RIGHT correctly identified by roll_mean>+8: "
          f"{tr_correct*100:.1f}%  (n={len(tr)})")

    # Distribution of roll means per class
    print(f"\n  Roll mean distribution per class:")
    for cls in CLASSES:
        sub = sdf[sdf["label"] == cls]["roll_mean"]
        if len(sub) == 0:
            continue
        print(f"  {cls:<14}  mean={sub.mean():+.2f}  "
              f"std={sub.std():.2f}  "
              f"min={sub.min():+.2f}  max={sub.max():+.2f}")

    ok = tl_correct >= 0.90 and tr_correct >= 0.90
    if ok:
        print(f"\n  [PASS] Data IS separable by a single rule. Problem is in model training, NOT data.")
        print(f"     -> The LSTM is ignoring roll entirely. See Check 4.")
    else:
        print(f"\n  [FAIL] Data is NOT cleanly separable -- CSV is corrupted.")
        print(f"     -> Regenerate the dataset from scratch.")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Check 4: Shallow RF probe
# ─────────────────────────────────────────────────────────────────────────────

def check_rf_probe(df: pd.DataFrame) -> bool:
    print("\n" + "="*60)
    print("CHECK 4 -- Depth-3 Random Forest on raw per-sample features")
    print("="*60)
    print("  If RF gets TILT_LEFT F1 > 0.85 -> the LSTM is the problem.")
    print("  If RF also fails -> feature engineering is broken.\n")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedShuffleSplit

    # Build simple per-sample features: mean/std/min/max of yaw, pitch, roll
    records = []
    for sid, grp in df.groupby("sample_id"):
        row = {"label": grp["label"].iloc[0]}
        for col in ["yaw", "pitch", "roll"]:
            v = grp[col].values
            row[f"{col}_mean"] = v.mean()
            row[f"{col}_std"]  = v.std()
            row[f"{col}_min"]  = v.min()
            row[f"{col}_max"]  = v.max()
            row[f"{col}_range"]= v.max() - v.min()
        records.append(row)

    sdf  = pd.DataFrame(records)
    feat_cols = [c for c in sdf.columns if c != "label"]
    X    = sdf[feat_cols].values.astype(np.float32)
    y    = np.array([CLASSES.index(l) for l in sdf["label"]])

    sss  = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_i, te_i = next(sss.split(X, y))

    rf = RandomForestClassifier(n_estimators=100, max_depth=3,
                                 class_weight="balanced", random_state=42)
    rf.fit(X[tr_i], y[tr_i])
    y_pred = rf.predict(X[te_i])

    print(f"  {'Class':<14} {'F1':>8}  {'Support':>9}")
    print(f"  {'-'*13} {'-'*8}  {'-'*9}")
    ok = True
    for i, cls in enumerate(CLASSES):
        mask = (y[te_i] == i)
        if mask.sum() == 0:
            continue
        tp = ((y_pred == i) & mask).sum()
        fp = ((y_pred == i) & ~mask).sum()
        fn = (mask & (y_pred != i)).sum()
        pr = tp/(tp+fp) if (tp+fp)>0 else 0
        rc = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0
        flag = "✅" if f1 >= 0.85 else "❌"
        print(f"  {flag} {cls:<12}  {f1:>8.4f}  {mask.sum():>9}")
        if cls == "TILT_LEFT" and f1 < 0.85:
            ok = False

    # Feature importances — which features matter most?
    importances = sorted(zip(feat_cols, rf.feature_importances_),
                         key=lambda x: -x[1])
    print(f"\n  Top 5 most important features (RF):")
    for name, imp in importances[:5]:
        print(f"    {name:<20} {imp:.4f}")

    if ok:
        print(f"\n  ✅ RF separates TILT_LEFT well with simple roll stats.")
        print(f"     → LSTM architecture or training is the problem.")
        print(f"     → Use RF as your primary classifier (it works now).")
        print(f"     → OR fix the LSTM with the gradient analysis below.")
    else:
        print(f"\n  ❌ Even a depth-3 tree cannot separate TILT_LEFT.")
        print(f"     → The feature engineering is broken after the CSV stage.")
        print(f"     → Check that sample_to_features() preserves roll sign.")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Final verdict
# ─────────────────────────────────────────────────────────────────────────────

def print_verdict(c1, c2, c3, c4):
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if not c1:
        print("  🔴 CSV data is broken. TILT_LEFT roll values are not negative.")
        print("     Action: regenerate synthetic dataset, re-check generator code.")
    elif not c3:
        print("  🔴 Roll signal is present in CSV but lost after feature engineering.")
        print("     Action: print sample_to_features() output for a TILT_LEFT sample")
        print("             and confirm feature[7] (roll_raw) is negative.")
    elif not c2:
        print("  🟡 Data is fine, 1-rule works, but features fed to model are wrong.")
        print("     Action: confirm N_FEAT=8 is actually being used in training.")
        print("             Add: print(X_train.shape) before model.fit().")
        print("             Confirm model input layer is (30, 8), not (30, 6) or (30, 7).")
    elif c4:
        print("  🟡 RF works, LSTM doesn't. LSTM training is broken.")
        print("     Action A (fastest): replace BiLSTM with RF as primary classifier.")
        print("     Action B (thorough): inspect LSTM gradient flow — roll features")
        print("                          may have near-zero gradients (dead features).")
        print("     → Run: python scripts/check_lstm_gradients.py")
    else:
        print("  🔴 Neither RF nor LSTM can separate TILT_LEFT from raw features.")
        print("     The feature engineering pipeline has a bug that destroys roll sign.")
        print("     Action: add this debug print to sample_to_features():")
        print("       print('roll_raw sample:', feat[-1, 7])  # should be negative for TILT_LEFT")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Path to labeled CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)
    print(f"Loaded {len(df)} rows, {df['sample_id'].nunique()} samples")
    print(f"Class counts: {df.drop_duplicates('sample_id')['label'].value_counts().to_dict()}")

    c1 = check_raw_csv(df)
    c3 = check_one_rule_separability(df)
    c2 = check_feature_output(df)
    c4 = check_rf_probe(df)
    print_verdict(c1, c2, c3, c4)


if __name__ == "__main__":
    main()