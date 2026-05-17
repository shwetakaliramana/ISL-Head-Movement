"""
Diagnose Test 4 failure: identify normalization mismatch in synthetic data.
"""
import sys, tempfile, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.ml.feature_engineering import DatasetBuilder, CLASSES, WINDOW, N_FEAT

def make_perfect_csv(path, n=80):
    """Create synthetic CSV with well-known TILT values."""
    import pandas as pd
    rows = []
    sid = 0

    # TILT_LEFT: roll = -20 (all frames const)
    for _ in range(n):
        for i in range(WINDOW):
            rows.append([sid, i, 0.0, 0.0, -20.0, 0.0, 0.0, 0.0, "TILT_LEFT"])
        sid += 1

    # TILT_RIGHT: roll = +20 (all frames const)
    for _ in range(n):
        for i in range(WINDOW):
            rows.append([sid, i, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, "TILT_RIGHT"])
        sid += 1

    df = pd.DataFrame(rows, columns=["sample_id", "frame", "yaw", "pitch", "roll", "dyaw", "dpitch", "droll", "label"])
    df.to_csv(path, index=False)

print("="*70)
print("  TEST 4 FAILURE DIAGNOSIS")
print("="*70)

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
    csv_path = Path(f.name)

make_perfect_csv(csv_path, n=40)
print(f"\n[1] Created synthetic CSV with:")
print(f"    TILT_LEFT (40 samples, n_frames=30):  roll = -20.0")
print(f"    TILT_RIGHT (40 samples, n_frames=30): roll = +20.0")

b = DatasetBuilder()
X, y = b.fit_transform(csv_path)
csv_path.unlink(missing_ok=True)

print(f"\n[2] Fitted normalizer stats:")
print(f"    mean (6 feat): {np.round(b._mean, 4)}")
print(f"    std  (6 feat): {np.round(b._std,  4)}")

tl_idx = CLASSES.index("TILT_LEFT")
tr_idx = CLASSES.index("TILT_RIGHT")

tl_mask = (y == tl_idx)
tr_mask = (y == tr_idx)

print(f"\n[3] Training data feature 2 (roll_norm) per class:")
print(f"    TILT_LEFT  [idx={tl_idx}]: mean={X[tl_mask][:,:,2].mean():+.4f}, std={X[tl_mask][:,:,2].std():.4f}")
print(f"    TILT_RIGHT [idx={tr_idx}]: mean={X[tr_mask][:,:,2].mean():+.4f}, std={X[tr_mask][:,:,2].std():.4f}")

print(f"\n[4] Training data feature 6 (roll_sign) and 7 (roll_abs) per class:")
print(f"    TILT_LEFT  roll_sign: mean={X[tl_mask][:,:,6].mean():+.4f}")
print(f"    TILT_LEFT  roll_abs:  mean={X[tl_mask][:,:,7].mean():+.4f}")
print(f"    TILT_RIGHT roll_sign: mean={X[tr_mask][:,:,6].mean():+.4f}")
print(f"    TILT_RIGHT roll_abs:  mean={X[tr_mask][:,:,7].mean():+.4f}")

# Now test a perfect window
print(f"\n[5] Perfect TILT_LEFT window (roll=-20, everything else 0):")
feat6_tl = np.zeros((WINDOW, 6), dtype=np.float32)
feat6_tl[:, 2] = -20.0

w_tl = b.transform_window(feat6_tl)[0]
print(f"    Frame 0 all features: {np.round(w_tl[0], 4)}")
print(f"    roll_norm (feat 2):   {w_tl[0, 2]:+.4f} (expected ≈ -1.5810)")
print(f"    roll_sign (feat 6):   {w_tl[0, 6]:+.4f} (expected -1.0)")
print(f"    roll_abs  (feat 7):   {w_tl[0, 7]:+.4f} (expected ≈0.444)")

print(f"\n[6] Perfect TILT_RIGHT window (roll=+20, everything else 0):")
feat6_tr = np.zeros((WINDOW, 6), dtype=np.float32)
feat6_tr[:, 2] = 20.0

w_tr = b.transform_window(feat6_tr)[0]
print(f"    Frame 0 all features: {np.round(w_tr[0], 4)}")
print(f"    roll_norm (feat 2):   {w_tr[0, 2]:+.4f} (expected ≈ +1.5810)")
print(f"    roll_sign (feat 6):   {w_tr[0, 6]:+.4f} (expected +1.0)")
print(f"    roll_abs  (feat 7):   {w_tr[0, 7]:+.4f} (expected ≈0.444)")

print(f"\n[7] DIAGNOSIS:")
tl_norm_expected = -1.5810
tr_norm_expected = 1.5810
tl_norm_actual = X[tl_mask][:,:,2].mean()
tr_norm_actual = X[tr_mask][:,:,2].mean()

tol = 0.1
tl_ok = abs(tl_norm_actual - tl_norm_expected) < tol
tr_ok = abs(tr_norm_actual - tr_norm_expected) < tol

if tl_ok and tr_ok:
    print("    ✅ Training data roll_norm values are CORRECT")
else:
    print("    ❌ Training data roll_norm values are WRONG:")
    if not tl_ok:
        print(f"       TILT_LEFT expected {tl_norm_expected:+.4f}, got {tl_norm_actual:+.4f}")
    if not tr_ok:
        print(f"       TILT_RIGHT expected {tr_norm_expected:+.4f}, got {tr_norm_actual:+.4f}")

print()
