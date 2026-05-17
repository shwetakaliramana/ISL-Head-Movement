"""
scripts/verify_tilt_fix.py  v3 — fixed test helper uses fitted stats
Run: python scripts/verify_tilt_fix.py
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = "  ✅"; FAIL = "  ❌"

# ─── synthetic CSV ────────────────────────────────────────────────────────────
def _make_csv(path, n=80):
    import pandas as pd
    rows, t = [], 0.0
    for _ in range(n):
        for i in range(30):
            p=15*np.sin(2*np.pi*i/15); rows.append([t,0,p,0,0,0,0,"NOD"]); t+=0.033
    for _ in range(n):
        for i in range(30):
            y=15*np.sin(2*np.pi*i/15); rows.append([t,y,0,0,0,0,0,"SHAKE"]); t+=0.033
    for _ in range(n):
        for i in range(30):
            rows.append([t,0,0,-20,0,0,0,"TILT_LEFT"]); t+=0.033
    for _ in range(n):
        for i in range(30):
            rows.append([t,0,0,20,0,0,0,"TILT_RIGHT"]); t+=0.033
    for _ in range(n):
        for i in range(30):
            rows.append([t,np.random.randn()*.4,np.random.randn()*.4,
                         np.random.randn()*.4,0,0,0,"STATIC"]); t+=0.033
    pd.DataFrame(rows,
        columns=["timestamp","yaw","pitch","roll","dyaw","dpitch","droll","label"]
    ).to_csv(path, index=False)

# ─── window helper — MUST use fitted builder ─────────────────────────────────
def _window(builder, cls):
    """Build a perfect (1, WINDOW, N_FEAT) window using FITTED normaliser stats."""
    from src.ml.feature_engineering import WINDOW
    roll = {"TILT_LEFT":-25.0,"TILT_RIGHT":25.0}.get(cls,0.0)
    frames=[]
    for i in range(WINDOW):
        y = 15*np.sin(2*np.pi*i/15) if cls=="SHAKE" else 0.0
        p = 15*np.sin(2*np.pi*i/15) if cls=="NOD"   else 0.0
        frames.append([y,p,roll,0.0,0.0,0.0])
    return builder.transform_window(np.array(frames, dtype=np.float32))

# ─── Test 1 ───────────────────────────────────────────────────────────────────
def test1():
    print("\n── Test 1: Feature Engineering + Label Mapping ─────────────")
    from src.ml.feature_engineering import DatasetBuilder, WINDOW, N_FEAT, LABEL2IDX, CLASSES
    ok = True

    # Label mapping must be fixed order, not alphabetic
    expected = {"NOD":0,"SHAKE":1,"TILT_LEFT":2,"TILT_RIGHT":3,"STATIC":4}
    for cls, exp_idx in expected.items():
        got = LABEL2IDX[cls]
        st  = PASS if got==exp_idx else FAIL
        if got!=exp_idx: ok=False
        print(f"{st} LABEL2IDX[{cls!r}] = {got}  (expected {exp_idx})")

    # roll_sign / roll_abs with real fitted stats
    b = DatasetBuilder(); b._mean=np.zeros(6,dtype=np.float32); b._std=np.ones(6,dtype=np.float32)
    for cls, exp in [("TILT_LEFT",-1.0),("TILT_RIGHT",1.0)]:
        w=b.transform_window(np.column_stack([
            np.zeros((WINDOW,2)), np.full((WINDOW,1), {"TILT_LEFT":-25,"TILT_RIGHT":25}[cls]),
            np.zeros((WINDOW,3))]).astype(np.float32))[0]
        rs=w[:,6]; ra=w[:,7]
        if np.all(rs==exp): print(f"{PASS} {cls:<14} roll_sign={exp:+.0f} ✓")
        else: print(f"{FAIL} {cls:<14} roll_sign wrong: {np.unique(rs)}"); ok=False
        if ra.mean()>0.3: print(f"{PASS} {cls:<14} roll_abs={ra.mean():.3f} ✓")
        else: print(f"{FAIL} {cls:<14} roll_abs too small: {ra.mean():.4f}"); ok=False
    return ok

# ─── Test 2 ───────────────────────────────────────────────────────────────────
def test2():
    print("\n── Test 2: Model Architecture ──────────────────────────────")
    from src.ml.bilstm_model import build_model
    from src.ml.feature_engineering import DatasetBuilder, CLASSES
    model=build_model()
    b=DatasetBuilder(); b._mean=np.zeros(6,dtype=np.float32); b._std=np.ones(6,dtype=np.float32)
    ok=True
    for cls in CLASSES:
        prob=model.predict(_window(b,cls),verbose=0)[0]
        good=not np.any(np.isnan(prob)) and abs(prob.sum()-1.0)<1e-4 and prob.shape==(5,)
        st=PASS if good else FAIL
        if not good: ok=False
        print(f"{st} {cls:<14} sum={prob.sum():.5f} shape={prob.shape}")
    return ok

# ─── Test 3 ───────────────────────────────────────────────────────────────────
def test3():
    print("\n── Test 3: Linear Separability ─────────────────────────────")
    from sklearn.linear_model import LogisticRegression
    from src.ml.feature_engineering import DatasetBuilder, WINDOW
    b=DatasetBuilder(); b._mean=np.zeros(6,dtype=np.float32); b._std=np.ones(6,dtype=np.float32)
    rng=np.random.default_rng(42); X,y=[],[]
    for lbl,yi in [("TILT_LEFT",0),("TILT_RIGHT",1)]:
        for _ in range(150):
            w=_window(b,lbl)[0]
            X.append([w[:,6].mean()+rng.normal(0,.01), w[:,7].mean()+rng.normal(0,.01)]); y.append(yi)
    acc=LogisticRegression(max_iter=200).fit(np.array(X),np.array(y)).score(np.array(X),np.array(y))
    if acc>=1.0: print(f"{PASS} Accuracy={acc:.4f} — linearly separable"); return True
    else: print(f"{FAIL} Accuracy={acc:.4f} < 1.0"); return False

# ─── Test 4 ───────────────────────────────────────────────────────────────────
def test4():
    print("\n── Test 4: Mini Training (20 epochs) ───────────────────────")
    print("    (~20-40 s on CPU)")
    import tensorflow as tf
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from src.ml.bilstm_model import build_model, get_class_weights
    from src.ml.feature_engineering import DatasetBuilder, CLASSES, LABEL2IDX

    with tempfile.NamedTemporaryFile(suffix=".csv",delete=False) as f:
        csv_path=Path(f.name)
    _make_csv(csv_path,n=80)
    b=DatasetBuilder(); X,y=b.fit_transform(csv_path,fit_stats=True)
    csv_path.unlink(missing_ok=True)

    print(f"  Normaliser mean={np.round(b._mean,3)}")
    print(f"  Normaliser std ={np.round(b._std,3)}")

    # Verify label indices in built dataset
    print("\n  Class distribution in built X,y:")
    all_present=True
    for i,cls in enumerate(CLASSES):
        n=int((y==i).sum())
        marker="✓" if n>0 else "← MISSING!"
        print(f"    [{i}] {cls:<14} n={n} {marker}")
        if n==0: all_present=False
    if not all_present:
        print(f"{FAIL} Missing classes in dataset — label mapping still broken"); return False

    # Verify TILT roll_norm sign
    tl_mask=(y==LABEL2IDX["TILT_LEFT"]); tr_mask=(y==LABEL2IDX["TILT_RIGHT"])
    tl_roll_mean=X[tl_mask][:,:,2].mean(); tr_roll_mean=X[tr_mask][:,:,2].mean()
    print(f"\n  TILT_LEFT  roll_norm mean = {tl_roll_mean:.4f}  (should be negative)")
    print(f"  TILT_RIGHT roll_norm mean = {tr_roll_mean:.4f}  (should be positive)")
    if tl_roll_mean >= 0:
        print(f"{FAIL} TILT_LEFT roll_norm is positive — labels still swapped!"); return False
    if tr_roll_mean <= 0:
        print(f"{FAIL} TILT_RIGHT roll_norm is negative — labels still swapped!"); return False
    print(f"{PASS} roll_norm signs correct")

    X_tr,X_va,y_tr,y_va=train_test_split(X,y,test_size=0.20,stratify=y,random_state=42)
    cw=get_class_weights(y_tr)
    model=build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    tf.get_logger().setLevel("ERROR")
    model.fit(X_tr,y_tr,validation_data=(X_va,y_va),
              epochs=20,batch_size=16,class_weight=cw,verbose=0)
    tf.get_logger().setLevel("WARNING")

    preds=np.argmax(model.predict(X_va,verbose=0),axis=1)
    rpt=classification_report(y_va,preds,target_names=CLASSES,output_dict=True,zero_division=0)
    ok=True
    print()
    for cls in CLASSES:
        f1=rpt.get(cls,{}).get("f1-score",0.0); rec=rpt.get(cls,{}).get("recall",0.0)
        thr=0.80 if cls in ("TILT_LEFT","TILT_RIGHT") else 0.60
        st=PASS if f1>=thr else FAIL
        if f1<thr: ok=False
        print(f"{st} {cls:<14} F1={f1:.4f} Recall={rec:.4f} (thr={thr:.2f})")

    # Perfect window test using FITTED builder
    print()
    for cls,exp_idx in [("TILT_LEFT",LABEL2IDX["TILT_LEFT"]),
                        ("TILT_RIGHT",LABEL2IDX["TILT_RIGHT"])]:
        x=_window(b,cls)   # ← uses fitted b, not zeroed builder
        print(f"  {cls} window roll_norm[0]={x[0,0,2]:.4f}  roll_sign={x[0,0,6]:.1f}")
        prob=model.predict(x,verbose=0)[0]; pred=int(np.argmax(prob))
        st=PASS if pred==exp_idx else FAIL
        if pred!=exp_idx: ok=False
        print(f"{st} Perfect {cls:<14} → {CLASSES[pred]} (P={prob[exp_idx]:.3f})")
    return ok

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("="*60); print("  ISL TILT Fix v3 — Verification Suite"); print("="*60)
    results={
        "Label mapping + features": test1(),
        "Model architecture":       test2(),
        "Linear separability":      test3(),
        "Mini training (20ep)":     test4(),
    }
    print("\n"+"="*60); print("  SUMMARY"); print("="*60)
    all_pass=True
    for name,passed in results.items():
        print(f"{'  ✅' if passed else '  ❌'}  {name}")
        if not passed: all_pass=False
    print("="*60)
    if all_pass:
        print("\n  All tests passed.\n")
        print("  python scripts/train_bilstm.py \\")
        print("      --data  data/raw/synthetic_dataset.csv \\")
        print("      --stats models/normaliser_stats.npz \\")
        print("      --focal --lr 3e-4 --epochs 150\n")
    else:
        print("\n  Tests FAILED — check output above.\n")

if __name__=="__main__":
    main()