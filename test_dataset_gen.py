#!/usr/bin/env python
"""Test script to debug dataset generation."""
import sys
import traceback

try:
    print("[TEST] Starting test script...", flush=True)
    from pathlib import Path
    print("[TEST] Imports OK", flush=True)
    
    sys.path.insert(0, str(Path('.').resolve()))
    print("[TEST] Path setup OK", flush=True)
    
    import scripts.generate_synthetic_dataset as gen_module
    print("[TEST] Module import OK", flush=True)
    
    # Try to build a small dataset
    print("[TEST] Building dataset...", flush=True)
    df = gen_module.build_dataset(samples_per_class=10, noise_std=1.5, seed=42, augment_factor=1)
    print(f"[TEST] Dataset built: {len(df)} rows", flush=True)
    
    # Try to save it
    out_path = Path("data/raw/test_dataset.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[TEST] Saved to {out_path}", flush=True)
    
except Exception as e:
    print(f"[ERROR] {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
