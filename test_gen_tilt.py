#!/usr/bin/env python3
"""Test gen_tilt_left directly to see if clipping works."""

import sys
from pathlib import Path
import importlib.util

# Resolve absolute path
root = Path(__file__).resolve().parents[0]
script_path = root / "scripts" / "generate_synthetic_dataset.py"

print(f"[DEBUG] Loading from: {script_path}")
print(f"[DEBUG] File exists: {script_path.exists()}")

# Import directly from the generation script
spec = importlib.util.spec_from_file_location("gen_module", str(script_path))
gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_module)

import numpy as np

print(f"[DEBUG] gen_tilt_left function: {gen_module.gen_tilt_left}")
print()

# Test gen_tilt_left directly
rng = np.random.default_rng(42)
for i in range(3):
    print(f"=== Sample {i} ===")
    sample = gen_module.gen_tilt_left(rng, noise_std=0.6)
    roll = sample[:, 2]
    print(f"  Roll: {roll}")
    print(f"  Max: {roll.max():.2f}, Min: {roll.min():.2f}")
    print(f"  Any > -12? {(roll > -12).any()}")
    print()

