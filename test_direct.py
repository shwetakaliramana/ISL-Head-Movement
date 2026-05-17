#!/usr/bin/env python3
"""Direct test of gen_tilt_left."""

import sys
sys.path.insert(0, "E:\\git Projects\\Computer_Vision")

import numpy as np
from scripts.generate_synthetic_dataset import gen_tilt_left

print("Starting direct test...")
print(f"gen_tilt_left: {gen_tilt_left}")
print()

rng = np.random.default_rng(42)
print("Calling gen_tilt_left...")
sample = gen_tilt_left(rng, noise_std=0.6)
print(f"Got sample: {sample.shape}")
print(f"Roll: {sample[:, 2]}")
