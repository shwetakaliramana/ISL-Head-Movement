#!/usr/bin/env python3
"""Test module import with error handling."""

import sys
import traceback

sys.path.insert(0, "E:\\git Projects\\Computer_Vision")

try:
    print("Attempting to import module...")
    from scripts import generate_synthetic_dataset
    print(f"Module imported successfully: {generate_synthetic_dataset}")
    print(f"gen_tilt_left in module: {hasattr(generate_synthetic_dataset, 'gen_tilt_left')}")
    
    if hasattr(generate_synthetic_dataset, 'gen_tilt_left'):
        gen_tilt_left = generate_synthetic_dataset.gen_tilt_left
        print(f"gen_tilt_left function: {gen_tilt_left}")
        print(f"Function source file: {gen_tilt_left.__code__.co_filename}")
        print()
        
        import numpy as np
        rng = np.random.default_rng(42)
        print("Calling function...")
        result = gen_tilt_left(rng, noise_std=0.6)
        print(f"Got result: {result.shape}")
        
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
