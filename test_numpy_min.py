#!/usr/bin/env python3
"""Test numpy minimum directly."""

import numpy as np

# Test 1: Simple minimum operation
a = np.array([0.45, -3, -8, -12, -15])
b = np.minimum(a, -12.0)
print(f"Test 1 - Simple minimum:")
print(f"  Input:  {a}")
print(f"  Output: {b}")
print(f"  Expected: [-12, -12, -12, -12, -15]")
print()

# Test 2: In-place assignment
a = np.array([0.45, -3, -8, -12, -15])
a = np.minimum(a, -12.0)
print(f"Test 2 - In-place assignment:")
print(f"  Output: {a}")
print()

# Test 3: Test with random data like the generator
rng = np.random.default_rng(42)
roll = rng.uniform(0, -20, 30)  # Random between 0 and -20
print(f"Test 3 - Random data:")
print(f"  Before: min={roll.min():.2f}, max={roll.max():.2f}")
roll = np.minimum(roll, -12.0)
print(f"  After:  min={roll.min():.2f}, max={roll.max():.2f}")
print(f"  All <= -12? {(roll <= -12.0).all()}")
