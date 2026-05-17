#!/usr/bin/env python3
"""Quick diagnostic for tilt generators."""

import numpy as np
import sys
sys.path.insert(0, ".")
from scripts.generate_synthetic_dataset import gen_tilt_left, gen_tilt_right, WINDOW

rng = np.random.default_rng(0)

for i in range(10):
    left  = gen_tilt_left(rng, 1.2)
    right = gen_tilt_right(rng, 1.2)
    print(f"LEFT  roll mean={left[:,2].mean():+.1f}°  min={left[:,2].min():+.1f}°")
    print(f"RIGHT roll mean={right[:,2].mean():+.1f}°  max={right[:,2].max():+.1f}°")
    assert left[:,2].mean() < -10, "LEFT tilt not negative enough!"
    assert right[:,2].mean() > 10, "RIGHT tilt not positive enough!"

print("\n✅ Tilt generators are correctly separated")
