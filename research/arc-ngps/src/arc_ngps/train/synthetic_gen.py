from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

# Placeholder: implement DSL sampling + rendering here for large-scale pretraining.
# Goal:
# 1) sample a typed program from DSL
# 2) sample input grids
# 3) execute program to get outputs
# 4) produce (train pairs, target DSL tokens) for supervised learning

def sample_synthetic_task(rng: np.random.Generator) -> Dict[str, Any]:
    # Very small demo: program = paint( grid, translate(select_color(grid, c), dy, dx), c2 )
    H = rng.integers(5, 11)
    W = rng.integers(5, 11)
    grid = rng.integers(0, 3, size=(H, W), dtype=np.int64)
    return {"train": [(grid, grid.copy())], "dsl_tokens": [1, 2, 3]}  # stub
