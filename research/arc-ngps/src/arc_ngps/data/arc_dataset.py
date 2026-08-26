from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import json
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ARCPair:
    inp: np.ndarray  # [H,W] ints 0..9
    out: np.ndarray  # [H,W] ints 0..9


@dataclass(frozen=True)
class ARCTask:
    task_id: str
    train: List[ARCPair]
    test: List[np.ndarray]  # list of input grids


def _to_np_grid(x: List[List[int]]) -> np.ndarray:
    a = np.asarray(x, dtype=np.int64)
    assert a.ndim == 2
    assert a.min() >= 0 and a.max() <= 9
    return a


def load_arc_task(path: str | Path) -> ARCTask:
    path = Path(path)
    raw = json.loads(path.read_text())
    train_pairs = []
    for ex in raw["train"]:
        train_pairs.append(ARCPair(_to_np_grid(ex["input"]), _to_np_grid(ex["output"])))
    test_inputs = [_to_np_grid(ex["input"]) for ex in raw.get("test", [])]
    return ARCTask(task_id=path.stem, train=train_pairs, test=test_inputs)


class ARCDataset(Dataset):
    """Loads ARC tasks from a directory of JSON files (Kaggle / ARC format)."""

    def __init__(self, root_dir: str | Path, split: str = "training"):
        self.root_dir = Path(root_dir)
        self.split = split
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")
        self.paths = sorted(split_dir.glob("*.json"))
        if not self.paths:
            raise RuntimeError(f"No ARC JSON files in {split_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> ARCTask:
        return load_arc_task(self.paths[idx])
