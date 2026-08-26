from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch


@dataclass
class TokenizedGrid:
    tokens: torch.LongTensor      # [N] color ids 0..9
    pos: torch.LongTensor         # [N,2] (y,x)
    mask: torch.BoolTensor        # [N] True for valid
    hw: Tuple[int, int]


def grid_to_tokens(grid: np.ndarray) -> TokenizedGrid:
    """Flatten a grid to variable-length pixel tokens."""
    h, w = grid.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    tokens = torch.as_tensor(grid.reshape(-1), dtype=torch.long)
    pos = torch.stack([
        torch.as_tensor(yy.reshape(-1), dtype=torch.long),
        torch.as_tensor(xx.reshape(-1), dtype=torch.long),
    ], dim=-1)
    mask = torch.ones(tokens.shape[0], dtype=torch.bool)
    return TokenizedGrid(tokens=tokens, pos=pos, mask=mask, hw=(h, w))


def pad_tokenized_grids(batch: List[TokenizedGrid], pad_to: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Pad variable-length token grids to [B, Nmax]."""
    nmax = max(t.tokens.numel() for t in batch)
    if pad_to is not None:
        nmax = max(nmax, pad_to)

    B = len(batch)
    tokens = torch.zeros((B, nmax), dtype=torch.long)
    pos = torch.zeros((B, nmax, 2), dtype=torch.long)
    mask = torch.zeros((B, nmax), dtype=torch.bool)
    hw = torch.zeros((B, 2), dtype=torch.long)

    for i, t in enumerate(batch):
        n = t.tokens.numel()
        tokens[i, :n] = t.tokens
        pos[i, :n] = t.pos
        mask[i, :n] = t.mask
        hw[i] = torch.tensor([t.hw[0], t.hw[1]], dtype=torch.long)

    return {"tokens": tokens, "pos": pos, "mask": mask, "hw": hw}
