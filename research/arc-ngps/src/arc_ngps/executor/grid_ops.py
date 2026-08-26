from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple, Iterable, Optional
import numpy as np


Coord = Tuple[int, int]


@dataclass(frozen=True)
class Obj:
    pixels: Tuple[Coord, ...]
    color: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        ys = [p[0] for p in self.pixels]
        xs = [p[1] for p in self.pixels]
        return min(ys), min(xs), max(ys), max(xs)


def select_color_objs(grid: np.ndarray, color: int, conn8: bool = False) -> List[Obj]:
    """Connected components of a given color."""
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    neigh4 = [(-1,0), (1,0), (0,-1), (0,1)]
    neigh8 = neigh4 + [(-1,-1), (-1,1), (1,-1), (1,1)]
    neigh = neigh8 if conn8 else neigh4

    objs: List[Obj] = []
    for y in range(H):
        for x in range(W):
            if visited[y, x] or grid[y, x] != color:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pix = []
            while stack:
                cy, cx = stack.pop()
                pix.append((cy, cx))
                for dy, dx in neigh:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and (not visited[ny, nx]) and grid[ny, nx] == color:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            objs.append(Obj(pixels=tuple(pix), color=color))
    return objs


def translate_objs(objs: List[Obj], dy: int, dx: int) -> List[Obj]:
    out = []
    for o in objs:
        out.append(Obj(pixels=tuple((y + dy, x + dx) for (y, x) in o.pixels), color=o.color))
    return out


def paint(grid: np.ndarray, objs: List[Obj], color: int) -> np.ndarray:
    out = grid.copy()
    H, W = out.shape
    for o in objs:
        for (y, x) in o.pixels:
            if 0 <= y < H and 0 <= x < W:
                out[y, x] = color
    return out
