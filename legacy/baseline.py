from typing import List, Dict, Any, Callable, Optional, Tuple
import copy

Grid = List[List[int]]

def shape(g: Grid) -> Tuple[int, int]:
    return (len(g), len(g[0]))

def rot90(g: Grid) -> Grid:
    h, w = shape(g)
    return [[g[h-1-y][x] for y in range(h)] for x in range(w)]

def rot180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]]

def rot270(g: Grid) -> Grid:
    return rot90(rot180(g))

def flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]

def flip_v(g: Grid) -> Grid:
    return g[::-1]

def transpose(g: Grid) -> Grid:
    h, w = shape(g)
    return [[g[y][x] for y in range(h)] for x in range(w)]

def equal(a: Grid, b: Grid) -> bool:
    if shape(a) != shape(b): return False
    return all(ra == rb for ra, rb in zip(a, b))


# color mapping
def build_color_bijection(src: Grid, dst: Grid) -> Optional[Dict[int,int]]:
    """
    If shapes equal and we can explain dst by a 1-1 mapping of colors applied to src,
    return that mapping {src_color -> dst_color}. Otherwise None.
    """
    if shape(src) != shape(dst):
        return None
    mapping = {}
    used = set()
    for y in range(len(src)):
        for x in range(len(src[0])):
            s, d = src[y][x], dst[y][x]
            if s in mapping:
                if mapping[s] != d:
                    return None
            else:
                if d in used:
                    return None
                mapping[s] = d
                used.add(d)
    return mapping


def apply_color_map(g: Grid, mp: Dict[int,int]) -> Grid:
    return [[mp.get(c, c) for c in row] for row in g]


# candidate transforms
def identity(g: Grid) -> Grid: return g
GEOM_FUNCS: List[Callable[[Grid], Grid]] = [identity, rot90, rot180, rot270, flip_h, flip_v, transpose]


class Transform:
    """
    Represents either:
      - a pure geometry op: f(g)
      - a color-map op: lambda g: apply_color_map(g, mp)
      - a composition: color-map after geometry (optional)
    """
    def __init__(self, geom: Callable[[Grid], Grid] = identity, cmap: Optional[Dict[int,int]] = None):
        self.geom = geom
        self.cmap = cmap

    def __call__(self, g: Grid) -> Grid:
        h = self.geom(g)
        return apply_color_map(h, self.cmap) if self.cmap else h
    
    def __repr__(self):
        parts = [self.geom.__name__]
        if self.cmap: parts.append(f"cmap={self.cmap}")
        return "Transform(" + ", ".join(parts) + ")"
    

def synthesize_transform(train_pairs: List[Dict[str, Grid]]) -> Optional[Transform]:
    """
    Try geometry-only first; then geometry + color bijection inferred from the first example.
    We require a single transform that works for all train pairs.
    """
    # 1) geometry only
    for gf in GEOM_FUNCS:
        ok = True
        for ex in train_pairs:
            if not equal(gf(ex["input"]), ex["output"]):
                ok = False
                break
        if ok:
            return Transform(geom=gf)
        
    # 2) geometry + color bijection (infer from first example)
    for gf in GEOM_FUNCS:
        # infer a color map on the first example; reuse across others
        g0_in = gf(train_pairs[0]["input"])
        g0_out = train_pairs[0]["output"]
        cmap = build_color_bijection(g0_in, g0_out)
        if cmap is None:
            continue
        t = Transform(geom=gf, cmap=cmap)
        if all(equal(t(ex["input"]), ex["output"]) for ex in train_pairs):
            return t
        
    # extend here: add crop/pad, tiling, majority-fill, etc.
    return None


def predict_task(task: Dict[str, Any]) -> List[Grid]:
    """
    Returns predictions for each test input in the task.
    """
    t = synthesize_transform(task["train"])
    if t is None:
        preds = [None for _ in task.get("test", [])]
        return preds, t
    preds = [t(ex["input"]) for ex in task.get("test", [])]
    return preds, t