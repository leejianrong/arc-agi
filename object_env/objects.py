"""Segmentation and region derivation over the vendored arc-dsl.

`dsl.objects(grid, univalued, diagonal, without_bg)` is the executor underneath
(ADR-0001 generalized). The three connectivity booleans are a real degree of
freedom the shipped `arc_env.actions` already exercises per selector, so the
named selectors here bake in the same variant each carries there — e.g.
`select_tallest` segments with `(True, False, False)` and picks by height,
matching `arc_env.actions._select_tallest`.
"""

from arc_env._dsl import dsl
from object_env.types import Grid, Obj


def _to_obj(dsl_obj) -> Obj:
    """Convert one arc-dsl object (a frozenset of `(value, (i, j))`) to an
    `Obj` — its color plus absolute cell indices."""
    return Obj(color=dsl.color(dsl_obj), cells=dsl.toindices(dsl_obj))


def segment(grid: Grid, univalued: bool, diagonal: bool, without_bg: bool) -> list[Obj]:
    """All objects on `grid` under the given connectivity, as `Obj`s."""
    return [_to_obj(o) for o in dsl.objects(grid, univalued, diagonal, without_bg)]


# --- named selectors: each fixes a connectivity variant + a pick rule,
#     mirroring the shipped `arc_env.actions` selectors 1:1. Return None (a
#     grammar no-op) on an empty grid so preconditions stay honest. ---
def _pick(grid, univalued, diagonal, without_bg, key) -> Obj | None:
    objs = segment(grid, univalued, diagonal, without_bg)
    if not objs:
        return None
    return max(objs, key=key)


def select_largest(grid: Grid) -> Obj | None:
    return _pick(grid, True, True, True, lambda o: o.size)


def select_smallest(grid: Grid) -> Obj | None:
    objs = segment(grid, True, True, True)
    return min(objs, key=lambda o: o.size) if objs else None


def select_largest_no_diag(grid: Grid) -> Obj | None:
    return _pick(grid, True, False, True, lambda o: o.size)


def select_tallest(grid: Grid) -> Obj | None:
    return _pick(grid, True, False, False, lambda o: o.height)


def select_by_color(grid: Grid, color: int) -> Obj | None:
    """The merged cells of every object of `color` (one addressable Object),
    matching `arc_env.actions._select_by_color`'s `colorfilter`+`merge`."""
    cells = frozenset(
        cell for o in segment(grid, True, True, True) if o.color == color for cell in o.cells
    )
    return Obj(color=color, cells=cells) if cells else None
