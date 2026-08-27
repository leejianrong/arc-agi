"""The curated discrete action space over `arc-dsl` primitives.

Per ADR-0001, actions call `arc-dsl` primitives directly - no separate AST or
executor layer of our own. Higher-order primitives (`compose`, `chain`,
`fork`, `rbind`, `lbind`, `power`) are excluded for good: they build
closures rather than transforming a grid, which a flat "pick one primitive
per step" action space can't represent (see `docs/adr/0001*.md`).

Beyond that exclusion, this module is further restricted to primitives whose
signature is `Grid [, scalar args] -> Grid` - i.e. primitives that operate
directly on the single grid the agent is editing, with typed scalar
arguments (a color, a scale factor, a coordinate, a dimension) rather than
an `Object`/`Indices`/`Callable` argument that would require other, unpicked
primitives to construct. This is what makes the action space "directly
steppable": every action maps one grid to the next.

This scalar-args-only restriction was derived by checking, for every
ARC-AGI-1 training task with a known-correct `arc-dsl` solver, whether that
solver only calls primitives from this module's action groups below.
Exactly 16 tasks qualify (11 same-shape, from V1; 5 variable-shape, added in
V3) - see `arc_env/task_loader.py`:CURATED_TASK_IDS - which is the curated
task subset and also the regression-test fixture set
(`tests/test_dsl_regression.py`).

V1 additionally restricted the *task subset* to same-shape-only pairs, as
the smallest possible first slice (ADR-0002) - not because the action space
couldn't already produce a variable-shape output via `trim`/`compress`/
`tophalf`/.../`upscale`/`downscale`. V3 lifts that task-subset restriction
and adds the two primitives ADR-0002 actually calls for: `canvas` (build a
fresh grid of any size up to 30x30, replacing the current one - the escape
hatch for outputs that aren't reachable by transforming the input at all)
and `commit` (crop the current grid to a chosen sub-region *and* end the
episode there - ADR-0002's "explicit commit-output action", fused with
`crop` itself since committing is the only reason to crop). No other
Object/Indices/Callable-arg primitives are added - ADR-0002 is explicit
that this is a bounded, one-action-plus-two-primitives growth, not
open-ended.

`fill_cell` and `canvas` are the two actions here NOT drawn from the solver
analysis: `fill_cell` is `dsl.fill(grid, color, patch)` with `patch` built
from the action's own (row, col) args as a single-cell `Indices` literal,
included so the curated space has at least one pixel-level edit (no
fixture task needs it, but structural transforms alone can never solve a
task that needs localized recoloring). `canvas` is included per ADR-0002's
decision even though none of the 5 new V3 fixture tasks happens to call it
directly - the general "build from scratch" escape hatch is the point of
the primitive, not any one task in this small a subset.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from arc_env._dsl import dsl, constants

Grid = tuple


@dataclass(frozen=True)
class ArgSpec:
    """One decoded argument slot for an action.

    `kind` documents intent; `decode` maps a raw `Discrete(RAW_ARG_RANGE)`
    sample (as produced by the env's factored action space) to the actual
    value passed to the DSL primitive.
    """

    name: str
    kind: str  # "color" | "factor" | "coord" | "dim"
    decode: Callable[[int], int]


def _decode_color(raw: int) -> int:
    return raw % 10


def _decode_factor(raw: int) -> int:
    return 2 + (raw % 3)  # {2, 3, 4}


def _decode_coord(raw: int) -> int:
    return raw  # clamped/validated against actual grid bounds at exec time


def _decode_dim(raw: int) -> int:
    return raw + 1  # {1, ..., 30} - RAW_ARG_RANGE is 30, so this covers the full canvas


COLOR_ARG = lambda name: ArgSpec(name, "color", _decode_color)
FACTOR_ARG = lambda name: ArgSpec(name, "factor", _decode_factor)
COORD_ARG = lambda name: ArgSpec(name, "coord", _decode_coord)
DIM_ARG = lambda name: ArgSpec(name, "dim", _decode_dim)


@dataclass(frozen=True)
class Action:
    """One curated action: a name (always the underlying `arc-dsl` primitive
    name, except for the derived `fill_cell`), the callable, and its typed
    argument slots (empty for zero-arg actions)."""

    name: str
    fn: Callable[..., Grid]
    args: tuple = field(default_factory=tuple)

    @property
    def arity(self) -> int:
        return len(self.args)


def _fill_cell(grid: Grid, color: int, row: int, col: int) -> Grid:
    return dsl.fill(grid, color, frozenset({(row, col)}))


def _canvas(grid: Grid, value: int, height: int, width: int) -> Grid:
    return dsl.canvas(value, (height, width))  # replaces `grid` outright; ignores it


def _commit(grid: Grid, row: int, col: int, height: int, width: int) -> Grid:
    return dsl.crop(grid, (row, col), (height, width))


# Zero-arg grid transforms.
ZERO_ARG = [
    Action("identity", dsl.identity),
    Action("rot90", dsl.rot90),
    Action("rot180", dsl.rot180),
    Action("rot270", dsl.rot270),
    Action("hmirror", dsl.hmirror),
    Action("vmirror", dsl.vmirror),
    Action("dmirror", dsl.dmirror),
    Action("cmirror", dsl.cmirror),
    Action("trim", dsl.trim),
    Action("compress", dsl.compress),
    Action("tophalf", dsl.tophalf),
    Action("bottomhalf", dsl.bottomhalf),
    Action("lefthalf", dsl.lefthalf),
    Action("righthalf", dsl.righthalf),
]

# One-arg (scale factor) grid transforms.
ONE_ARG = [
    Action("hupscale", dsl.hupscale, (FACTOR_ARG("factor"),)),
    Action("vupscale", dsl.vupscale, (FACTOR_ARG("factor"),)),
    Action("downscale", dsl.downscale, (FACTOR_ARG("factor"),)),
    Action("upscale", dsl.upscale, (FACTOR_ARG("factor"),)),
]

# Two-arg (color pair) grid transforms.
TWO_ARG = [
    Action("replace", dsl.replace, (COLOR_ARG("replacee"), COLOR_ARG("replacer"))),
    Action("switch", dsl.switch, (COLOR_ARG("a"), COLOR_ARG("b"))),
]

# Three-arg (color, row, col) pixel edit, and (color, height, width) fresh-
# canvas construction (ADR-0002). See module docstring.
THREE_ARG = [
    Action("fill_cell", _fill_cell, (COLOR_ARG("color"), COORD_ARG("row"), COORD_ARG("col"))),
    Action("canvas", _canvas, (COLOR_ARG("value"), DIM_ARG("height"), DIM_ARG("width"))),
]

# Four-arg (row, col, height, width) crop-and-end-episode (ADR-0002's
# "commit" action). See module docstring and `execute`'s commit-specific
# bounds check below (row + height <= grid height, col + width <= grid width
# - a cross-argument constraint the generic per-arg validity loop can't
# express, so it's handled as a one-off rather than a general mechanism).
FOUR_ARG = [
    Action("commit", _commit, (COORD_ARG("row"), COORD_ARG("col"), DIM_ARG("height"), DIM_ARG("width"))),
]

ACTIONS: list = ZERO_ARG + ONE_ARG + TWO_ARG + THREE_ARG + FOUR_ARG
ACTION_BY_NAME = {a.name: i for i, a in enumerate(ACTIONS)}
MAX_ARITY = max(a.arity for a in ACTIONS)
RAW_ARG_RANGE = 30  # matches ARC's max grid dimension; also covers colors/factors with room to spare

MAX_GRID_DIM = 30  # ARC-AGI's own max grid dimension (ADR-0002), reused here as the upscale bound


def _grid_shape(grid: Grid) -> tuple:
    return (len(grid), len(grid[0]) if grid else 0)


def execute(primitive_index: int, raw_args: tuple, grid: Grid) -> tuple:
    """Execute one action against `grid`.

    Returns `(new_grid, decoded_args, valid)`. `decoded_args` is a dict of
    the actual (post-`decode`) argument values, present even when `valid` is
    False, for logging. On invalid input (bad primitive index, out-of-bounds
    coordinate, a transform that would exceed the 30x30 canvas or collapse a
    grid dimension to zero), `new_grid` is `grid` unchanged and `valid` is
    False - the env applies the no-op-with-penalty behavior (Q7).
    """

    if not (0 <= primitive_index < len(ACTIONS)):
        return grid, {}, False

    action = ACTIONS[primitive_index]
    decoded = {
        spec.name: spec.decode(raw)
        for spec, raw in zip(action.args, raw_args)
    }

    h, w = _grid_shape(grid)

    for spec, value in zip(action.args, decoded.values()):
        if spec.kind == "coord":
            axis_len = h if spec.name == "row" else w
            if not (0 <= value < axis_len):
                return grid, decoded, False
        elif spec.kind == "factor":
            if action.name in ("hupscale", "upscale") and w * value > MAX_GRID_DIM:
                return grid, decoded, False
            if action.name in ("vupscale", "upscale") and h * value > MAX_GRID_DIM:
                return grid, decoded, False

    if action.name == "commit":
        if decoded["row"] + decoded["height"] > h or decoded["col"] + decoded["width"] > w:
            return grid, decoded, False

    try:
        new_grid = action.fn(grid, *decoded.values())
    except Exception:
        # Q7: invalid/out-of-bounds actions are a no-op with a penalty, not a
        # crash - guards against any edge case in a DSL primitive we didn't
        # anticipate (e.g. a degenerate grid shape) on top of the explicit
        # bounds checks above.
        return grid, decoded, False

    new_h, new_w = _grid_shape(new_grid)
    if new_h == 0 or new_w == 0:
        return grid, decoded, False

    return new_grid, decoded, True
