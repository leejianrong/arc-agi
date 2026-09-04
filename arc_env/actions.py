"""The curated discrete action space over `arc-dsl` primitives.

Per ADR-0001, actions call `arc-dsl` primitives directly - no separate AST or
executor layer of our own. Higher-order primitives (`compose`, `chain`,
`fork`, `rbind`, `lbind`, `power`) are excluded for good: they build
closures rather than transforming a grid, which a flat "pick one primitive
per step" action space can't represent (see `docs/adr/0001*.md`).

Beyond that exclusion, this module is further restricted to primitives whose
signature is `Grid [, scalar args] -> Grid` - i.e. primitives that operate
directly on the single grid the agent is editing, with typed scalar
arguments (a color, a scale factor, a coordinate, a dimension) rather than a
`Callable` argument that would require other, unpicked primitives to
construct. This is what makes the action space "directly steppable": every
action maps one grid to the next. ADR-0011 (ADR-0010 Phase 2 Slice 1) and
ADR-0012 (Slices A/B, the rest of that menu) are the one deliberate
carve-out from "no `Object`/`Indices` argument": a small set of
`"select"`/`"act_on_selection"` actions thread one extra piece of state (the
currently selected patch, an `Indices`) alongside the grid - never a
`Callable`. Unlike ADR-0011's original two selectors, ADR-0012's
`select_by_color` *does* take a scalar `color` argument - the criterion
itself (e.g. "argmax by size", "cells of color K") is still always fixed
internally per action, never an agent-chosen `Callable`. See ADR-0011,
ADR-0012, and `execute`'s docstring below for the mechanism.

This scalar-args-only restriction (plus the selection carve-out) was
derived by checking, for every ARC-AGI-1 training task with a known-correct
`arc-dsl` solver, whether that solver only calls primitives from this
module's action groups below. 29 tasks qualify (13 same-shape, 16
variable-shape: see `arc_env/task_loader.py`'s module docstring for the
full per-ADR breakdown) - `arc_env/task_loader.py`:CURATED_TASK_IDS is the
curated task subset and also the regression-test fixture set
(`tests/test_dsl_regression.py`).

ADR-0010 Phase 1 (2026-08-29) added `hconcat_self`, `hconcat_self_vmirror`,
`vconcat_self_hmirror_top`, and `vconcat_self_hmirror_bottom` - a repo-audit
of `third_party/arc-dsl/solvers.py` found several solvers doubling the grid
by concatenating it with itself (optionally mirrored) via `dsl.hconcat`/
`dsl.vconcat`, which don't fit this module's "Grid [, scalar args] -> Grid"
signature directly (they're `Grid, Grid -> Grid`) but *do* fit it once the
second `Grid` argument is always a fixed function of the first - the same
"derived, not drawn from a solver 1:1" pattern `fill_cell` already
established. No new representational mechanism, per ADR-0010's Phase 1
scope: still zero-arg, still `Grid -> Grid`.

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

ADR-0012's `select_by_color`, `select_unique_color`, `delete_selected`, and
`paint_selected_at` are the same kind of not-drawn-from-a-solver-1:1
addition: a `solvers.py` audit for a literal (no `mapply`/`compose`/lambda)
single-selection fixture came back empty for these four specifically
(unlike `move_selected`/`recolor_selected`, which do have one - `25ff71a9`/
`ea32f347`), so they're verified by direct unit test against hand-
constructed grids (`tests/test_actions.py`) instead of a curated ARC task,
the same bar `fill_cell`/`canvas` were already held to above.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from arc_env._dsl import constants, dsl

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


def _decode_direction(raw: int) -> int:
    return raw % 4  # index into _DIRECTIONS below


COLOR_ARG = lambda name: ArgSpec(name, "color", _decode_color)
FACTOR_ARG = lambda name: ArgSpec(name, "factor", _decode_factor)
COORD_ARG = lambda name: ArgSpec(name, "coord", _decode_coord)
DIM_ARG = lambda name: ArgSpec(name, "dim", _decode_dim)
DIRECTION_ARG = lambda name: ArgSpec(name, "direction", _decode_direction)

# ADR-0012: a small fixed menu of cardinal directions for `move_selected`,
# mirroring `arc-dsl`'s own DOWN/UP/LEFT/RIGHT constants - kept as a curated
# discrete choice (like FACTOR_ARG's {2,3,4}) rather than a raw signed
# offset, since the audit fixture (`25ff71a9`) only ever needs one of these
# four and an unconstrained signed offset would blow up the raw-arg range.
_DIRECTIONS = (constants.DOWN, constants.UP, constants.LEFT, constants.RIGHT)


@dataclass(frozen=True)
class Action:
    """One curated action: a name (always the underlying `arc-dsl` primitive
    name, except for the derived `fill_cell`), the callable, and its typed
    argument slots (empty for zero-arg actions).

    `kind` (ADR-0011): `"transform"` (default - today's `Grid [, args] ->
    Grid`, unaffected by the selection mechanism) - `fn(grid, *decoded_args)
    -> Grid`; `"select"` - `fn(grid) -> Indices`, updates the current
    selection without touching the grid, invalid (no-op) if it finds
    nothing to select; `"act_on_selection"` - `fn(grid, selected,
    *decoded_args) -> Grid`, invalid if there is no current selection.
    """

    name: str
    fn: Callable[..., Grid]
    args: tuple = field(default_factory=tuple)
    kind: str = "transform"

    @property
    def arity(self) -> int:
        return len(self.args)


def _fill_cell(grid: Grid, color: int, row: int, col: int) -> Grid:
    return dsl.fill(grid, color, frozenset({(row, col)}))


def _canvas(grid: Grid, value: int, height: int, width: int) -> Grid:
    return dsl.canvas(value, (height, width))  # replaces `grid` outright; ignores it


def _commit(grid: Grid, row: int, col: int, height: int, width: int) -> Grid:
    return dsl.crop(grid, (row, col), (height, width))


# ADR-0010 Phase 1: self-concatenation (optionally mirrored) - see module
# docstring. Each fixes `dsl.hconcat`/`dsl.vconcat`'s second `Grid` argument
# as a function of the first, so the action stays zero-arg `Grid -> Grid`.
def _hconcat_self(grid: Grid) -> Grid:
    return dsl.hconcat(grid, grid)


def _hconcat_self_vmirror(grid: Grid) -> Grid:
    return dsl.hconcat(grid, dsl.vmirror(grid))


def _vconcat_self_hmirror_top(grid: Grid) -> Grid:
    return dsl.vconcat(dsl.hmirror(grid), grid)


def _vconcat_self_hmirror_bottom(grid: Grid) -> Grid:
    return dsl.vconcat(grid, dsl.hmirror(grid))


# ADR-0011 Phase 2 Slice 1: object selection. `_OBJECTS` fixes `dsl.objects`'s
# (univalued, diagonal, without_bg) triple to the one variant this pass
# curates (see that ADR's module-level rationale for why only one variant is
# landed now) - `select_largest`/`select_smallest` pick one object out of
# that segmentation by `dsl.size`, mirroring the dominant `argmax(_, size)`/
# `argmin(_, size)` pattern the ADR's solver audit found. Both are "select"
# actions: `fn(grid) -> Indices`, invalid (no objects found) rather than
# `Grid -> Grid`.
def _objects(grid: Grid):
    return dsl.objects(grid, True, True, True)


def _select_largest(grid: Grid):
    objs = _objects(grid)
    return dsl.toindices(dsl.argmax(objs, dsl.size)) if objs else frozenset()


def _select_smallest(grid: Grid):
    objs = _objects(grid)
    return dsl.toindices(dsl.argmin(objs, dsl.size)) if objs else frozenset()


def _commit_selection(grid: Grid, selected) -> Grid:
    return dsl.subgrid(selected, grid)


# ADR-0012: the rest of ADR-0011's deferred menu. Unlike Slice 1's selectors,
# `select_by_color` takes a `color` argument - `execute()`'s `"select"`
# branch now passes decoded args through, same as every other action kind
# already does.
def _select_by_color(grid: Grid, color: int):
    matches = dsl.colorfilter(_objects(grid), color)
    return dsl.toindices(dsl.merge(matches)) if matches else frozenset()


def _select_unique_color(grid: Grid):
    objs = _objects(grid)
    counts = Counter(dsl.color(obj) for obj in objs)
    unique_objs = frozenset(obj for obj in objs if counts[dsl.color(obj)] == 1)
    return dsl.toindices(dsl.merge(unique_objs)) if unique_objs else frozenset()


def _delete_selected(grid: Grid, selected) -> Grid:
    return dsl.cover(grid, selected)


def _recolor_selected(grid: Grid, selected, color: int) -> Grid:
    return dsl.fill(grid, color, selected)


def _move_selected(grid: Grid, selected, direction_index: int) -> Grid:
    obj = dsl.toobject(selected, grid)
    return dsl.move(grid, obj, _DIRECTIONS[direction_index])


def _paint_selected_at(grid: Grid, selected, row: int, col: int) -> Grid:
    obj = dsl.toobject(selected, grid)
    ul_row, ul_col = dsl.ulcorner(selected)
    return dsl.paint(grid, dsl.shift(obj, (row - ul_row, col - ul_col)))


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
    # ADR-0010 Phase 1: self-concatenation, optionally mirrored.
    Action("hconcat_self", _hconcat_self),
    Action("hconcat_self_vmirror", _hconcat_self_vmirror),
    Action("vconcat_self_hmirror_top", _vconcat_self_hmirror_top),
    Action("vconcat_self_hmirror_bottom", _vconcat_self_hmirror_bottom),
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

# ADR-0011 Phase 2 Slice 1: object selection (see module docstring addendum
# and that ADR for the full design). `select_*` actions update the selection
# without touching the grid; `commit_selection` mirrors `commit`'s
# crop-and-end-episode semantics, using the selected patch's bounding box
# instead of 4 literal coordinate/dimension args.
SELECT = [
    Action("select_largest", _select_largest, kind="select"),
    Action("select_smallest", _select_smallest, kind="select"),
    # ADR-0012: rest of the deferred selection-criteria menu. No literal
    # single-selection fixture solver was found for either in the
    # `solvers.py` audit (see that ADR's Consequences) - verified by direct
    # unit test against hand-constructed grids instead, the same bar
    # `fill_cell`/`canvas` (Phase 1) were held to.
    Action("select_by_color", _select_by_color, (COLOR_ARG("color"),), kind="select"),
    Action("select_unique_color", _select_unique_color, kind="select"),
]
ACT_ON_SELECTION = [
    Action("commit_selection", _commit_selection, kind="act_on_selection"),
    # ADR-0012: `recolor_selected` and `move_selected` each have a verified
    # curated fixture task (`ea32f347`, `25ff71a9`); `delete_selected` and
    # `paint_selected_at` don't (same audit-negative-result caveat as
    # `select_by_color`/`select_unique_color` above).
    Action("delete_selected", _delete_selected, kind="act_on_selection"),
    Action("recolor_selected", _recolor_selected, (COLOR_ARG("color"),), kind="act_on_selection"),
    Action("move_selected", _move_selected, (DIRECTION_ARG("direction"),), kind="act_on_selection"),
    Action(
        "paint_selected_at",
        _paint_selected_at,
        (COORD_ARG("row"), COORD_ARG("col")),
        kind="act_on_selection",
    ),
]

ACTIONS: list = ZERO_ARG + ONE_ARG + TWO_ARG + THREE_ARG + FOUR_ARG + SELECT + ACT_ON_SELECTION
ACTION_BY_NAME = {a.name: i for i, a in enumerate(ACTIONS)}
MAX_ARITY = max(a.arity for a in ACTIONS)
RAW_ARG_RANGE = 30  # matches ARC's max grid dimension; also covers colors/factors with room to spare

MAX_GRID_DIM = 30  # ARC-AGI's own max grid dimension (ADR-0002), reused here as the upscale bound


def _grid_shape(grid: Grid) -> tuple:
    return (len(grid), len(grid[0]) if grid else 0)


def execute(primitive_index: int, raw_args: tuple, grid: Grid, selected=None) -> tuple:
    """Execute one action against `grid` (and, per ADR-0011, the current
    object `selected`, if any).

    A `"select"` action's `fn` now also receives any decoded args (ADR-0012:
    `select_by_color`'s `color`), same as `"transform"`/`"act_on_selection"`
    already did - a no-op for ADR-0011's original zero-arg selectors.

    Returns `(new_grid, new_selected, decoded_args, valid)`. `decoded_args`
    is a dict of the actual (post-`decode`) argument values, present even
    when `valid` is False, for logging. On invalid input (bad primitive
    index, out-of-bounds coordinate, a transform that would exceed the
    30x30 canvas or collapse a grid dimension to zero, a `"select"` action
    finding nothing to select, or an `"act_on_selection"` action with no
    current selection), `new_grid`/`new_selected` are `grid`/`selected`
    unchanged and `valid` is False - the env applies the no-op-with-penalty
    behavior (Q7).
    """

    if not (0 <= primitive_index < len(ACTIONS)):
        return grid, selected, {}, False

    action = ACTIONS[primitive_index]
    decoded = {
        spec.name: spec.decode(raw)
        for spec, raw in zip(action.args, raw_args)
    }

    if action.kind == "select":
        new_selected = action.fn(grid, *decoded.values())
        if not new_selected:
            return grid, selected, decoded, False
        return grid, new_selected, decoded, True

    if action.kind == "act_on_selection":
        if not selected:
            return grid, selected, decoded, False
        try:
            new_grid = action.fn(grid, selected, *decoded.values())
        except Exception:  # noqa: BLE001
            return grid, selected, decoded, False
        new_h, new_w = _grid_shape(new_grid)
        if new_h == 0 or new_w == 0 or new_h > MAX_GRID_DIM or new_w > MAX_GRID_DIM:
            return grid, selected, decoded, False
        return new_grid, selected, decoded, True

    h, w = _grid_shape(grid)

    for spec, value in zip(action.args, decoded.values()):
        if spec.kind == "coord":
            axis_len = h if spec.name == "row" else w
            if not (0 <= value < axis_len):
                return grid, selected, decoded, False
        elif spec.kind == "factor":
            if action.name in ("hupscale", "upscale") and w * value > MAX_GRID_DIM:
                return grid, selected, decoded, False
            if action.name in ("vupscale", "upscale") and h * value > MAX_GRID_DIM:
                return grid, selected, decoded, False

    if action.name == "commit" and (
        decoded["row"] + decoded["height"] > h or decoded["col"] + decoded["width"] > w
    ):
        return grid, selected, decoded, False

    try:
        new_grid = action.fn(grid, *decoded.values())
    except Exception:  # noqa: BLE001
        # Q7: invalid/out-of-bounds actions are a no-op with a penalty, not a
        # crash - guards against any edge case in a DSL primitive we didn't
        # anticipate (e.g. a degenerate grid shape) on top of the explicit
        # bounds checks above.
        return grid, selected, decoded, False

    new_h, new_w = _grid_shape(new_grid)
    if new_h == 0 or new_w == 0 or new_h > MAX_GRID_DIM or new_w > MAX_GRID_DIM:
        return grid, selected, decoded, False

    # ADR-0011: a successful ordinary transform invalidates any stale
    # selection (its indices may no longer correspond to meaningful cells
    # after a rotation/resize/etc.) - re-selecting is cheap, a silently
    # wrong stale selection surviving an unrelated edit is not.
    return new_grid, None, decoded, True
