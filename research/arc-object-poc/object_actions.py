"""F15 POC — a minimal *object-centric* action space over the set-op family.

This is throwaway proof-of-concept code (ADR-0027, gated). It does NOT import
or modify `arc_env/actions.py`, `trainers/`, or `viz/`. It reuses only the
vendored `arc-dsl` primitives (via `arc_env._dsl`) as the executor underneath,
exactly the way the shipped action space does (ADR-0001 generalized, not
discarded).

The representation difference from the shipped single-mutable-grid model: the
state carries *named, first-class objects* — two region sub-grids and two
index-sets — that actions reference by slot (the `(verb, object-ref, arg)`
axis ADR-0027 describes), instead of one flat grid plus a bolted-on selection
mask. For the region-set-op task family this collapses to a tiny, general
vocabulary:

    split(axis) -> select_color(slot, region, color) x2
                -> combine(op) -> paint_canvas(bg, fill) | paint_onto_region(fill)

Mirrors the shipped executor's *surface* (`ACTIONS`, `RAW_ARG_RANGE`,
`MAX_ARITY`, `execute(index, raw_args, state) -> (new_state, decoded, valid)`)
so a GP genome speaking `(action_index, raw_args)` plugs straight in.
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arc_env._dsl import dsl  # noqa: E402  (path set up above)

Grid = tuple  # tuple[tuple[int, ...], ...]
Indices = frozenset  # frozenset[tuple[int, int]]

RAW_ARG_RANGE = 30  # matches the shipped action space's raw arg range


@dataclass(frozen=True)
class ObjState:
    """First-class object state: the working grid, two region sub-grids, and
    two named index-sets (region-local coordinates). `region_a`/`region_b` and
    `set_a`/`set_b` are the addressable "objects" actions reference by slot."""

    grid: Grid
    region_a: Grid | None = None
    region_b: Grid | None = None
    set_a: Indices | None = None
    set_b: Indices | None = None


# --- arg decoders: a raw int in [0, RAW_ARG_RANGE) -> a small typed value ---
def _axis(raw: int) -> int:
    return raw % 2  # 0 = top/bottom, 1 = left/right


def _slot(raw: int) -> int:
    return raw % 2  # 0 = a, 1 = b


def _region(raw: int) -> int:
    return raw % 2  # 0 = region_a, 1 = region_b


def _op(raw: int) -> int:
    return raw % 4  # 0 intersect, 1 union, 2 symdiff, 3 difference


def _color(raw: int) -> int:
    return raw % 10  # ARC colors 0..9


@dataclass(frozen=True)
class ArgSpec:
    name: str
    decode: object  # Callable[[int], int]


@dataclass(frozen=True)
class Action:
    name: str
    fn: object  # Callable[..., tuple[ObjState | None, ...]]
    args: tuple = ()

    @property
    def arity(self) -> int:
        return len(self.args)


AXIS = ArgSpec("axis", _axis)
SLOT = ArgSpec("slot", _slot)
REGION = ArgSpec("region", _region)
OP = ArgSpec("op", _op)
COLOR = lambda name: ArgSpec(name, _color)  # noqa: E731 (tiny factory, POC)


# --- the five action implementations: each returns ObjState | None (None = invalid) ---
def _split(state: ObjState, axis: int) -> ObjState | None:
    g = state.grid
    if not g or not g[0]:
        return None
    if axis == 0:
        ra, rb = dsl.tophalf(g), dsl.bottomhalf(g)
    else:
        ra, rb = dsl.lefthalf(g), dsl.righthalf(g)
    # region-local index-sets are only comparable when the two regions align
    if (len(ra), len(ra[0])) != (len(rb), len(rb[0])):
        return None
    return replace(state, region_a=ra, region_b=rb, set_a=None, set_b=None)


def _select_color(state: ObjState, slot: int, region: int, color: int) -> ObjState | None:
    reg = state.region_a if region == 0 else state.region_b
    if reg is None:
        return None
    idx = dsl.ofcolor(reg, color)
    return replace(state, **{"set_a" if slot == 0 else "set_b": idx})


def _combine(state: ObjState, op: int) -> ObjState | None:
    a, b = state.set_a, state.set_b
    if a is None or b is None:
        return None
    if op == 0:
        result = a & b
    elif op == 1:
        result = a | b
    elif op == 2:
        result = a ^ b
    else:
        result = a - b
    return replace(state, set_a=frozenset(result), set_b=None)


def _paint_canvas(state: ObjState, bg: int, fill: int) -> ObjState | None:
    if state.region_a is None or state.set_a is None:
        return None
    dims = (len(state.region_a), len(state.region_a[0]))
    grid = dsl.fill(dsl.canvas(bg, dims), fill, state.set_a)
    return replace(state, grid=grid)


def _paint_onto_region(state: ObjState, fill: int) -> ObjState | None:
    if state.region_a is None or state.set_a is None:
        return None
    grid = dsl.fill(state.region_a, fill, state.set_a)
    return replace(state, grid=grid)


ACTIONS = [
    Action("split", _split, (AXIS,)),
    Action("select_color", _select_color, (SLOT, REGION, COLOR("color"))),
    Action("combine", _combine, (OP,)),
    Action("paint_canvas", _paint_canvas, (COLOR("bg"), COLOR("fill"))),
    Action("paint_onto_region", _paint_onto_region, (COLOR("fill"),)),
]
ACTION_BY_NAME = {a.name: i for i, a in enumerate(ACTIONS)}
MAX_ARITY = max(a.arity for a in ACTIONS)


def execute(index: int, raw_args: tuple, state: ObjState) -> tuple:
    """Dispatch one action. Returns `(new_state, decoded, valid)`. On any
    invalidity (bad index, unmet precondition) returns the input state
    unchanged with `valid=False` — the same no-op convention the shipped
    `arc_env.actions.execute` uses (Q7)."""

    if not (0 <= index < len(ACTIONS)):
        return state, {}, False
    action = ACTIONS[index]
    decoded_vals = [spec.decode(int(raw_args[i])) for i, spec in enumerate(action.args)]
    decoded = {spec.name: v for spec, v in zip(action.args, decoded_vals)}
    new_state = action.fn(state, *decoded_vals)
    if new_state is None:
        return state, decoded, False
    return new_state, decoded, True
