"""The typed action vocabulary, generalized past the POC's 5 set-op verbs to
object selection-by-attribute, object move/recolor/crop, and canvas/transform —
enough that the grammar spans task families (SLICES.md V5).

Every verb is an `Action` (see `grammar.py`) declaring its parameter slots and
its symbolic read/write/clear footprint over `ObjState`'s named slots. The `fn`
executes over the vendored arc-dsl, resolving derived colors against live state.
None = precondition unmet = no-op (Q7).
"""

from dataclasses import replace

from arc_env._dsl import constants, dsl
from object_env import objects
from object_env.colors import COLOR_DOMAIN, resolve
from object_env.grammar import Action, Param
from object_env.state import ObjState
from object_env.types import ArgType

_AB = ("a", "b")
# move directions (0-3) and set-ops (0-3, in `_combine`): DOWN/UP/LEFT/RIGHT;
# intersect/union/symdiff/difference.
_DIRECTIONS = (constants.DOWN, constants.UP, constants.LEFT, constants.RIGHT)

# reusable parameter specs
_COLOR = lambda name: Param(name, ArgType.COLOR, tuple(COLOR_DOMAIN))
_AXIS = Param("axis", ArgType.AXIS, (0, 1))
_SLOT = Param("slot", ArgType.AXIS, (0, 1), structural=True)  # a/b write selector
_REGION = Param("region", ArgType.AXIS, (0, 1), structural=True)  # a/b read selector
_OP = Param("op", ArgType.SETOP, (0, 1, 2, 3))
_DIR = Param("direction", ArgType.DIRECTION, (0, 1, 2, 3))
_SIZE = lambda name: Param(name, ArgType.SIZE, (1, 2, 3, 4, 5))


# ---------------- set-op family (POC's 5, reimplemented clean) ----------------
def _split(state: ObjState, axis: int):
    g = state.grid
    if not g or not g[0]:
        return None
    ra, rb = (dsl.tophalf(g), dsl.bottomhalf(g)) if axis == 0 else (dsl.lefthalf(g), dsl.righthalf(g))
    if (len(ra), len(ra[0])) != (len(rb), len(rb[0])):
        return None  # region-local index-sets only comparable when aligned
    return replace(state, region_a=ra, region_b=rb, set_a=None, set_b=None, obj=None)


def _select_color(state: ObjState, slot: int, region: int, color):
    reg = state.region_a if region == 0 else state.region_b
    if reg is None:
        return None
    idx = dsl.ofcolor(reg, resolve(color, replace(state, grid=reg)))
    return replace(state, **{f"set_{_AB[slot]}": idx})


def _combine(state: ObjState, op: int):
    a, b = state.set_a, state.set_b
    if a is None or b is None:
        return None
    result = (a & b, a | b, a ^ b, a - b)[op]
    return replace(state, set_a=frozenset(result), set_b=None)


def _paint_canvas(state: ObjState, bg, fill):
    if state.region_a is None or state.set_a is None:
        return None
    dims = (len(state.region_a), len(state.region_a[0]))
    grid = dsl.fill(dsl.canvas(resolve(bg, state), dims), resolve(fill, state), state.set_a)
    return replace(state, grid=grid)


def _paint_onto_region(state: ObjState, fill):
    if state.region_a is None or state.set_a is None:
        return None
    return replace(state, grid=dsl.fill(state.region_a, resolve(fill, state), state.set_a))


# ---------------- object family (select-by-attribute, then act) ----------------
def _selector(pick):
    def fn(state: ObjState, **args):
        obj = pick(state.grid, **args)
        return replace(state, obj=obj) if obj is not None else None

    return fn


def _move_object(state: ObjState, direction: int):
    if state.obj is None:
        return None
    colored = dsl.toobject(state.obj.cells, state.grid)
    grid = dsl.move(state.grid, colored, _DIRECTIONS[direction])
    return replace(state, grid=grid, obj=None)


def _recolor_object(state: ObjState, color):
    if state.obj is None:
        return None
    return replace(state, grid=dsl.fill(state.grid, resolve(color, state), state.obj.cells), obj=None)


def _crop_to_object(state: ObjState):
    if state.obj is None:
        return None
    grid = dsl.subgrid(state.obj.cells, state.grid)
    return replace(state, grid=grid, obj=None, region_a=None, region_b=None, set_a=None, set_b=None)


# ---------------- grid transforms (no object needed) ----------------
def _trim(state: ObjState):
    g = state.grid
    if len(g) < 3 or len(g[0]) < 3:
        return None
    return replace(state, grid=dsl.trim(g))


def _replace_color(state: ObjState, from_color, to_color):
    return replace(state, grid=dsl.replace(state.grid, resolve(from_color, state), resolve(to_color, state)))


def _canvas_mostcolor(state: ObjState, height: int, width: int):
    grid = dsl.canvas(dsl.mostcolor(state.grid), (height, width))
    return replace(state, grid=grid, region_a=None, region_b=None, set_a=None, set_b=None, obj=None)


# convenience footprint helpers
def _to_grid(_a):
    return ("grid",)


def _needs(*slots):
    return lambda _a: slots


ACTIONS = [
    # --- set-op ---
    Action("split", (_AXIS,), _split,
           reads=_needs("grid"), writes=_needs("region_a", "region_b"),
           clears=_needs("set_a", "set_b", "obj")),
    Action("select_color", (_SLOT, _REGION, _COLOR("color")), _select_color,
           reads=lambda a: (f"region_{_AB[a['region']]}",),
           writes=lambda a: (f"set_{_AB[a['slot']]}",)),
    Action("combine", (_OP,), _combine,
           reads=_needs("set_a", "set_b"), writes=_needs("set_a"), clears=_needs("set_b")),
    Action("paint_canvas", (_COLOR("bg"), _COLOR("fill")), _paint_canvas,
           reads=_needs("region_a", "set_a"), writes=_to_grid),
    Action("paint_onto_region", (_COLOR("fill"),), _paint_onto_region,
           reads=_needs("region_a", "set_a"), writes=_to_grid),
    # --- object selection-by-attribute ---
    Action("select_largest", (), _selector(lambda g: objects.select_largest(g)),
           reads=_needs("grid"), writes=_needs("obj")),
    Action("select_smallest", (), _selector(lambda g: objects.select_smallest(g)),
           reads=_needs("grid"), writes=_needs("obj")),
    Action("select_largest_no_diag", (), _selector(lambda g: objects.select_largest_no_diag(g)),
           reads=_needs("grid"), writes=_needs("obj")),
    Action("select_tallest", (), _selector(lambda g: objects.select_tallest(g)),
           reads=_needs("grid"), writes=_needs("obj")),
    Action("select_by_color", (_COLOR("color"),),
           _selector(lambda g, color: objects.select_by_color(g, resolve(color, ObjState(grid=g)))),
           reads=_needs("grid"), writes=_needs("obj")),
    # --- object ops ---
    Action("move_object", (_DIR,), _move_object,
           reads=_needs("obj"), writes=_to_grid,
           clears=_needs("obj", "region_a", "region_b", "set_a", "set_b")),
    Action("recolor_object", (_COLOR("color"),), _recolor_object,
           reads=_needs("obj"), writes=_to_grid, clears=_needs("obj")),
    Action("crop_to_object", (), _crop_to_object,
           reads=_needs("obj"), writes=_to_grid,
           clears=_needs("obj", "region_a", "region_b", "set_a", "set_b")),
    # --- grid transforms ---
    Action("trim", (), _trim, reads=_needs("grid"), writes=_to_grid),
    Action("replace_color", (_COLOR("from_color"), _COLOR("to_color")), _replace_color,
           reads=_needs("grid"), writes=_to_grid),
    Action("canvas_mostcolor", (_SIZE("height"), _SIZE("width")), _canvas_mostcolor,
           reads=_needs("grid"), writes=_to_grid,
           clears=_needs("region_a", "region_b", "set_a", "set_b", "obj")),
]

ACTION_BY_NAME = {a.name: a for a in ACTIONS}
