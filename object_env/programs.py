"""The V5 fixture basket: ~16 curated tasks with hand-written object-space
programs, spanning families so the grammar's generalization past the 8 set-op
tasks is actually tested (fork 2, moderate basket).

Each program is a list of `(action_name, args)`. `build(task_id)` type-checks it
into a `Program` at load; `tests/test_object_grammar_regression.py` replays every
one against the task's real train+test pairs (the object-track analogue of
`tests/test_dsl_regression.py`).
"""

from object_env.actions import ACTION_BY_NAME
from object_env.grammar import Program, Step

# fmt: off
PROGRAMS: dict[str, list] = {
    # ---- set-op cluster (8) — POC's hand programs over the clean API ----
    # split top/bottom, blank(0) cells of each half, intersect, paint green(3)
    "6430c8c4": [("split", {"axis": 0}), ("select_color", {"slot": 0, "region": 0, "color": 0}),
                 ("select_color", {"slot": 1, "region": 1, "color": 0}), ("combine", {"op": 0}),
                 ("paint_canvas", {"bg": 0, "fill": 3})],
    "94f9d214": [("split", {"axis": 0}), ("select_color", {"slot": 0, "region": 0, "color": 0}),
                 ("select_color", {"slot": 1, "region": 1, "color": 0}), ("combine", {"op": 0}),
                 ("paint_canvas", {"bg": 0, "fill": 2})],
    # canvas prefilled green(3), intersection punched to 0
    "ce4f8723": [("split", {"axis": 0}), ("select_color", {"slot": 0, "region": 0, "color": 0}),
                 ("select_color", {"slot": 1, "region": 1, "color": 0}), ("combine", {"op": 0}),
                 ("paint_canvas", {"bg": 3, "fill": 0})],
    "f2829549": [("split", {"axis": 1}), ("select_color", {"slot": 0, "region": 0, "color": 0}),
                 ("select_color", {"slot": 1, "region": 1, "color": 0}), ("combine", {"op": 0}),
                 ("paint_canvas", {"bg": 0, "fill": 3})],
    "fafffa47": [("split", {"axis": 0}), ("select_color", {"slot": 0, "region": 0, "color": 0}),
                 ("select_color", {"slot": 1, "region": 1, "color": 0}), ("combine", {"op": 0}),
                 ("paint_canvas", {"bg": 0, "fill": 2})],
    # symmetric-difference of blanks
    "99b1bc43": [("split", {"axis": 0}), ("select_color", {"slot": 0, "region": 0, "color": 0}),
                 ("select_color", {"slot": 1, "region": 1, "color": 0}), ("combine", {"op": 2}),
                 ("paint_canvas", {"bg": 0, "fill": 3})],
    # color-2 cells, symmetric-difference, paint green
    "3428a4f5": [("split", {"axis": 0}), ("select_color", {"slot": 0, "region": 0, "color": 2}),
                 ("select_color", {"slot": 1, "region": 1, "color": 2}), ("combine", {"op": 2}),
                 ("paint_canvas", {"bg": 0, "fill": 3})],
    # left/right, union of color-4(left) and color-3(right), paint brown(6) onto left region
    "dae9d2b5": [("split", {"axis": 1}), ("select_color", {"slot": 0, "region": 0, "color": 4}),
                 ("select_color", {"slot": 1, "region": 1, "color": 3}), ("combine", {"op": 1}),
                 ("paint_onto_region", {"fill": 6})],

    # ---- move (2) ----
    "25ff71a9": [("select_largest", {}), ("move_object", {"direction": 0})],
    "a79310a0": [("select_largest_no_diag", {}), ("move_object", {"direction": 0}),
                 ("replace_color", {"from_color": 8, "to_color": 2})],

    # ---- recolor-by-attribute (1) ----
    "ea32f347": [("replace_color", {"from_color": 5, "to_color": 4}),
                 ("select_largest", {}), ("recolor_object", {"color": 1}),
                 ("select_smallest", {}), ("recolor_object", {"color": 2})],

    # ---- crop / extract (3) ----
    "1f85a75f": [("select_largest", {}), ("crop_to_object", {})],
    "23b5c85d": [("select_smallest", {}), ("crop_to_object", {})],
    "1c786137": [("select_tallest", {}), ("crop_to_object", {}), ("trim", {})],

    # ---- canvas, derived color (1) ----
    "5582e5ca": [("canvas_mostcolor", {"height": 3, "width": 3})],

    # ---- global transform (1) ----
    "b1948b0a": [("replace_color", {"from_color": 6, "to_color": 2})],
}
# fmt: on

TARGET_TASK_IDS = list(PROGRAMS.keys())
SET_OP_TASK_IDS = TARGET_TASK_IDS[:8]


def build(task_id: str) -> Program:
    """Type-check the recorded program for `task_id` into a `Program`."""
    steps = [Step(ACTION_BY_NAME[name], dict(args)) for name, args in PROGRAMS[task_id]]
    return Program(steps)
