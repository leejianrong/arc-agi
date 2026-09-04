"""Unit tests (SLICES.md V1): each curated DSL primitive wrapped as an
action produces the same grid as calling the vendored `arc-dsl` function
directly, for representative inputs."""

import pytest

from arc_env import actions
from arc_env._dsl import dsl

GRIDS = [
    ((1, 2, 3), (4, 5, 6), (7, 8, 9)),
    ((0, 0), (0, 1)),
    ((3,),),
    ((2, 2, 2, 2),),
]

# ADR-0010 Phase 1's self-concatenation actions are derived (composed from
# more than one `dsl` call, like `fill_cell`/`canvas`/`commit`), not a 1:1
# `dsl.<name>` call - excluded from the generic 1:1 check below, covered by
# their own dedicated tests instead.
DERIVED_ZERO_ARG_NAMES = {
    "hconcat_self", "hconcat_self_vmirror", "vconcat_self_hmirror_top", "vconcat_self_hmirror_bottom",
}
DIRECT_ZERO_ARG = [a for a in actions.ZERO_ARG if a.name not in DERIVED_ZERO_ARG_NAMES]


@pytest.mark.parametrize("action", DIRECT_ZERO_ARG, ids=lambda a: a.name)
@pytest.mark.parametrize("grid", GRIDS)
def test_zero_arg_action_matches_direct_dsl_call(action, grid):
    dsl_fn = getattr(dsl, action.name)
    try:
        expected = dsl_fn(grid)
    except Exception:  # noqa: BLE001 - any DSL error means "not applicable here", skip
        pytest.skip(f"{action.name} not well-defined for this grid shape")
    assert action.fn(grid) == expected


@pytest.mark.parametrize("grid", GRIDS)
def test_hconcat_self_matches_dsl_hconcat_with_itself(grid):
    action = actions.ACTIONS[actions.ACTION_BY_NAME["hconcat_self"]]
    assert action.fn(grid) == dsl.hconcat(grid, grid)


@pytest.mark.parametrize("grid", GRIDS)
def test_hconcat_self_vmirror_matches_dsl_hconcat_with_vmirror(grid):
    action = actions.ACTIONS[actions.ACTION_BY_NAME["hconcat_self_vmirror"]]
    assert action.fn(grid) == dsl.hconcat(grid, dsl.vmirror(grid))


@pytest.mark.parametrize("grid", GRIDS)
def test_vconcat_self_hmirror_top_matches_dsl_vconcat_with_hmirror_first(grid):
    action = actions.ACTIONS[actions.ACTION_BY_NAME["vconcat_self_hmirror_top"]]
    assert action.fn(grid) == dsl.vconcat(dsl.hmirror(grid), grid)


@pytest.mark.parametrize("grid", GRIDS)
def test_vconcat_self_hmirror_bottom_matches_dsl_vconcat_with_hmirror_second(grid):
    action = actions.ACTIONS[actions.ACTION_BY_NAME["vconcat_self_hmirror_bottom"]]
    assert action.fn(grid) == dsl.vconcat(grid, dsl.hmirror(grid))


@pytest.mark.parametrize("action", actions.ONE_ARG, ids=lambda a: a.name)
def test_one_arg_action_matches_direct_dsl_call(action):
    grid = ((1, 2, 3, 4), (5, 6, 7, 8))
    dsl_fn = getattr(dsl, action.name)
    for factor in (2, 3):
        assert action.fn(grid, factor) == dsl_fn(grid, factor)


@pytest.mark.parametrize("action", actions.TWO_ARG, ids=lambda a: a.name)
def test_two_arg_action_matches_direct_dsl_call(action):
    grid = ((1, 2, 3), (4, 1, 6))
    dsl_fn = getattr(dsl, action.name)
    assert action.fn(grid, 1, 9) == dsl_fn(grid, 1, 9)


def test_fill_cell_paints_exactly_one_cell():
    grid = ((0, 0), (0, 0))
    result = actions.ACTIONS[actions.ACTION_BY_NAME["fill_cell"]].fn(grid, 5, 1, 0)
    assert result == ((0, 0), (5, 0))


def test_canvas_replaces_the_grid_with_a_fresh_uniformly_colored_one():
    grid = ((1, 2, 3), (4, 5, 6))
    result = actions.ACTIONS[actions.ACTION_BY_NAME["canvas"]].fn(grid, 7, 2, 4)
    assert result == ((7, 7, 7, 7), (7, 7, 7, 7))


# SLICES.md V3 unit test: commit/crop produces the expected sub-grid for a
# range of hand-constructed painted regions (corners, full canvas, single
# cell).
GRID_4X4 = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
)


@pytest.mark.parametrize(
    "row,col,height,width,expected",
    [
        (0, 0, 2, 2, ((1, 2), (5, 6))),  # top-left corner
        (0, 2, 2, 2, ((3, 4), (7, 8))),  # top-right corner
        (2, 0, 2, 2, ((9, 10), (13, 14))),  # bottom-left corner
        (2, 2, 2, 2, ((11, 12), (15, 16))),  # bottom-right corner
        (0, 0, 4, 4, GRID_4X4),  # full canvas
        (1, 1, 1, 1, ((6,),)),  # single cell
    ],
)
def test_commit_crops_to_the_expected_sub_grid(row, col, height, width, expected):
    result = actions.ACTIONS[actions.ACTION_BY_NAME["commit"]].fn(GRID_4X4, row, col, height, width)
    assert result == expected


@pytest.mark.parametrize(
    "action_name,primitive_index_offset",
    [(a.name, i) for i, a in enumerate(actions.ACTIONS)],
)
def test_action_registry_index_is_consistent(action_name, primitive_index_offset):
    assert actions.ACTIONS[actions.ACTION_BY_NAME[action_name]].name == action_name


# ADR-0011: object selection - `select_largest`/`select_smallest` (kind
# "select") and `commit_selection` (kind "act_on_selection").
OBJECTS_GRID = (
    (0, 0, 0, 0),
    (0, 2, 0, 3),
    (0, 2, 0, 0),
    (0, 0, 0, 0),
)  # two 4-connected, single-colored objects (background 0): a 2-cell "2" and a 1-cell "3"


def test_select_largest_picks_the_biggest_object_by_size():
    selected = actions.ACTIONS[actions.ACTION_BY_NAME["select_largest"]].fn(OBJECTS_GRID)
    assert selected == frozenset({(1, 1), (2, 1)})  # the 2-cell "2" object


def test_select_smallest_picks_the_smallest_object_by_size():
    selected = actions.ACTIONS[actions.ACTION_BY_NAME["select_smallest"]].fn(OBJECTS_GRID)
    assert selected == frozenset({(1, 3)})  # the 1-cell "3" object


def test_select_returns_empty_for_a_grid_with_no_objects():
    blank = ((0, 0), (0, 0))
    assert actions.ACTIONS[actions.ACTION_BY_NAME["select_largest"]].fn(blank) == frozenset()


def test_commit_selection_crops_to_the_selected_patchs_bounding_box():
    selected = frozenset({(1, 1), (2, 1)})
    result = actions.ACTIONS[actions.ACTION_BY_NAME["commit_selection"]].fn(OBJECTS_GRID, selected)
    assert result == ((2,), (2,))


class TestExecuteSelectionThreading:
    """`execute`'s `selected` parameter/return (ADR-0011), exercised through
    the full raw-args interface rather than `Action.fn` directly."""

    def test_select_action_updates_selection_without_touching_the_grid(self):
        idx = actions.ACTION_BY_NAME["select_largest"]
        new_grid, new_selected, decoded, valid = actions.execute(idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID, None)
        assert valid
        assert new_grid == OBJECTS_GRID
        assert new_selected == frozenset({(1, 1), (2, 1)})
        assert decoded == {}

    def test_select_action_is_invalid_when_no_objects_found(self):
        blank = ((0, 0), (0, 0))
        idx = actions.ACTION_BY_NAME["select_largest"]
        new_grid, new_selected, _decoded, valid = actions.execute(idx, (0,) * actions.MAX_ARITY, blank, None)
        assert not valid
        assert new_grid == blank
        assert new_selected is None  # unchanged - there was nothing selected before either

    def test_act_on_selection_is_invalid_with_no_current_selection(self):
        idx = actions.ACTION_BY_NAME["commit_selection"]
        new_grid, new_selected, _decoded, valid = actions.execute(idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID, None)
        assert not valid
        assert new_grid == OBJECTS_GRID
        assert new_selected is None

    def test_act_on_selection_consumes_a_prior_selection(self):
        select_idx = actions.ACTION_BY_NAME["select_smallest"]
        commit_idx = actions.ACTION_BY_NAME["commit_selection"]
        grid, selected, _, valid = actions.execute(select_idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID, None)
        assert valid
        grid, selected, _, valid = actions.execute(commit_idx, (0,) * actions.MAX_ARITY, grid, selected)
        assert valid
        assert grid == ((3,),)

    def test_a_successful_ordinary_transform_clears_a_stale_selection(self):
        select_idx = actions.ACTION_BY_NAME["select_largest"]
        vmirror_idx = actions.ACTION_BY_NAME["vmirror"]
        _, selected, _, valid = actions.execute(select_idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID, None)
        assert valid and selected

        _, selected_after, _, valid = actions.execute(vmirror_idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID, selected)
        assert valid
        assert selected_after is None

    def test_a_failed_ordinary_transform_leaves_the_selection_untouched(self):
        select_idx = actions.ACTION_BY_NAME["select_largest"]
        fill_cell_idx = actions.ACTION_BY_NAME["fill_cell"]
        _, selected, _, valid = actions.execute(select_idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID, None)
        assert valid and selected

        # fill_cell's row/col args decode to out-of-bounds coordinates for this grid.
        out_of_bounds_raw = (0, 29, 29, 0)
        _, selected_after, _, valid = actions.execute(fill_cell_idx, out_of_bounds_raw, OBJECTS_GRID, selected)
        assert not valid
        assert selected_after == selected


# ADR-0012: rest of the object-selection menu.
UNIQUE_COLOR_GRID = (
    (0, 0, 0, 0, 0),
    (0, 2, 0, 2, 0),  # two separate single-cell "2" objects - color 2 is NOT unique
    (0, 0, 0, 0, 0),
    (0, 0, 3, 0, 0),  # one "3" object - color 3 IS unique
    (0, 0, 0, 0, 0),
)


def test_select_by_color_picks_the_object_of_the_given_color():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_by_color"]]
    assert action.fn(OBJECTS_GRID, 2) == frozenset({(1, 1), (2, 1)})
    assert action.fn(OBJECTS_GRID, 3) == frozenset({(1, 3)})


def test_select_by_color_returns_empty_when_no_object_has_that_color():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_by_color"]]
    assert action.fn(OBJECTS_GRID, 5) == frozenset()


def test_select_unique_color_picks_only_the_singly_occurring_color():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_unique_color"]]
    assert action.fn(UNIQUE_COLOR_GRID) == frozenset({(3, 2)})


def test_delete_selected_covers_the_selection_with_the_background_color():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["delete_selected"]]
    result = action.fn(OBJECTS_GRID, frozenset({(1, 1), (2, 1)}))
    assert result == ((0, 0, 0, 0), (0, 0, 0, 3), (0, 0, 0, 0), (0, 0, 0, 0))


def test_recolor_selected_fills_the_selection_with_the_given_color():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["recolor_selected"]]
    result = action.fn(OBJECTS_GRID, frozenset({(1, 3)}), 5)
    assert result == ((0, 0, 0, 0), (0, 2, 0, 5), (0, 2, 0, 0), (0, 0, 0, 0))


def test_move_selected_shifts_the_selection_in_the_given_direction():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["move_selected"]]
    result = action.fn(OBJECTS_GRID, frozenset({(1, 3)}), 0)  # 0 = DOWN
    assert result == ((0, 0, 0, 0), (0, 2, 0, 0), (0, 2, 0, 3), (0, 0, 0, 0))


def test_paint_selected_at_stamps_the_selection_without_removing_the_original():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["paint_selected_at"]]
    result = action.fn(OBJECTS_GRID, frozenset({(1, 3)}), 3, 0)
    assert result == ((0, 0, 0, 0), (0, 2, 0, 3), (0, 2, 0, 0), (3, 0, 0, 0))


def test_execute_threads_decoded_args_into_a_select_action():
    idx = actions.ACTION_BY_NAME["select_by_color"]
    raw_args = (2,) + (0,) * (actions.MAX_ARITY - 1)
    _, selected, decoded, valid = actions.execute(idx, raw_args, OBJECTS_GRID, None)
    assert valid
    assert selected == frozenset({(1, 1), (2, 1)})
    assert decoded == {"color": 2}
