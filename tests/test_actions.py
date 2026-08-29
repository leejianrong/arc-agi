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


@pytest.mark.parametrize("action", actions.ZERO_ARG, ids=lambda a: a.name)
@pytest.mark.parametrize("grid", GRIDS)
def test_zero_arg_action_matches_direct_dsl_call(action, grid):
    dsl_fn = getattr(dsl, action.name)
    try:
        expected = dsl_fn(grid)
    except Exception:  # noqa: BLE001 - any DSL error means "not applicable here", skip
        pytest.skip(f"{action.name} not well-defined for this grid shape")
    assert action.fn(grid) == expected


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
