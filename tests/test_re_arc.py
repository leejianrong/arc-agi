"""Unit tests for `arc_env.re_arc` - generating extra practice instances via
the vendored `re-arc` (SLICES.md V2 build plan step 3)."""

import pytest

from arc_env.re_arc import generate_pair
from arc_env.task_loader import CURATED_TASK_IDS


@pytest.mark.parametrize("task_id", sorted(CURATED_TASK_IDS))
def test_generate_pair_produces_a_valid_nondegenerate_grid_pair(task_id):
    pair = generate_pair(task_id)
    assert pair.input != pair.output
    for grid in (pair.input, pair.output):
        assert 1 <= len(grid) <= 30
        assert all(1 <= len(row) <= 30 for row in grid)
        assert len({len(row) for row in grid}) == 1
        assert all(0 <= v <= 9 for row in grid for v in row)


def test_generate_pair_rejects_unknown_task_id():
    with pytest.raises(ValueError):
        generate_pair("not-a-real-task")


def test_generate_pair_varies_across_calls():
    # Not a strict guarantee for every task, but true often enough with
    # random dims/colors that a fixed run of 20 calls should show variety.
    grids = {generate_pair("67a3c6ac").input for _ in range(20)}
    assert len(grids) > 1
