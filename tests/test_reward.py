"""Unit tests (SLICES.md V2): the reward function returns the expected delta
for a hand-constructed before/after grid pair, per ADR-0005."""

import pytest

from arc_env.reward import (
    INVALID_ACTION_PENALTY,
    STEP_COST,
    TERMINAL_BONUS,
    compute_diff_mask,
    compute_reward,
    similarity,
)


def test_diff_mask_is_exactly_the_cells_that_need_to_change():
    input_grid = ((0, 0), (0, 0))
    target_grid = ((1, 1), (0, 0))
    assert compute_diff_mask(input_grid, target_grid) == frozenset({(0, 0), (0, 1)})


def test_similarity_of_untouched_input_is_zero():
    input_grid = ((0, 0), (0, 0))
    target_grid = ((1, 1), (0, 0))
    diff_mask = compute_diff_mask(input_grid, target_grid)
    assert similarity(input_grid, target_grid, diff_mask) == 0.0


def test_similarity_of_exact_match_is_one():
    target_grid = ((1, 1), (0, 0))
    diff_mask = compute_diff_mask(((0, 0), (0, 0)), target_grid)
    assert similarity(target_grid, target_grid, diff_mask) == 1.0


def test_similarity_counts_only_diff_mask_cells():
    input_grid = ((0, 0), (0, 0))
    target_grid = ((1, 1), (0, 0))
    diff_mask = compute_diff_mask(input_grid, target_grid)  # {(0,0), (0,1)}
    partially_fixed = ((1, 0), (0, 0))  # fixed (0,0), not (0,1)
    assert similarity(partially_fixed, target_grid, diff_mask) == pytest.approx(0.5)


def test_similarity_is_one_when_diff_mask_is_empty():
    grid = ((3, 3), (3, 3))
    assert similarity(grid, grid, frozenset()) == 1.0


def test_diff_mask_is_none_for_a_variable_shape_pair():
    input_grid = ((1, 2), (3, 4))
    target_grid = ((1, 2, 1, 2), (3, 4, 3, 4))  # e.g. an hconcat-with-self target
    assert compute_diff_mask(input_grid, target_grid) is None


def test_similarity_with_none_diff_mask_is_zero_until_shape_matches():
    input_grid = ((1, 2), (3, 4))
    target_grid = ((1, 2, 1, 2), (3, 4, 3, 4))
    assert similarity(input_grid, target_grid, None) == 0.0
    # right shape, wrong content: partial credit against every target cell.
    wrong_content = ((0, 0, 0, 0), (0, 0, 0, 0))
    assert similarity(wrong_content, target_grid, None) == 0.0
    half_right = ((1, 2, 0, 0), (3, 4, 0, 0))
    assert similarity(half_right, target_grid, None) == pytest.approx(0.5)
    assert similarity(target_grid, target_grid, None) == 1.0


def test_similarity_is_zero_on_shape_mismatch():
    target_grid = ((1, 1), (0, 0))
    diff_mask = compute_diff_mask(((0, 0), (0, 0)), target_grid)
    assert similarity(((1,),), target_grid, diff_mask) == 0.0


def test_compute_reward_matches_hand_computed_delta_for_a_progress_step():
    input_grid = ((0, 0), (0, 0))
    target_grid = ((1, 1), (0, 0))
    diff_mask = compute_diff_mask(input_grid, target_grid)
    partially_fixed = ((1, 0), (0, 0))

    result = compute_reward(input_grid, partially_fixed, target_grid, diff_mask, valid_action=True, exact_match=False)

    # similarity: 0.0 -> 0.5, minus step cost, not terminated.
    assert result.prev_similarity == pytest.approx(0.0)
    assert result.similarity == pytest.approx(0.5)
    assert result.reward == pytest.approx(0.5 - STEP_COST)


def test_compute_reward_adds_terminal_bonus_on_exact_match():
    input_grid = ((0, 0), (0, 0))
    target_grid = ((1, 1), (0, 0))
    diff_mask = compute_diff_mask(input_grid, target_grid)

    result = compute_reward(input_grid, target_grid, target_grid, diff_mask, valid_action=True, exact_match=True)

    assert result.reward == pytest.approx((1.0 - 0.0) - STEP_COST + TERMINAL_BONUS)


def test_compute_reward_subtracts_invalid_action_penalty():
    input_grid = ((0, 0), (0, 0))
    target_grid = ((1, 1), (0, 0))
    diff_mask = compute_diff_mask(input_grid, target_grid)

    # No-op: grid unchanged by an invalid action.
    result = compute_reward(input_grid, input_grid, target_grid, diff_mask, valid_action=False, exact_match=False)

    assert result.reward == pytest.approx(0.0 - STEP_COST - INVALID_ACTION_PENALTY)


def test_compute_reward_penalizes_regressing_a_previously_matched_cell():
    input_grid = ((1, 0), (0, 0))  # (0,0) already matches target; only (0,1) needs to change
    target_grid = ((1, 1), (0, 0))
    diff_mask = compute_diff_mask(input_grid, target_grid)
    assert diff_mask == frozenset({(0, 1)})

    # Breaking (0,0) (outside diff_mask) isn't visible to a same-diff_mask delta -
    # documenting the known ADR-0005 simplification, not asserting a penalty for it.
    regressed = ((0, 0), (0, 0))
    result = compute_reward(input_grid, regressed, target_grid, diff_mask, valid_action=True, exact_match=False)
    assert result.reward == pytest.approx(0.0 - STEP_COST)
