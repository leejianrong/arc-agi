"""Unit tests (SLICES.md V4): fitness evaluation on a hand-constructed
program/task pair matches a hand-computed expected value."""

import pytest

from arc_env import actions
from arc_env.task_loader import Pair, Task
from trainers.gp.fitness import evaluate_fitness, run_program


def _task(pairs):
    return Task(task_id="fixture", train=tuple(pairs), test=())


def test_run_program_applies_genes_in_order():
    vmirror_idx = actions.ACTION_BY_NAME["vmirror"]
    program = [(vmirror_idx, (0,) * actions.MAX_ARITY)]
    grid = ((1, 2), (3, 4))
    assert run_program(program, grid) == ((2, 1), (4, 3))


def test_run_program_stops_early_on_a_valid_commit():
    commit_idx = actions.ACTION_BY_NAME["commit"]
    replace_idx = actions.ACTION_BY_NAME["replace"]
    grid = ((1, 2), (3, 4))
    # commit(row=0, col=0, height=1, width=1) -> ((1,),) - crops and should
    # end the program, so the trailing `replace` never runs.
    program = [(commit_idx, (0, 0, 0, 0)), (replace_idx, (1, 9, 0, 0))]
    assert run_program(program, grid) == ((1,),)


def test_run_program_stops_early_on_reaching_the_target_when_given():
    # Regression test: this mirrors ArcEnv.step's own termination - without
    # it, evaluate_fitness's score and trainers.gp.replay's logged trace
    # for the same program could silently disagree whenever a gene ran
    # after an exact match was already reached.
    vmirror_idx = actions.ACTION_BY_NAME["vmirror"]
    replace_idx = actions.ACTION_BY_NAME["replace"]
    grid = ((1, 2), (3, 4))
    target = ((2, 1), (4, 3))  # == vmirror(grid)
    # The trailing replace(0, 9) would repaint every 0-cell if it ran -
    # there are none here, but the point is it must not even be attempted.
    program = [(vmirror_idx, (0,) * actions.MAX_ARITY), (replace_idx, (0, 9, 0, 0))]

    assert run_program(program, grid, target=target) == target
    # Without a target, the whole program still runs (no early stop).
    assert run_program(program, grid) == target  # replace(0,9) is a no-op here, so the result matches either way


def test_run_program_without_target_does_not_stop_on_exact_match():
    vmirror_idx = actions.ACTION_BY_NAME["vmirror"]
    switch_idx = actions.ACTION_BY_NAME["switch"]
    grid = ((1, 2), (3, 4))
    # vmirror(grid) coincidentally has no 1s/2s adjacency to exploit here,
    # so use a grid where the post-vmirror grid genuinely gets mutated by
    # a trailing gene to prove it wasn't skipped.
    program = [(vmirror_idx, (0,) * actions.MAX_ARITY), (switch_idx, (2, 1, 0, 0))]
    # vmirror(grid) = ((2,1),(4,3)); switch(2,1) swaps those two colors.
    assert run_program(program, grid) == ((1, 2), (4, 3))


def test_run_program_ignores_an_invalid_gene_as_a_noop():
    vmirror_idx = actions.ACTION_BY_NAME["vmirror"]
    fill_cell_idx = actions.ACTION_BY_NAME["fill_cell"]
    grid = ((1, 2), (3, 4))
    # fill_cell at row=29 is out of bounds for a 2x2 grid - a no-op, not a crash.
    program = [(fill_cell_idx, (5, 29, 0, 0)), (vmirror_idx, (0,) * actions.MAX_ARITY)]
    assert run_program(program, grid) == ((2, 1), (4, 3))


def test_evaluate_fitness_matches_hand_computed_value():
    vmirror_idx = actions.ACTION_BY_NAME["vmirror"]
    program = [(vmirror_idx, (0,) * actions.MAX_ARITY)]

    # Pair 1: vmirror gets it exactly right.
    pair1 = Pair(input=((1, 2), (3, 4)), output=((2, 1), (4, 3)))
    # Pair 2: vmirror does NOT solve it. diff_mask (cells where
    # input != target) is {(0,0), (0,1), (1,0)} - (1,1) already matches, so
    # it's excluded. vmirror((0,0),(0,1)) = ((0,0),(1,0)); of the 3
    # diff_mask cells, only (1,0) (value 1) matches the target -> 1/3.
    pair2 = Pair(input=((0, 0), (0, 1)), output=((1, 1), (1, 1)))
    task = _task([pair1, pair2])

    result = evaluate_fitness(program, task)

    assert result.per_pair_exact_match == (True, False)
    assert result.fitness[0] == pytest.approx(0.5)  # 1 of 2 pairs exact
    assert result.fitness[1] == pytest.approx((1.0 + 1 / 3) / 2)  # mean similarity


def test_evaluate_fitness_is_a_tuple_comparable_lexicographically():
    # A program matching more pairs always beats one that matches fewer,
    # regardless of similarity - the tiebreaker never overrides exact matches.
    assert (1.0, 0.0) > (0.5, 1.0)
    assert (0.5, 0.9) > (0.5, 0.1)
