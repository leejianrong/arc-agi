"""Unit tests (SLICES.md V6): fitness evaluation on a hand-constructed
genome/task pair matches a hand-computed expected value."""

import pytest

from arc_env.tasks import Pair, SearchTask
from object_env.actions import ACTIONS
from object_env.grammar import legal_steps
from trainers.gp_object.fitness import evaluate_fitness
from trainers.gp_object.genome import to_program


def _gene_for(action_name, args, filled=frozenset({"grid"})):
    """The menu index (a valid `Gene`) that decodes to `action_name(args)`
    given `filled` - avoids hardcoding a magic int while staying fully
    deterministic (`legal_steps`'s order is fixed for a given `ACTIONS`)."""

    menu = legal_steps(filled, ACTIONS)
    for i, step in enumerate(menu):
        if step.action.name == action_name and step.args == args:
            return i
    raise AssertionError(f"{action_name}{args} is not legal given filled={sorted(filled)}")


def _task(pairs):
    return SearchTask(task_id="fixture", train=tuple(pairs), test_inputs=())


def test_to_program_applies_genes_in_order():
    genome = [_gene_for("replace_color", {"from_color": 1, "to_color": 9})]
    grid = ((1, 2), (3, 4))
    assert to_program(genome).run(grid) == ((9, 2), (3, 4))


def test_evaluate_fitness_matches_hand_computed_value():
    genome = [_gene_for("replace_color", {"from_color": 0, "to_color": 1})]

    # Pair 1: replace(0 -> 1) gets it exactly right.
    pair1 = Pair(input=((0, 2), (3, 4)), output=((1, 2), (3, 4)))
    # Pair 2: replace(0 -> 1) does NOT solve it - output is ((1, 1), (1, 1)),
    # target is ((1, 1), (1, 2)). diff_mask (cells where input != target) is
    # every cell: {(0,0), (0,1), (1,0), (1,1)}. Of those, 3 match the
    # produced grid ((1,1),(1,1)) - only (1,1) (2 vs 1) doesn't -> 3/4.
    pair2 = Pair(input=((0, 0), (0, 1)), output=((1, 1), (1, 2)))
    task = _task([pair1, pair2])

    result = evaluate_fitness(genome, task)

    assert result.per_pair_exact_match == (True, False)
    assert result.fitness[0] == pytest.approx(0.5)  # 1 of 2 pairs exact
    assert result.fitness[1] == pytest.approx((1.0 + 0.75) / 2)  # mean similarity


def test_evaluate_fitness_is_a_tuple_comparable_lexicographically():
    # A genome matching more pairs always beats one that matches fewer,
    # regardless of similarity - the tiebreaker never overrides exact matches.
    assert (1.0, 0.0) > (0.5, 1.0)
    assert (0.5, 0.9) > (0.5, 0.1)
