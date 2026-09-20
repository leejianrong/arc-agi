"""Fitness evaluation (ADR-0003): run a candidate program against a task's
train pairs, reusing `arc_env`'s executor and reward machinery directly -
"one definition of how close is this grid to correct" shared with PPO
(ADR-0005's consequences).

Fitness is `(exact_match_fraction, mean_similarity)`, compared
lexicographically (Python tuple comparison) - exact matches always
outweigh any amount of partial similarity, and similarity only breaks ties
between programs matching the same number of pairs, exactly as SLICES.md's
V4 build plan specifies ("fitness = fraction of train pairs matched,
falling back to the ADR-0005 similarity measure as a tiebreaker").
"""

from dataclasses import dataclass

from arc_env import actions
from arc_env import reward as reward_mod
from arc_env.task_loader import Task
from trainers.gp.genome import Program

Fitness = tuple  # (exact_match_fraction: float, mean_similarity: float)

ZERO_FITNESS: Fitness = (0.0, 0.0)


def run_program(program: Program, grid: tuple) -> tuple:
    """Applies `program`'s genes to `grid` in order via `actions.execute`,
    using target-independent endpoint semantics. The static end of the
    genome is an implicit RETURN; a valid `commit` is an explicit RETURN.

    An invalid gene (out-of-bounds for the current grid) is a no-op, same
    as the env's Q7 behavior, and execution continues from the unchanged
    grid. Crucially, no target is accepted by this function: a transient
    exact match cannot stop execution or hide destructive trailing genes.
    """

    selected = {0: None, 1: None}
    selected_region = {0: None, 1: None}
    for primitive_index, raw_args in program:
        new_grid, selected, selected_region, _, valid = actions.execute(
            primitive_index, raw_args, grid, selected, selected_region
        )
        grid = new_grid
        is_commit = (
            valid and 0 <= primitive_index < len(actions.ACTIONS)
            and actions.ACTIONS[primitive_index].name in ("commit", "commit_selection")
        )
        if is_commit:
            break
    return grid


@dataclass(frozen=True)
class FitnessResult:
    fitness: Fitness
    per_pair_exact_match: tuple  # tuple[bool, ...] - for diagnostics/logging


def evaluate_fitness(program: Program, task: Task) -> FitnessResult:
    exact_matches = []
    similarities = []
    for pair in task.train:
        diff_mask = reward_mod.compute_diff_mask(pair.input, pair.output)
        final_grid = run_program(program, pair.input)
        matched = final_grid == pair.output
        exact_matches.append(matched)
        similarities.append(reward_mod.similarity(final_grid, pair.output, diff_mask))

    fitness = (sum(exact_matches) / len(exact_matches), sum(similarities) / len(similarities))
    return FitnessResult(fitness=fitness, per_pair_exact_match=tuple(exact_matches))
