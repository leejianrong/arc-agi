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

from arc_env import actions, reward as reward_mod
from arc_env.task_loader import Task
from trainers.gp.genome import Program

Fitness = tuple  # (exact_match_fraction: float, mean_similarity: float)

ZERO_FITNESS: Fitness = (0.0, 0.0)


def run_program(program: Program, grid: tuple, target: tuple = None) -> tuple:
    """Applies `program`'s genes to `grid` in order via `actions.execute`,
    mirroring `ArcEnv.step`'s own termination conditions exactly (an
    earlier version of this function didn't, which meant a found program's
    fitness score and its logged replay trace - `trainers.gp.replay`, which
    *does* go through `ArcEnv.step` - could silently disagree about what
    the program actually does whenever a trailing gene ran after an exact
    match. Never worth it: matching what the env would actually do is
    strictly more informative than running blindly to the end).

    An invalid gene (out-of-bounds for the current grid) is a no-op, same
    as the env's Q7 behavior - the program keeps running from the
    unchanged grid. The program stops early - exactly when `ArcEnv.step`
    would set `terminated=True` - on a valid `commit` (its whole point is
    "this is my final answer") or, if `target` is given, on reaching an
    exact match.
    """

    if target is not None and grid == target:
        return grid

    for primitive_index, raw_args in program:
        new_grid, _, valid = actions.execute(primitive_index, raw_args, grid)
        grid = new_grid
        is_commit = valid and 0 <= primitive_index < len(actions.ACTIONS) and actions.ACTIONS[primitive_index].name == "commit"
        if is_commit or (target is not None and grid == target):
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
        final_grid = run_program(program, pair.input, target=pair.output)
        matched = final_grid == pair.output
        exact_matches.append(matched)
        similarities.append(reward_mod.similarity(final_grid, pair.output, diff_mask))

    fitness = (sum(exact_matches) / len(exact_matches), sum(similarities) / len(similarities))
    return FitnessResult(fitness=fitness, per_pair_exact_match=tuple(exact_matches))
