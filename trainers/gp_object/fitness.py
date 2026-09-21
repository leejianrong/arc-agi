"""Fitness evaluation over the object grammar (SLICES.md V6): decode a
genome to a `Program` (`trainers.gp_object.genome.to_program`) and score it
against a task's train pairs with the exact same "how close is this grid to
correct" definition the flat-space GP and PPO share (ADR-0005's
`arc_env.reward`).

Fitness is `(exact_match_fraction, mean_similarity)`, compared
lexicographically - the same shape and tiebreak rule as
`trainers/gp/fitness.py`. The per-pair loop mirrors
`object_env.search.evaluate`'s exception-handling convention (a malformed
program can raise any DSL error; score that pair 0.0 similarity rather than
let one bad candidate crash the run) while also keeping a
`per_pair_exact_match` tuple for diagnostic parity with
`trainers/gp/fitness.FitnessResult`.
"""

from dataclasses import dataclass

from arc_env import reward as reward_mod
from arc_env.tasks import SearchTask
from trainers.gp_object.genome import Genome, to_program

Fitness = tuple  # (exact_match_fraction: float, mean_similarity: float)

ZERO_FITNESS: Fitness = (0.0, 0.0)


@dataclass(frozen=True)
class FitnessResult:
    fitness: Fitness
    per_pair_exact_match: tuple  # tuple[bool, ...] - for diagnostics/logging


def evaluate_fitness(genome: Genome, task: SearchTask) -> FitnessResult:
    program = to_program(genome)
    exact_matches = []
    similarities = []
    for pair in task.train:
        diff_mask = reward_mod.compute_diff_mask(pair.input, pair.output)
        try:
            out = program.run(pair.input)
        except Exception:  # noqa: BLE001 - a malformed program can raise any DSL error
            exact_matches.append(False)
            similarities.append(0.0)
            continue
        exact_matches.append(out == pair.output)
        similarities.append(reward_mod.similarity(out, pair.output, diff_mask))

    fitness = (sum(exact_matches) / len(exact_matches), sum(similarities) / len(similarities))
    return FitnessResult(fitness=fitness, per_pair_exact_match=tuple(exact_matches))
