"""The GP generational loop (ADR-0003): tournament selection, elitism,
crossover + mutation, stopping early on a perfect-fitness program."""

import random
from dataclasses import asdict, dataclass, field

from arc_env.task_loader import Task
from trainers.gp.fitness import ZERO_FITNESS, evaluate_fitness
from trainers.gp.genome import Program, crossover, mutate, random_program


@dataclass
class GPConfig:
    population_size: int = 100
    n_generations: int = 50
    max_program_length: int = 6
    tournament_size: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    elitism: int = 2
    seed: int = 0
    # ADR-0014: how often (in generations) to snapshot the generation's best
    # program for later replay - generation 0 and the final generation
    # (whether that's an early stop on perfect fitness or n_generations - 1)
    # are always snapshotted regardless of this interval. 10 is a middling
    # default: ~11 snapshots for the 100-generation default config, each one
    # extra `ArcEnv` rollout at replay time (cheap - programs are at most
    # `max_program_length` steps) but still enough to see real evolution
    # rather than every generation's near-identical neighbor.
    snapshot_interval: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationRecord:
    generation: int
    best_fitness: float  # exact_match_fraction
    best_similarity: float
    population_mean_fitness: float


@dataclass
class GPResult:
    best_program: Program
    best_fitness: tuple
    history: list = field(default_factory=list)  # list[GenerationRecord]
    n_generations_run: int = 0
    # ADR-0014: (generation, program) for each snapshotted generation's own
    # best program, generation 0 first - lets a replay show evolution across
    # the run instead of only the final `best_program`. Elitism guarantees
    # each generation's own best fitness is non-decreasing run-over-run (the
    # previous top `elitism` genomes always carry over unchanged and
    # re-score identically, since fitness is deterministic), so this series
    # is itself a monotonically-improving trace, not a noisy one.
    snapshots: list = field(default_factory=list)  # list[tuple[int, Program]]


def _tournament_select(scored: list, rng: random.Random, k: int) -> Program:
    contenders = rng.sample(scored, min(k, len(scored)))
    return max(contenders, key=lambda item: item[0])[1]


def run_gp(task: Task, config: GPConfig, seed_programs: list | None = None) -> GPResult:
    """`seed_programs` (F12's "LLM-seeded search" refinement, `docs/
    QUESTIONS.md`): an optional list of `Program`s to include in generation
    0's population, in place of some of its random slots rather than on top
    of `population_size` - e.g. a demonstration proposed by an LLM (or any
    other external source) from the task's train pairs, verified to
    actually solve it before being handed in here. Capped to
    `config.population_size` entries (extras beyond that are dropped, never
    grown past the configured population); the remaining slots are filled
    with `random_program` exactly as before. Omitting `seed_programs` (the
    default `None`, and an empty list) reproduces today's pure-random
    generation-0 population byte-for-byte - this parameter is strictly
    additive and opt-in, chosen as a `run_gp` argument rather than a new
    `GPConfig` field so `GPConfig.to_dict()` (serialized verbatim into
    `run_meta.json`'s `config`) stays a flat, JSON-trivial hyperparameter
    record - a list of programs doesn't belong there."""

    rng = random.Random(config.seed)
    if seed_programs:
        seeds = list(seed_programs)[:config.population_size]
        n_random = config.population_size - len(seeds)
        population = seeds + [random_program(rng, config.max_program_length) for _ in range(n_random)]
    else:
        population = [random_program(rng, config.max_program_length) for _ in range(config.population_size)]

    best_program, best_fitness = population[0], ZERO_FITNESS
    history = []
    snapshots = []
    snapshot_interval = max(1, config.snapshot_interval)
    generation = 0

    for generation in range(config.n_generations):
        scored = [(evaluate_fitness(p, task).fitness, p) for p in population]
        scored.sort(key=lambda item: item[0], reverse=True)

        gen_best_fitness, gen_best_program = scored[0]
        if gen_best_fitness > best_fitness:
            best_fitness, best_program = gen_best_fitness, gen_best_program

        mean_fitness = sum(f[0] for f, _ in scored) / len(scored)
        history.append(GenerationRecord(
            generation=generation,
            best_fitness=gen_best_fitness[0],
            best_similarity=gen_best_fitness[1],
            population_mean_fitness=mean_fitness,
        ))

        is_final_generation = best_fitness[0] >= 1.0 or generation == config.n_generations - 1
        if generation % snapshot_interval == 0 or is_final_generation:
            snapshots.append((generation, gen_best_program))

        if best_fitness[0] >= 1.0:
            break

        next_population = [p for _, p in scored[:config.elitism]]
        while len(next_population) < config.population_size:
            parent1 = _tournament_select(scored, rng, config.tournament_size)
            parent2 = _tournament_select(scored, rng, config.tournament_size)
            child = crossover(parent1, parent2, rng, config.max_program_length) \
                if rng.random() < config.crossover_rate else list(parent1)
            child = mutate(child, rng, config.mutation_rate, config.max_program_length)
            next_population.append(child)
        population = next_population

    return GPResult(
        best_program=best_program, best_fitness=best_fitness, history=history,
        n_generations_run=generation + 1, snapshots=snapshots,
    )
