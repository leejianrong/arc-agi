"""The GP generational loop over the object grammar (SLICES.md V6):
tournament selection, elitism, crossover + mutation, stopping early on a
perfect-fitness genome - the same algorithm shape as `trainers/gp/evolve.py`,
swapped onto `trainers.gp_object.genome`/`trainers.gp_object.fitness`.

No `seed_programs` support: F12's LLM-seeded-search refinement is a
flat-genome feature (`trainers/gp/evolve.py`'s `run_gp`) and is out of scope
for V6.
"""

import random
from dataclasses import asdict, dataclass, field

from arc_env.tasks import SearchTask
from trainers.gp_object.fitness import ZERO_FITNESS, evaluate_fitness
from trainers.gp_object.genome import Genome, crossover, mutate, random_genome


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
    # genome for later replay - generation 0 and the final generation are
    # always snapshotted regardless of this interval. See
    # `trainers/gp/evolve.py`'s `GPConfig` for the full rationale; unchanged
    # here.
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
    best_genome: Genome
    best_fitness: tuple
    history: list = field(default_factory=list)  # list[GenerationRecord]
    n_generations_run: int = 0
    snapshots: list = field(default_factory=list)  # list[tuple[int, Genome]]


def _tournament_select(scored: list, rng: random.Random, k: int) -> Genome:
    contenders = rng.sample(scored, min(k, len(scored)))
    return max(contenders, key=lambda item: item[0])[1]


def run_gp(task: SearchTask, config: GPConfig) -> GPResult:
    rng = random.Random(config.seed)
    population = [random_genome(rng, config.max_program_length) for _ in range(config.population_size)]

    best_genome, best_fitness = population[0], ZERO_FITNESS
    history = []
    snapshots = []
    snapshot_interval = max(1, config.snapshot_interval)
    generation = 0

    for generation in range(config.n_generations):
        scored = [(evaluate_fitness(g, task).fitness, g) for g in population]
        scored.sort(key=lambda item: item[0], reverse=True)

        gen_best_fitness, gen_best_genome = scored[0]
        if gen_best_fitness > best_fitness:
            best_fitness, best_genome = gen_best_fitness, gen_best_genome

        mean_fitness = sum(f[0] for f, _ in scored) / len(scored)
        history.append(GenerationRecord(
            generation=generation,
            best_fitness=gen_best_fitness[0],
            best_similarity=gen_best_fitness[1],
            population_mean_fitness=mean_fitness,
        ))

        is_final_generation = best_fitness[0] >= 1.0 or generation == config.n_generations - 1
        if generation % snapshot_interval == 0 or is_final_generation:
            snapshots.append((generation, gen_best_genome))

        if best_fitness[0] >= 1.0:
            break

        next_population = [g for _, g in scored[:config.elitism]]
        while len(next_population) < config.population_size:
            parent1 = _tournament_select(scored, rng, config.tournament_size)
            parent2 = _tournament_select(scored, rng, config.tournament_size)
            child = crossover(parent1, parent2, rng, config.max_program_length) \
                if rng.random() < config.crossover_rate else list(parent1)
            child = mutate(child, rng, config.mutation_rate, config.max_program_length)
            next_population.append(child)
        population = next_population

    return GPResult(
        best_genome=best_genome, best_fitness=best_fitness, history=history,
        n_generations_run=generation + 1, snapshots=snapshots,
    )
