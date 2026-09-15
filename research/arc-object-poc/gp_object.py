"""Gate 2 — searchability. A small, self-contained GP over the object action
space (`object_actions`), reusing the shipped GP's *algorithm shape*
(tournament + single-point crossover + mutation + elitism, `trainers/gp` values)
but NOT its modules, which are hard-wired to `arc_env.actions`.

Fitness reuses `arc_env.reward` (mean real-train exact-match, tie-broken by mean
similarity) — the same signal `trainers/gp/fitness.evaluate_fitness` uses.

The test: does GP *discover* a fitness-1.0 program over these 5 object actions,
for tasks GP over the flat 85-action space solved 0/8 in the 2026-09-15 pass?

    uv run python research/arc-object-poc/gp_object.py [--seeds 3] [--pop 200] [--gens 60]
"""

import argparse
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arc_env.reward import compute_diff_mask, similarity  # noqa: E402
from arc_env.task_loader import load_task  # noqa: E402

from object_actions import ACTIONS, MAX_ARITY, RAW_ARG_RANGE  # noqa: E402
from programs import TARGET_TASK_IDS  # noqa: E402
from verify import run_program  # noqa: E402


def _random_gene(rng):
    return (rng.randrange(len(ACTIONS)), tuple(rng.randrange(RAW_ARG_RANGE) for _ in range(MAX_ARITY)))


def _random_program(rng, max_len):
    return [_random_gene(rng) for _ in range(rng.randint(1, max_len))]


def _crossover(p1, p2, rng, max_len):
    if len(p1) < 1 or len(p2) < 1:
        return list(p1)
    c1 = rng.randint(0, len(p1))
    c2 = rng.randint(0, len(p2))
    child = list(p1[:c1]) + list(p2[c2:])
    return child[:max_len] if child else [_random_gene(rng)]


def _mutate(program, rng, rate, max_len):
    out = []
    for gene in program:
        if rng.random() < rate:
            out.append(_random_gene(rng))
        else:
            out.append(gene)
    if rng.random() < rate and len(out) < max_len:
        out.insert(rng.randrange(len(out) + 1), _random_gene(rng))
    if rng.random() < rate and len(out) > 1:
        del out[rng.randrange(len(out))]
    return out


def evaluate(program, task):
    exact = 0
    sims = []
    for pair in task.train:
        diff = compute_diff_mask(pair.input, pair.output)
        try:
            out = run_program(program, pair.input)
            sims.append(similarity(out, pair.output, diff))
            if out == pair.output:
                exact += 1
        except Exception:
            sims.append(0.0)
    n = len(task.train)
    return (exact / n, statistics.mean(sims) if sims else 0.0)


@dataclass
class GPConfig:
    population: int = 200
    generations: int = 60
    max_len: int = 6
    tournament: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    elitism: int = 2


@dataclass
class GPRun:
    task_id: str
    solved: bool
    generation: int | None
    best_fitness: tuple
    program: list = field(default_factory=list)


def _tournament(scored, rng, k):
    return max(rng.sample(scored, min(k, len(scored))), key=lambda sp: sp[1])[0]


def run_gp(task, config, seed) -> GPRun:
    rng = random.Random(seed)
    pop = [_random_program(rng, config.max_len) for _ in range(config.population)]
    best_prog, best_fit = None, (-1.0, -1.0)
    for gen in range(config.generations):
        scored = [(p, evaluate(p, task)) for p in pop]
        scored.sort(key=lambda sp: sp[1], reverse=True)
        if scored[0][1] > best_fit:
            best_prog, best_fit = scored[0]
        if best_fit[0] >= 1.0:
            return GPRun(task.task_id, True, gen, best_fit, best_prog)
        elite = [p for p, _ in scored[: config.elitism]]
        nxt = list(elite)
        while len(nxt) < config.population:
            p1 = _tournament(scored, rng, config.tournament)
            if rng.random() < config.crossover_rate:
                child = _crossover(p1, _tournament(scored, rng, config.tournament), rng, config.max_len)
            else:
                child = list(p1)
            nxt.append(_mutate(child, rng, config.mutation_rate, config.max_len))
        pop = nxt
    return GPRun(task.task_id, False, None, best_fit, best_prog)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--gens", type=int, default=60)
    args = ap.parse_args()
    config = GPConfig(population=args.pop, generations=args.gens)

    print(f"Gate 2 — object-space GP searchability (pop={args.pop} gens={args.gens} seeds={args.seeds})\n")
    solved_any = 0
    for task_id in TARGET_TASK_IDS:
        task = load_task(task_id)
        outcomes = [run_gp(task, config, seed) for seed in range(args.seeds)]
        wins = [o for o in outcomes if o.solved]
        if wins:
            solved_any += 1
            g = min(o.generation for o in wins)
            print(f"  SOLVED  {task_id}  {len(wins)}/{args.seeds} seeds  (earliest gen {g})")
        else:
            bf = max(o.best_fitness for o in outcomes)
            print(f"  ----    {task_id}  0/{args.seeds}  best_fitness={bf[0]:.2f}/{bf[1]:.3f}")

    print(f"\n  object-space GP solved: {solved_any}/{len(TARGET_TASK_IDS)}")
    print("\nGATE 2: " + ("PASS (>=1 discovered)" if solved_any >= 1 else "FAIL"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
