#!/usr/bin/env python3
"""`train.py --algo ppo|gp --task_id <id> [--config ...]` (ADR-0003/0008,
SLICES.md V2/V4). `--algo ppo` trains one dedicated PPO policy for
`task_id` at solve-time (ADR-0008); `--algo gp` evolves one dedicated
population of DSL-program genomes for it instead (ADR-0003) - never shared
across tasks, either way. Both log to `runs/<run_id>/` in the same shape
(ADR-0006): `run_meta.json`, `metrics.jsonl` (reward/success rate per
update - PPO's per-rollout-update, GP's per-generation), and
`episodes/*.jsonl` (PPO: periodic greedy-policy eval episodes, so early-
and late-training replay can be compared side by side; GP: the single
best-found program's execution trace). PPO additionally checkpoints
(`checkpoints/update_*.pt`) - GP's "checkpoint" is just its best program,
already captured in the logged episode trace, so there's nothing separate
to resume from.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from arc_env.env import ArcEnv, DEFAULT_MAX_STEPS
from arc_env.episode_log import EpisodeWriter, RunMeta, write_run_meta
from arc_env.re_arc import generate_pair
from arc_env.task_loader import CURATED_TASK_IDS, load_task
from trainers.gp.evolve import GPConfig, run_gp
from trainers.gp.replay import program_to_episode_trace
from trainers.ppo.network import ActorCritic
from trainers.ppo.ppo import PPOConfig, ppo_update
from trainers.ppo.rollout import RolloutCollector, evaluate_episode

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def make_next_pair_fn(task_id: str, re_arc_prob: float, rng: random.Random):
    """Mixes the task's native train pairs with fresh `re-arc`-generated
    instances (SLICES.md V2 build plan step 3), beyond ARC's native ~3-5
    train pairs. Difficulty is sampled as a random +/-0.15 band per instance
    - varied practice without a curriculum schedule this milestone."""

    task = load_task(task_id)

    def next_pair():
        if rng.random() < re_arc_prob:
            center = rng.uniform(0.0, 1.0)
            pair = generate_pair(task_id, max(0.0, center - 0.15), min(1.0, center + 0.15))
        else:
            pair = task.train[rng.randrange(len(task.train))]
        return task_id, pair

    return next_pair


def save_checkpoint(run_dir: Path, update: int, network: ActorCritic, optimizer: torch.optim.Optimizer) -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoints_dir / f"update_{update:05d}.pt"
    torch.save({
        "update": update,
        "network_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    return path


def load_checkpoint(path: Path, network: ActorCritic, optimizer: torch.optim.Optimizer = None) -> int:
    """Returns the update index the checkpoint was saved at."""

    checkpoint = torch.load(path, weights_only=True)
    network.load_state_dict(checkpoint["network_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["update"]


def append_metrics(run_dir: Path, row: dict) -> None:
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def _write_episode(run_dir: Path, episode_id: str, env: ArcEnv, task_id: str, pair, result: dict) -> None:
    """`result` is the `{"steps", "success", "total_reward"}` shape both
    `evaluate_episode` (PPO) and `program_to_episode_trace` (GP) return."""

    with EpisodeWriter(run_dir, episode_id) as writer:
        writer.start(task_id=task_id, pair_index=0, input_grid=pair.input, target_grid=pair.output, max_steps=env.max_steps)
        for i, step in enumerate(result["steps"]):
            writer.step(
                step=i,
                grid_before=step["grid_before"],
                action_name=step["action_name"],
                action_args=step["action_args"],
                grid_after=step["grid_after"],
                reward=step["reward"],
                terminated=step["terminated"],
                truncated=step["truncated"],
                valid_action=step["valid_action"],
                exact_match=step["exact_match"],
            )
        writer.end(n_steps=len(result["steps"]), success=result["success"], total_reward=result["total_reward"])


def log_eval_episode(run_dir: Path, env: ArcEnv, network: ActorCritic, task_id: str, update: int) -> None:
    task = load_task(task_id)
    pair = task.train[0]
    result = evaluate_episode(env, network, task_id, pair)
    _write_episode(run_dir, f"eval-update{update:05d}", env, task_id, pair, result)


def train_ppo(
    task_id: str,
    run_dir: Path,
    n_updates: int,
    rollout_steps: int,
    eval_every: int,
    re_arc_prob: float,
    max_steps: int,
    seed: int,
    config: PPOConfig,
    resume_from: Path = None,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    env = ArcEnv(max_steps=max_steps)
    eval_env = ArcEnv(max_steps=max_steps)
    network = ActorCritic()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr)

    start_update = 0
    if resume_from is not None:
        start_update = load_checkpoint(resume_from, network, optimizer) + 1

    write_run_meta(run_dir, RunMeta(
        run_id=run_dir.name, algo="ppo", task_ids=[task_id],
        config={"n_updates": n_updates, "rollout_steps": rollout_steps, "eval_every": eval_every,
                "re_arc_prob": re_arc_prob, "max_steps": max_steps, "seed": seed, **config.to_dict()},
    ))

    collector = RolloutCollector(env, network, make_next_pair_fn(task_id, re_arc_prob, rng))

    for update in range(start_update, n_updates):
        buf = collector.collect(rollout_steps)
        stats = ppo_update(network, optimizer, buf, config)

        mean_reward = float(np.mean(buf.episode_returns)) if buf.episode_returns else None
        success_rate = float(np.mean(buf.episode_successes)) if buf.episode_successes else None
        append_metrics(run_dir, {
            "update": update,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_episodes": len(buf.episode_returns),
            "mean_reward": mean_reward,
            "success_rate": success_rate,
            **stats,
        })
        print(f"update {update:5d} | episodes {len(buf.episode_returns):3d} | "
              f"reward {mean_reward if mean_reward is not None else float('nan'):6.2f} | "
              f"success {success_rate if success_rate is not None else float('nan'):.2f}")

        if update % eval_every == 0 or update == n_updates - 1:
            log_eval_episode(run_dir, eval_env, network, task_id, update)
            save_checkpoint(run_dir, update, network, optimizer)


def train_gp(task_id: str, run_dir: Path, config: GPConfig, max_steps: int) -> None:
    task = load_task(task_id)
    env = ArcEnv(max_steps=max_steps)

    write_run_meta(run_dir, RunMeta(
        run_id=run_dir.name, algo="gp", task_ids=[task_id],
        config={"max_steps": max_steps, **config.to_dict()},
    ))

    result = run_gp(task, config)

    # GP runs to completion in one call (typically far faster than a PPO
    # update loop - see docstring), so metrics are logged after the fact
    # from `result.history`, not streamed live update-by-update like PPO's.
    for record in result.history:
        append_metrics(run_dir, {
            "update": record.generation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_episodes": config.population_size,
            "mean_reward": record.best_similarity,
            "success_rate": record.best_fitness,
            "population_mean_fitness": record.population_mean_fitness,
        })
        print(f"generation {record.generation:5d} | best_fitness {record.best_fitness:.2f} | "
              f"best_similarity {record.best_similarity:.2f} | pop_mean {record.population_mean_fitness:.3f}")

    trace = program_to_episode_trace(env, result.best_program, task_id, task.train[0])
    _write_episode(run_dir, "best-program", env, task_id, task.train[0], trace)

    print(f"\nGP finished after {result.n_generations_run} generation(s): "
          f"best_fitness={result.best_fitness[0]:.2f}, best_similarity={result.best_fitness[1]:.2f}")
    print(f"best program: {result.best_program}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["ppo", "gp"], default="ppo")
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--run_id", default=None, help="Defaults to a timestamp.")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs_dir", type=Path, default=RUNS_DIR)

    ppo_group = parser.add_argument_group("--algo ppo")
    ppo_group.add_argument("--n_updates", type=int, default=200)
    ppo_group.add_argument("--rollout_steps", type=int, default=256)
    ppo_group.add_argument("--eval_every", type=int, default=10)
    ppo_group.add_argument("--re_arc_prob", type=float, default=0.5)
    ppo_group.add_argument("--resume_from", type=Path, default=None)
    ppo_group.add_argument("--lr", type=float, default=3e-4)
    ppo_group.add_argument("--n_epochs", type=int, default=4)
    ppo_group.add_argument("--minibatch_size", type=int, default=64)

    gp_group = parser.add_argument_group("--algo gp")
    gp_group.add_argument("--population_size", type=int, default=200)
    gp_group.add_argument("--n_generations", type=int, default=100)
    gp_group.add_argument("--max_program_length", type=int, default=6)
    gp_group.add_argument("--tournament_size", type=int, default=3)
    gp_group.add_argument("--crossover_rate", type=float, default=0.7)
    gp_group.add_argument("--mutation_rate", type=float, default=0.3)
    gp_group.add_argument("--elitism", type=int, default=2)

    args = parser.parse_args()

    if args.task_id not in CURATED_TASK_IDS:
        parser.error(f"{args.task_id!r} is not in the curated task subset: {sorted(CURATED_TASK_IDS)}")

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.runs_dir / run_id

    if args.algo == "gp":
        config = GPConfig(
            population_size=args.population_size, n_generations=args.n_generations,
            max_program_length=args.max_program_length, tournament_size=args.tournament_size,
            crossover_rate=args.crossover_rate, mutation_rate=args.mutation_rate,
            elitism=args.elitism, seed=args.seed,
        )
        train_gp(task_id=args.task_id, run_dir=run_dir, config=config, max_steps=args.max_steps)
    else:
        config = PPOConfig(lr=args.lr, n_epochs=args.n_epochs, minibatch_size=args.minibatch_size)
        train_ppo(
            task_id=args.task_id, run_dir=run_dir, n_updates=args.n_updates,
            rollout_steps=args.rollout_steps, eval_every=args.eval_every, re_arc_prob=args.re_arc_prob,
            max_steps=args.max_steps, seed=args.seed, config=config, resume_from=args.resume_from,
        )
    print(f"\nwrote run to {run_dir}")


if __name__ == "__main__":
    main()
