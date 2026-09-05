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
to resume from. PPO also tracks the best-so-far eval checkpoint by the fixed
eval pair's greedy-policy reward (`checkpoints/best.pt`, `eval_success`/
`eval_reward` in `metrics.jsonl` - KAN-1177), since that per-update row's
`success_rate`/`mean_reward` are averaged over the training rollout's own
noisy, shifting mix of re-arc-generated and native pairs, not this fixed
pair - a real but short-lived swing in that rollout statistic is not the
same thing as the policy regressing on the pair eval/replay actually cares
about.
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from arc_env.env import DEFAULT_MAX_STEPS, ArcEnv
from arc_env.episode_log import EpisodeWriter, RunMeta, write_run_meta
from arc_env.re_arc import GenerationError, generate_pair
from arc_env.task_loader import CURATED_TASK_IDS, load_task
from trainers.gp.evolve import GPConfig, run_gp
from trainers.gp.replay import program_to_episode_trace
from trainers.ppo.network import ActorCritic
from trainers.ppo.ppo import PPOConfig, ppo_update
from trainers.ppo.rollout import RolloutCollector, evaluate_episode
from trainers.ppo.warm_start import load_demonstration, pretrain_from_demonstration

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
            try:
                pair = generate_pair(task_id, max(0.0, center - 0.15), min(1.0, center + 0.15))
            except GenerationError:
                # A narrow difficulty band can make every re-arc attempt land
                # on a degenerate (input == output) instance - e.g. 67a3c6ac's
                # generator picks width 1 with non-trivial odds near diff 0,
                # and vmirror is a no-op on a single column. Rare per call,
                # but over a full training run's worth of calls, likely
                # enough to hit eventually - fall back to a native pair
                # rather than crash the run.
                pair = task.train[rng.randrange(len(task.train))]
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


def check_warm_start_compatible(task_id: str, warm_start_from: Path) -> str | None:
    """Returns an error message if `warm_start_from` isn't a GP run for
    `task_id` (ADR-0009's same-task-only constraint), else `None`."""

    meta_path = warm_start_from / "run_meta.json"
    if not meta_path.is_file():
        return f"{warm_start_from} is not a run directory (no run_meta.json)"
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("algo") != "gp":
        return f"{warm_start_from} is not a GP run (algo={meta.get('algo')!r})"
    if meta.get("task_ids") != [task_id]:
        return (
            f"{warm_start_from} was trained on {meta.get('task_ids')!r}, not [{task_id!r}] - "
            "warm-start is same-task only (ADR-0009)"
        )
    return None


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
                selected=step["selected"],
            )
        writer.end(n_steps=len(result["steps"]), success=result["success"], total_reward=result["total_reward"])


def log_eval_episode(run_dir: Path, env: ArcEnv, network: ActorCritic, task_id: str, update: int) -> dict:
    """Returns the `evaluate_episode` result dict (`{"steps", "success",
    "total_reward"}`) in addition to writing the trace, so callers can fold
    the greedy-policy outcome on the literal held-out pair into `metrics.jsonl`
    (KAN-1177) - distinct from that row's `success_rate`/`mean_reward`, which
    are averaged over the *training* rollout's own mix of re-arc-generated
    and native pairs, not this fixed eval pair."""

    task = load_task(task_id)
    pair = task.train[0]
    result = evaluate_episode(env, network, task_id, pair)
    _write_episode(run_dir, f"eval-update{update:05d}", env, task_id, pair, result)
    return result


def is_new_best_eval(eval_reward: float, best_eval_reward: float | None) -> bool:
    """Whether `eval_reward` (an eval checkpoint's greedy-policy total reward
    on the fixed held-out pair) beats the best eval reward seen so far in
    this run. `None` means no eval checkpoint has run yet, so anything
    counts as a new best.

    KAN-1177: investigating PPO runs that reach a real success rate at some
    intermediate checkpoint and lose it by the final one. The rollout-based
    `success_rate` logged every update is noisy (a handful of stochastic-
    policy episodes over a shifting mix of re-arc-generated + native pairs)
    and is *not* the same thing as whether the policy still solves the fixed
    eval pair - reproductions for KAN-1177 found the latter stayed stable
    even when the former swung from ~80% to 0% between adjacent updates.
    Tracking the best eval checkpoint (`checkpoints/best.pt`) by this
    less-noisy, fixed-target signal gives a safety net against genuine
    eval-pair regression without having to trust a single final checkpoint."""

    return best_eval_reward is None or eval_reward > best_eval_reward


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
    resume_from: Path | None = None,
    warm_start_from: Path | None = None,
    warm_start_epochs: int = 50,
    warm_start_batch_size: int = 32,
    warm_start_lr: float = 1e-3,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    env = ArcEnv(max_steps=max_steps)
    eval_env = ArcEnv(max_steps=max_steps)
    network = ActorCritic()

    # ADR-0009: a one-time supervised pretrain phase against a same-task GP
    # run's best-program trajectory, before the normal PPO optimizer/loop
    # exist - skipped when also resuming, since `load_checkpoint` below
    # would immediately overwrite these weights with the checkpoint's own
    # (already-trained, possibly already warm-started) state anyway.
    warm_start_losses = None
    if warm_start_from is not None and resume_from is None:
        demonstration = load_demonstration(warm_start_from)
        print(f"warm-starting from {warm_start_from} ({len(demonstration)} demonstrated steps)")
        warm_start_losses = pretrain_from_demonstration(
            network, demonstration, n_epochs=warm_start_epochs,
            batch_size=warm_start_batch_size, lr=warm_start_lr,
        )
        for epoch, loss in enumerate(warm_start_losses):
            print(f"warm-start epoch {epoch:4d} | bc_loss {loss:.4f}")

    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr)

    start_update = 0
    if resume_from is not None:
        start_update = load_checkpoint(resume_from, network, optimizer) + 1

    write_run_meta(run_dir, RunMeta(
        run_id=run_dir.name, algo="ppo", task_ids=[task_id],
        config={"n_updates": n_updates, "rollout_steps": rollout_steps, "eval_every": eval_every,
                "re_arc_prob": re_arc_prob, "max_steps": max_steps, "seed": seed,
                "warm_start_from": str(warm_start_from) if warm_start_from else None,
                "warm_start_epochs": warm_start_epochs if warm_start_losses else None,
                "warm_start_final_loss": warm_start_losses[-1] if warm_start_losses else None,
                **config.to_dict()},
    ))

    collector = RolloutCollector(env, network, make_next_pair_fn(task_id, re_arc_prob, rng))
    best_eval_reward = None  # KAN-1177: best-so-far fixed-eval-pair reward, not resumed across --resume_from

    for update in range(start_update, n_updates):
        buf = collector.collect(rollout_steps)
        stats = ppo_update(network, optimizer, buf, config)

        mean_reward = float(np.mean(buf.episode_returns)) if buf.episode_returns else None
        success_rate = float(np.mean(buf.episode_successes)) if buf.episode_successes else None
        eval_success = None
        eval_reward = None

        if update % eval_every == 0 or update == n_updates - 1:
            eval_result = log_eval_episode(run_dir, eval_env, network, task_id, update)
            eval_success = eval_result["success"]
            eval_reward = eval_result["total_reward"]
            checkpoint_path = save_checkpoint(run_dir, update, network, optimizer)
            if is_new_best_eval(eval_reward, best_eval_reward):
                best_eval_reward = eval_reward
                shutil.copy2(checkpoint_path, checkpoint_path.parent / "best.pt")

        append_metrics(run_dir, {
            "update": update,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_episodes": len(buf.episode_returns),
            "mean_reward": mean_reward,
            "success_rate": success_rate,
            "eval_success": eval_success,
            "eval_reward": eval_reward,
            **stats,
        })
        print(f"update {update:5d} | episodes {len(buf.episode_returns):3d} | "
              f"reward {mean_reward if mean_reward is not None else float('nan'):6.2f} | "
              f"success {success_rate if success_rate is not None else float('nan'):.2f}")


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

    # ADR-0014: one replay per snapshotted generation, so the visualizer's
    # existing multi-episode picker can show evolution across the run - not
    # just the final winner. Named with a zero-padded numeric prefix so it
    # sorts chronologically *and* before "best-program" below (digits sort
    # before letters), which matters because `main.ts` picks the
    # alphabetically-first/last episode IDs for its early-vs-late comparison
    # panels.
    for generation, program in result.snapshots:
        trace = program_to_episode_trace(env, program, task_id, task.train[0])
        _write_episode(run_dir, f"{generation:05d}-gen", env, task_id, task.train[0], trace)

    # Kept as its own, unchanged-name episode (not just the last snapshot
    # above, even though they're equivalent by construction - see
    # `GPResult.snapshots`'s docstring) because ADR-0009's warm-start path
    # (`trainers/ppo/warm_start.py`'s `load_demonstration`) hardcodes
    # `episode_id="best-program"` - renaming or dropping this would silently
    # break `--warm_start_from`.
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
    ppo_group.add_argument(
        "--warm_start_from", type=Path, default=None,
        help="An existing runs/<run_id>/ from a prior `--algo gp` run for the same --task_id (ADR-0009).",
    )
    ppo_group.add_argument("--warm_start_epochs", type=int, default=50)
    ppo_group.add_argument("--warm_start_batch_size", type=int, default=32)
    ppo_group.add_argument("--warm_start_lr", type=float, default=1e-3)

    gp_group = parser.add_argument_group("--algo gp")
    gp_group.add_argument("--population_size", type=int, default=200)
    gp_group.add_argument("--n_generations", type=int, default=100)
    gp_group.add_argument("--max_program_length", type=int, default=6)
    gp_group.add_argument("--tournament_size", type=int, default=3)
    gp_group.add_argument("--crossover_rate", type=float, default=0.7)
    gp_group.add_argument("--mutation_rate", type=float, default=0.3)
    gp_group.add_argument("--elitism", type=int, default=2)
    gp_group.add_argument("--snapshot_interval", type=int, default=10)

    args = parser.parse_args()

    if args.task_id not in CURATED_TASK_IDS:
        parser.error(f"{args.task_id!r} is not in the curated task subset: {sorted(CURATED_TASK_IDS)}")

    if args.warm_start_from is not None:
        if args.algo != "ppo":
            parser.error("--warm_start_from is only valid with --algo ppo")
        error = check_warm_start_compatible(args.task_id, args.warm_start_from)
        if error is not None:
            parser.error(error)

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.runs_dir / run_id

    if args.algo == "gp":
        config = GPConfig(
            population_size=args.population_size, n_generations=args.n_generations,
            max_program_length=args.max_program_length, tournament_size=args.tournament_size,
            crossover_rate=args.crossover_rate, mutation_rate=args.mutation_rate,
            elitism=args.elitism, seed=args.seed, snapshot_interval=args.snapshot_interval,
        )
        train_gp(task_id=args.task_id, run_dir=run_dir, config=config, max_steps=args.max_steps)
    else:
        config = PPOConfig(lr=args.lr, n_epochs=args.n_epochs, minibatch_size=args.minibatch_size)
        train_ppo(
            task_id=args.task_id, run_dir=run_dir, n_updates=args.n_updates,
            rollout_steps=args.rollout_steps, eval_every=args.eval_every, re_arc_prob=args.re_arc_prob,
            max_steps=args.max_steps, seed=args.seed, config=config, resume_from=args.resume_from,
            warm_start_from=args.warm_start_from, warm_start_epochs=args.warm_start_epochs,
            warm_start_batch_size=args.warm_start_batch_size, warm_start_lr=args.warm_start_lr,
        )
    print(f"\nwrote run to {run_dir}")


if __name__ == "__main__":
    main()
