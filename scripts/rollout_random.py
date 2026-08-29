#!/usr/bin/env python3
"""V1 random-policy rollout script (SLICES.md V1, step 3).

Steps a random policy through the env for one or more curated tasks and
writes `runs/<run_id>/episodes/<episode_id>.jsonl` + `run_meta.json`
(ADR-0006), for the visualizer to replay.

    python scripts/rollout_random.py --task_id 67a3c6ac
    python scripts/rollout_random.py --all --run_id demo
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_env.env import DEFAULT_MAX_STEPS, ArcEnv
from arc_env.episode_log import EpisodeWriter, RunMeta, write_run_meta
from arc_env.task_loader import CURATED_TASK_IDS, load_task

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def run_episode(env: ArcEnv, run_dir: Path, task_id: str, pair_index: int, episode_id: str) -> dict:
    task = load_task(task_id)
    _obs, info = env.reset(task_id=task_id, pair_index=pair_index, task=task)

    with EpisodeWriter(run_dir, episode_id) as writer:
        writer.start(
            task_id=task_id,
            pair_index=pair_index,
            input_grid=env.get_grid(),
            target_grid=task.train[pair_index].output,
            max_steps=env.max_steps,
        )

        total_reward = 0.0
        step = 0
        terminated = truncated = False
        exact_match = False
        while not (terminated or truncated):
            grid_before = env.get_grid()
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            exact_match = info["exact_match"]
            writer.step(
                step=step,
                grid_before=grid_before,
                action_name=info["action_name"],
                action_args=info["action_args"],
                grid_after=env.get_grid(),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                valid_action=info["valid_action"],
                exact_match=exact_match,
            )
            step += 1

        writer.end(n_steps=step, success=exact_match, total_reward=total_reward)

    return {"episode_id": episode_id, "task_id": task_id, "success": exact_match, "n_steps": step}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_id", help="A single curated task_id to roll out.")
    parser.add_argument("--all", action="store_true", help="Roll out every curated task's first train pair.")
    parser.add_argument("--pair_index", type=int, default=0)
    parser.add_argument("--run_id", default=None, help="Defaults to a timestamp.")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs_dir", type=Path, default=RUNS_DIR, help="Defaults to <repo>/runs.")
    args = parser.parse_args()

    if not args.task_id and not args.all:
        parser.error("pass --task_id <id> or --all")
    if args.task_id and args.task_id not in CURATED_TASK_IDS:
        parser.error(f"{args.task_id!r} is not in the curated task subset: {sorted(CURATED_TASK_IDS)}")

    task_ids = sorted(CURATED_TASK_IDS) if args.all else [args.task_id]
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.runs_dir / run_id

    env = ArcEnv(max_steps=args.max_steps)
    env.action_space.seed(args.seed)

    write_run_meta(
        run_dir,
        RunMeta(run_id=run_id, algo="random", task_ids=task_ids, config={
            "max_steps": args.max_steps, "seed": args.seed, "pair_index": args.pair_index,
        }),
    )

    results = []
    for task_id in task_ids:
        episode_id = f"{task_id}-p{args.pair_index}"
        result = run_episode(env, run_dir, task_id, args.pair_index, episode_id)
        results.append(result)
        print(f"{result['episode_id']}: {'SOLVED' if result['success'] else 'not solved'} in {result['n_steps']} steps")

    print(f"\nwrote {len(results)} episode(s) to {run_dir}")


if __name__ == "__main__":
    main()
