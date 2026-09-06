#!/usr/bin/env python3
"""Build one hand-scripted episode from a curated task's known solver
program, for headless-browser verification of the visualizer (see
../SKILL.md). Not part of the training pipeline - writes to
`<repo>/runs/<run_id>/`, which is gitignored; delete it once you're done
looking at it.

Usage:
    uv run python build_verify_episode.py
        # default: task 1f85a75f, program [select_largest, commit_selection]
        # - exercises the amber selection-overlay rendering end to end.
    uv run python build_verify_episode.py --task_id 23b5c85d --run_id my-check
"""

import argparse
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise SystemExit("could not find repo root (no pyproject.toml in any parent dir)")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))

from arc_env import actions
from arc_env.env import ArcEnv
from arc_env.episode_log import EpisodeWriter, RunMeta, write_run_meta
from arc_env.task_loader import CURATED_TASK_IDS, load_task


def build(task_id: str, run_id: str, episode_id: str, runs_dir: Path) -> Path:
    if task_id not in CURATED_TASK_IDS:
        raise SystemExit(f"{task_id!r} is not a curated task: {sorted(CURATED_TASK_IDS)}")
    program = CURATED_TASK_IDS[task_id]
    if not program:
        raise SystemExit(f"{task_id!r} has no known solver program scripted in task_loader.py")

    run_dir = runs_dir / run_id
    env = ArcEnv()
    task = load_task(task_id)
    env.reset(task_id=task_id, pair_index=0, task=task)

    with EpisodeWriter(run_dir, episode_id) as writer:
        writer.start(
            task_id=task_id,
            pair_index=0,
            input_grid=env.get_grid(),
            target_grid=task.train[0].output,
            max_steps=env.max_steps,
        )
        total_reward = 0.0
        exact_match = False
        step = 0
        for name, program_args in program:
            primitive_index = actions.ACTION_BY_NAME[name]
            arity = actions.ACTIONS[primitive_index].arity
            action = {"primitive": primitive_index}
            for i in range(actions.MAX_ARITY):
                action[f"arg{i + 1}"] = program_args[i] if i < arity else 0
            grid_before = env.get_grid()
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
                selected=info["selected"],
            )
            step += 1
        writer.end(n_steps=step, success=exact_match, total_reward=total_reward)

    write_run_meta(run_dir, RunMeta(run_id=run_id, algo="human", task_ids=[task_id], config={}))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task_id",
        default="1f85a75f",
        help="A curated task_id with a known solver program (default exercises the selection overlay).",
    )
    parser.add_argument("--run_id", default="verify")
    parser.add_argument("--episode_id", default="verify-episode")
    parser.add_argument("--runs_dir", type=Path, default=REPO_ROOT / "runs")
    args = parser.parse_args()
    run_dir = build(args.task_id, args.run_id, args.episode_id, args.runs_dir)
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
