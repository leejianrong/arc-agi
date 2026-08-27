"""JSONL trajectory + run_meta.json writers, per ADR-0006.

Each `episodes/<episode_id>.jsonl` file is a sequence of line-delimited JSON
records, one `"start"` record, one `"step"` record per env step, and one
`"end"` record - `EpisodeWriter` is the only thing that should produce this
shape, so trainers (V2/V4) and this slice's random rollout script write
byte-identical schemas and the visualizer needs one parser for both.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1


def grid_to_list(grid: tuple) -> list:
    return [list(row) for row in grid]


@dataclass
class RunMeta:
    run_id: str
    algo: str
    task_ids: list
    config: dict


def write_run_meta(run_dir: Path, meta: RunMeta) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": meta.run_id,
        "algo": meta.algo,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_ids": meta.task_ids,
        "config": meta.config,
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(payload, f, indent=2)


class EpisodeWriter:
    """Writes one `episodes/<episode_id>.jsonl` file.

    Usage::

        with EpisodeWriter(run_dir, episode_id) as w:
            w.start(task_id, pair_index, input_grid, target_grid, max_steps)
            w.step(step_idx, grid_before, action_name, action_args,
                    grid_after, reward, terminated, truncated, valid_action)
            ...
            w.end(n_steps, success, total_reward)
    """

    def __init__(self, run_dir: Path, episode_id: str):
        self.episode_id = episode_id
        episodes_dir = run_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        self._path = episodes_dir / f"{episode_id}.jsonl"
        self._f = None

    def __enter__(self):
        self._f = open(self._path, "w")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._f is not None:
            self._f.close()

    def _write(self, record: dict) -> None:
        self._f.write(json.dumps(record) + "\n")

    def start(self, task_id: str, pair_index: int, input_grid: tuple, target_grid: tuple, max_steps: int) -> None:
        self._write({
            "type": "start",
            "episode_id": self.episode_id,
            "task_id": task_id,
            "pair_index": pair_index,
            "grid": grid_to_list(input_grid),
            "target_grid": grid_to_list(target_grid),
            "max_steps": max_steps,
        })

    def step(
        self,
        step: int,
        grid_before: tuple,
        action_name: str,
        action_args: dict,
        grid_after: tuple,
        reward: float,
        terminated: bool,
        truncated: bool,
        valid_action: bool,
        exact_match: bool = None,
    ) -> None:
        """`terminated` can now be true without `exact_match` (V3's `commit`
        action ends the episode on a chosen crop whether or not it matches)
        - `exact_match` defaults to `terminated` for callers written before
        V3 that only ever had one way to terminate."""

        self._write({
            "type": "step",
            "step": step,
            "grid_before": grid_to_list(grid_before),
            "action": {"name": action_name, "args": action_args},
            "grid_after": grid_to_list(grid_after),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": terminated or truncated,
            "valid_action": valid_action,
            "exact_match": terminated if exact_match is None else exact_match,
        })

    def end(self, n_steps: int, success: bool, total_reward: float) -> None:
        self._write({
            "type": "end",
            "n_steps": n_steps,
            "success": success,
            "total_reward": total_reward,
        })
