"""Loads the V1 same-shape-only, curated-action-space task subset.

`CURATED_TASK_IDS` was derived, not hand-picked: a task qualifies iff (a)
every train/test pair's output grid has the exact same shape as its input
grid, and (b) `third_party/arc-dsl/solvers.py`'s known-correct solver for
that task calls only primitives in `arc_env.actions`'s
`ZERO_ARG`/`ONE_ARG`/`TWO_ARG` sets (i.e. never touches an `Object`/
`Indices`/`Callable`-typed primitive, and never uses `canvas`/`crop`). See
`arc_env/actions.py`'s module docstring for the full reasoning; the
derivation script's output is reproduced in `tests/test_dsl_regression.py`'s
module docstring for anyone who wants to re-run the check.

This is also exactly the V1 regression-test fixture set
(`tests/test_dsl_regression.py`): each task's solver program, replayed
through `arc_env.actions.execute`, must reproduce the task's expected output
exactly.
"""

import json
from dataclasses import dataclass
from pathlib import Path

TRAINING_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "third_party" / "ARC-AGI" / "data" / "training"
)

# task_id -> the known-correct solver's call sequence, as
# (primitive_name, real_args) pairs transcribed from
# `third_party/arc-dsl/solvers.py`'s `solve_<task_id>`. This is both the V1
# task subset and the regression-test fixture table
# (`tests/test_dsl_regression.py` replays each sequence through
# `arc_env.actions` and checks it reproduces the task's exact output).
CURATED_TASK_IDS = {
    "6150a2bd": [("rot180", ())],
    "b1948b0a": [("replace", (6, 2))],
    "3c9b0459": [("rot180", ())],
    "9dfd6313": [("dmirror", ())],
    "c8f0f002": [("replace", (7, 5))],
    "ed36ccf7": [("rot270", ())],
    "74dd1130": [("dmirror", ())],
    "d511f180": [("switch", (5, 8))],
    "67a3c6ac": [("vmirror", ())],
    "68b16354": [("hmirror", ())],
    "0d3d703e": [
        ("switch", (3, 4)),
        ("switch", (8, 9)),
        ("switch", (2, 6)),
        ("switch", (1, 5)),
    ],
}


@dataclass(frozen=True)
class Pair:
    input: tuple
    output: tuple


@dataclass(frozen=True)
class Task:
    task_id: str
    train: tuple  # tuple[Pair, ...]
    test: tuple  # tuple[Pair, ...]


def _to_grid(rows: list) -> tuple:
    return tuple(tuple(row) for row in rows)


def load_task(task_id: str) -> Task:
    with open(TRAINING_DATA_DIR / f"{task_id}.json") as f:
        raw = json.load(f)
    train = tuple(Pair(_to_grid(p["input"]), _to_grid(p["output"])) for p in raw["train"])
    test = tuple(Pair(_to_grid(p["input"]), _to_grid(p["output"])) for p in raw["test"])
    return Task(task_id=task_id, train=train, test=test)


def load_curated_tasks() -> dict:
    """Returns `{task_id: Task}` for every task in `CURATED_TASK_IDS`."""

    return {task_id: load_task(task_id) for task_id in CURATED_TASK_IDS}


def iter_curated_pairs():
    """Yields `(task_id, pair_index, Pair)` for every train pair of every
    curated task - the unit of one V1 episode."""

    for task_id, task in load_curated_tasks().items():
        for i, pair in enumerate(task.train):
            yield task_id, i, pair
