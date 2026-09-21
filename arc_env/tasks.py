"""ARC task models and trust-aware loaders.

``load_task`` is the scorer/inspection loader and can expose complete pairs.
Search and training code must use ``load_search_task``: it rejects locked
splits and returns test inputs without their outputs.
"""

import json
import re
from dataclasses import dataclass

from arc_env.splits import DEVELOPMENT, get_split, require_search_allowed

Grid = tuple[tuple[int, ...], ...]
_TASK_ID_RE = re.compile(r"^[0-9a-f]{8}$")


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class Task:
    task_id: str
    train: tuple[Pair, ...]
    test: tuple[Pair, ...]
    split: str = DEVELOPMENT


@dataclass(frozen=True)
class SearchTask:
    """The only task view trainers/search algorithms should receive."""

    task_id: str
    train: tuple[Pair, ...]
    test_inputs: tuple[Grid, ...]
    split: str = DEVELOPMENT


def _validate_task_id(task_id: str) -> None:
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError(f"invalid ARC task id: {task_id!r}")


def _to_grid(rows: list) -> Grid:
    return tuple(tuple(row) for row in rows)


def _read_raw_task(task_id: str, split: str) -> dict:
    _validate_task_id(task_id)
    spec = get_split(split)
    with open(spec.data_dir / f"{task_id}.json") as task_file:
        return json.load(task_file)


def load_task(task_id: str, split: str = DEVELOPMENT) -> Task:
    """Load complete data for trusted scoring, auditing, or visualization."""

    raw = _read_raw_task(task_id, split)
    train = tuple(Pair(_to_grid(p["input"]), _to_grid(p["output"])) for p in raw["train"])
    test = tuple(Pair(_to_grid(p["input"]), _to_grid(p["output"])) for p in raw["test"])
    return Task(task_id=task_id, train=train, test=test, split=split)


def load_search_task(task_id: str, split: str = DEVELOPMENT) -> SearchTask:
    """Load a target-safe task view for search, training, and selection."""

    require_search_allowed(split)
    raw = _read_raw_task(task_id, split)
    train = tuple(Pair(_to_grid(p["input"]), _to_grid(p["output"])) for p in raw["train"])
    test_inputs = tuple(_to_grid(p["input"]) for p in raw["test"])
    return SearchTask(task_id=task_id, train=train, test_inputs=test_inputs, split=split)
