"""Immutable ARC dataset split policy.

The manifest is deliberately data-only: importing the approved task IDs must
not import the task-specific known-solver fixtures in ``task_loader.py``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(__file__).with_name("dataset_splits.json")
DEVELOPMENT = "development"
LOCKED_EVALUATION = "locked_evaluation"


@dataclass(frozen=True)
class SplitSpec:
    name: str
    data_dir: Path
    task_count: int
    sha256: str
    search_allowed: bool
    outputs_locked: bool


with open(MANIFEST_PATH) as _manifest_file:
    _MANIFEST = json.load(_manifest_file)

DATASET_NAME = _MANIFEST["dataset"]["name"]
DATASET_SOURCE = _MANIFEST["dataset"]["source"]
DATASET_VERSION = _MANIFEST["dataset"]["version"]
HASH_ALGORITHM = _MANIFEST["hash_algorithm"]

SPLITS = {
    name: SplitSpec(
        name=name,
        data_dir=REPO_ROOT / raw["relative_path"],
        task_count=raw["task_count"],
        sha256=raw["sha256"],
        search_allowed=raw["search_allowed"],
        outputs_locked=raw["outputs_locked"],
    )
    for name, raw in _MANIFEST["splits"].items()
}

SEARCH_TASK_IDS = frozenset(_MANIFEST["search_task_ids"])


class SplitPolicyError(ValueError):
    """Raised when search attempts to cross the locked split boundary."""


def get_split(name: str) -> SplitSpec:
    try:
        return SPLITS[name]
    except KeyError:
        raise ValueError(f"unknown dataset split {name!r}; expected one of {sorted(SPLITS)}") from None


def require_search_allowed(name: str) -> SplitSpec:
    spec = get_split(name)
    if not spec.search_allowed:
        raise SplitPolicyError(
            f"split {name!r} is locked for scoring only; its tasks and outputs may not enter search, "
            "training, warm starts, or checkpoint selection"
        )
    return spec
