"""Reproducible run provenance and locked-input integrity checks."""

import hashlib
from pathlib import Path

from arc_env.splits import (
    DATASET_NAME,
    DATASET_SOURCE,
    DATASET_VERSION,
    DEVELOPMENT,
    HASH_ALGORITHM,
    REPO_ROOT,
    get_split,
)

ACTION_LIBRARY_FILES = (
    REPO_ROOT / "arc_env/actions.py",
    REPO_ROOT / "arc_env/_dsl.py",
    REPO_ROOT / "third_party/arc-dsl/dsl.py",
    REPO_ROOT / "third_party/arc-dsl/arc_types.py",
    REPO_ROOT / "third_party/arc-dsl/constants.py",
)
PINNED_ACTION_LIBRARY_SHA256 = "95143b0797fb2f0068e416a8d5b7ae716a93015a08a81d3280338eb39671b77b"


class ProvenanceMismatch(RuntimeError):
    """A pinned dataset or action library changed without manifest review."""


def canonical_files_sha256(paths: list[Path] | tuple[Path, ...]) -> str:
    """Hash paths and bytes without depending on directory order or mtimes."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(REPO_ROOT).as_posix()):
        relative = path.relative_to(REPO_ROOT).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def dataset_sha256(split: str) -> tuple[str, int]:
    spec = get_split(split)
    files = tuple(spec.data_dir.glob("*.json"))
    return canonical_files_sha256(files), len(files)


def action_library_sha256() -> str:
    return canonical_files_sha256(ACTION_LIBRARY_FILES)


def verify_pinned_inputs(split: str = DEVELOPMENT) -> None:
    spec = get_split(split)
    actual_dataset_hash, actual_count = dataset_sha256(split)
    if (actual_dataset_hash, actual_count) != (spec.sha256, spec.task_count):
        raise ProvenanceMismatch(
            f"{split} dataset differs from dataset_splits.json: "
            f"expected {spec.task_count} tasks/{spec.sha256}, got {actual_count}/{actual_dataset_hash}"
        )
    actual_action_hash = action_library_sha256()
    if actual_action_hash != PINNED_ACTION_LIBRARY_SHA256:
        raise ProvenanceMismatch(
            "action library differs from its pinned hash: "
            f"expected {PINNED_ACTION_LIBRARY_SHA256}, got {actual_action_hash}"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_provenance(
    *,
    task_ids: list[str],
    seed: int | None,
    compute_budget: dict,
    macro_provenance: dict,
    split: str = DEVELOPMENT,
    checkpoint_selection: dict | None = None,
) -> dict:
    """Build a verified, JSON-safe scientific provenance manifest."""

    verify_pinned_inputs(split)
    spec = get_split(split)
    return {
        "schema_version": 1,
        "dataset": {
            "name": DATASET_NAME,
            "source": DATASET_SOURCE,
            "version": DATASET_VERSION,
            "split": split,
            "task_ids": sorted(task_ids),
            "task_count": spec.task_count,
            "sha256": spec.sha256,
            "hash_algorithm": HASH_ALGORITHM,
            "outputs_locked": spec.outputs_locked,
        },
        "action_library": {
            "sha256": PINNED_ACTION_LIBRARY_SHA256,
            "files": [path.relative_to(REPO_ROOT).as_posix() for path in ACTION_LIBRARY_FILES],
        },
        "seed": seed,
        "compute_budget": compute_budget,
        "macro_provenance": macro_provenance,
        "checkpoint_selection": checkpoint_selection or {
            "used": False,
            "locked_evaluation_outputs_used": False,
        },
    }


def no_seed_macros() -> dict:
    return {
        "seed_source": "none",
        "task_specific_seed": False,
        "action_catalog": "arc_env.actions.ACTIONS",
        "action_catalog_hash_recorded": True,
    }
