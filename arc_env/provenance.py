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

# The object-grammar track's own executable action library (SLICES.md V6):
# `trainers.gp_object` never imports `arc_env.actions` at all, so a run's
# provenance should hash *this* set of files, not the flat one above -
# `train.py`'s `train_gp_object` passes `object_action_library()` as
# `build_run_provenance`'s `action_library` override. Not pinned/verified
# like `ACTION_LIBRARY_FILES` (no `docs/scientific-validity.md` manifest
# commitment for it yet - `object_env` is still under active development,
# unlike the shipped flat action space).
OBJECT_ACTION_LIBRARY_FILES = (
    REPO_ROOT / "object_env/actions.py",
    REPO_ROOT / "object_env/grammar.py",
    REPO_ROOT / "object_env/objects.py",
    REPO_ROOT / "object_env/colors.py",
    REPO_ROOT / "object_env/state.py",
    REPO_ROOT / "object_env/types.py",
    REPO_ROOT / "arc_env/_dsl.py",
    REPO_ROOT / "third_party/arc-dsl/dsl.py",
    REPO_ROOT / "third_party/arc-dsl/arc_types.py",
    REPO_ROOT / "third_party/arc-dsl/constants.py",
)


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


def object_action_library() -> dict:
    """The object-grammar track's `action_library` provenance block -
    computed fresh each call (unpinned), unlike the flat space's hardcoded
    `PINNED_ACTION_LIBRARY_SHA256`."""

    return {
        "sha256": canonical_files_sha256(OBJECT_ACTION_LIBRARY_FILES),
        "files": [path.relative_to(REPO_ROOT).as_posix() for path in OBJECT_ACTION_LIBRARY_FILES],
    }


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
    action_library: dict | None = None,
) -> dict:
    """Build a verified, JSON-safe scientific provenance manifest.

    `action_library` overrides the default flat-space block (e.g.
    `object_action_library()` for `trainers.gp_object` runs, whose
    executable action library is `object_env`, not `arc_env.actions`) -
    `verify_pinned_inputs` still checks the flat library's pinned hash
    regardless (a cheap, always-relevant repo-integrity sanity check, not a
    claim about which library the caller actually executed)."""

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
        "action_library": action_library or {
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


def no_seed_macros(action_catalog: str = "arc_env.actions.ACTIONS") -> dict:
    return {
        "seed_source": "none",
        "task_specific_seed": False,
        "action_catalog": action_catalog,
        "action_catalog_hash_recorded": True,
    }
