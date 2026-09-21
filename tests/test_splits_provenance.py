import json
import subprocess
import sys

import pytest

from arc_env.provenance import (
    PINNED_ACTION_LIBRARY_SHA256,
    action_library_sha256,
    build_run_provenance,
    dataset_sha256,
    no_seed_macros,
)
from arc_env.splits import DEVELOPMENT, LOCKED_EVALUATION, SPLITS, SplitPolicyError
from arc_env.tasks import load_search_task

TASK_ID = "67a3c6ac"


def test_pinned_dataset_and_action_hashes_match_checkout():
    for split, spec in SPLITS.items():
        assert dataset_sha256(split) == (spec.sha256, spec.task_count)
    assert action_library_sha256() == PINNED_ACTION_LIBRARY_SHA256


def test_search_task_exposes_no_test_outputs():
    task = load_search_task(TASK_ID)
    assert task.split == DEVELOPMENT
    assert task.train
    assert task.test_inputs
    assert not hasattr(task, "test")


def test_locked_evaluation_split_is_rejected_by_search_loader():
    with pytest.raises(SplitPolicyError, match="locked for scoring only"):
        load_search_task(TASK_ID, split=LOCKED_EVALUATION)


def test_run_provenance_records_scientific_controls():
    provenance = build_run_provenance(
        task_ids=[TASK_ID],
        seed=17,
        compute_budget={"unit": "environment_steps", "planned": 1024},
        macro_provenance=no_seed_macros(),
        checkpoint_selection={
            "used": True,
            "split": DEVELOPMENT,
            "partition": "train_examples",
            "locked_evaluation_outputs_used": False,
        },
    )
    assert provenance["dataset"]["sha256"] == SPLITS[DEVELOPMENT].sha256
    assert provenance["action_library"]["sha256"] == PINNED_ACTION_LIBRARY_SHA256
    assert provenance["seed"] == 17
    assert provenance["compute_budget"]["planned"] == 1024
    assert provenance["checkpoint_selection"]["locked_evaluation_outputs_used"] is False


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        ("train", "arc_env.task_loader"),
        ("train", "object_env.programs"),
        ("object_env.search", "object_env.programs"),
    ],
)
def test_search_imports_do_not_import_known_solver_fixture_modules(module, forbidden):
    code = f"import importlib, json, sys; importlib.import_module({module!r}); print(json.dumps({forbidden!r} in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) is False
