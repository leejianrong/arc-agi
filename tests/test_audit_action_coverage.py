"""Unit tests for `scripts/audit_action_coverage.py`'s pure logic (AST
parsing, bucketing) plus one smoke test against the real repo state."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_action_coverage import (
    ACTIONS_PATH,
    SOLVERS_PATH,
    bucket_tasks,
    curated_primitives_by_name,
    solver_primitive_usage,
)


def test_curated_primitives_by_name_finds_every_dsl_dot_reference():
    src = """
from arc_env._dsl import dsl

def _fill_cell(grid, color, row, col):
    return dsl.fill(grid, color, frozenset({(row, col)}))

ACTIONS = [Action("rot90", dsl.rot90)]
"""
    assert curated_primitives_by_name(src) == {"fill", "rot90"}


def test_solver_primitive_usage_only_counts_known_dsl_names():
    src = """
def solve_abc123(I):
    x1 = vmirror(I)
    O = replace(x1, 1, 2)
    return O

def solve_def456(I):
    O = made_up_local_helper(I)
    return O

def not_a_solver(I):
    return rot90(I)
"""
    dsl_names = {"vmirror", "replace", "rot90"}
    usage = solver_primitive_usage(src, dsl_names)
    assert usage == {"abc123": {"vmirror", "replace"}, "def456": set()}


def test_bucket_tasks_sorts_by_gap_size_and_flags_higher_order_first():
    remaining = {
        "excluded_task": {"mapply", "vmirror"},
        "free_task": {"vmirror", "replace"},
        "one_new_task": {"vmirror", "ofcolor"},
        "two_new_task": {"vmirror", "ofcolor", "first"},
    }
    higher_order = {"mapply"}
    curated_by_name = {"vmirror", "replace"}

    buckets = bucket_tasks(remaining, higher_order, curated_by_name)

    assert buckets["excluded"] == {"excluded_task": {"mapply"}}
    assert buckets["free_by_name"] == {"free_task": {"vmirror", "replace"}}
    assert buckets["one_new"] == {"one_new_task": {"ofcolor"}}
    assert buckets["two_new"] == {"two_new_task": {"ofcolor", "first"}}
    assert buckets["three_plus_new"] == {}


def test_bucket_tasks_prioritizes_the_higher_order_check_over_a_name_gap():
    # A task using both a higher-order primitive AND a genuinely new
    # first-order one should land in "excluded", not "one_new" - there's no
    # point ranking it by primitive-gap size when it's unreachable regardless.
    remaining = {"t": {"mapply", "brand_new_primitive"}}
    buckets = bucket_tasks(remaining, higher_order={"mapply"}, curated_by_name=set())
    assert buckets["excluded"] == {"t": {"mapply"}}
    assert all(len(buckets[b]) == 0 for b in ("free_by_name", "one_new", "two_new", "three_plus_new"))


def test_smoke_every_curated_task_is_excluded_from_remaining():
    from arc_env.task_loader import CURATED_TASK_IDS

    usage = solver_primitive_usage(SOLVERS_PATH.read_text(), dsl_func_names=set(_all_dsl_names()))
    remaining = {t: p for t, p in usage.items() if t not in CURATED_TASK_IDS}

    assert len(usage) == 400  # every ARC-AGI-1 training task has a solvers.py entry
    assert len(remaining) == len(usage) - len(CURATED_TASK_IDS)
    assert not (set(remaining) & set(CURATED_TASK_IDS))


def _all_dsl_names() -> set:
    import inspect

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "arc-dsl"))
    import dsl

    return {name for name, obj in vars(dsl).items() if inspect.isfunction(obj) and obj.__module__ == "dsl"}


def test_actions_py_is_parseable_and_nonempty():
    # Guards against the audit silently reporting 0 curated primitives if
    # arc_env/actions.py's dsl-reference style ever changes.
    assert len(curated_primitives_by_name(ACTIONS_PATH.read_text())) > 20
