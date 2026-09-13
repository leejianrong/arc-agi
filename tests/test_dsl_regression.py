"""Integration test (SLICES.md V1 + V3): replay each curated task's
known-correct `arc-dsl` solver program through the env's action executor and
check it reproduces the task's exact expected output, for every train and
test pair - now covering V3's variable-shape tasks (via `canvas`/`commit`)
alongside V1's same-shape ones.

This is the "riskiest-mechanism-first" check for V1 (PLAN.md Testing
approach / ADR-0001 consequences): it's free ground truth, since the ARC
dataset + `arc-dsl`'s solvers already establish what "correct" is.
"""

import pytest

from arc_env import actions
from arc_env.task_loader import CURATED_TASK_IDS, load_task


def replay(task_id: str, grid: tuple) -> tuple:
    # ADR-0020: a dual-slot selection dict (`0`="a"/`1`="b") plus its
    # matching region-tag dict, mirroring `actions.execute`'s own dispatch.
    selected = {0: None, 1: None}
    selected_region = {0: None, 1: None}
    for primitive_name, args in CURATED_TASK_IDS[task_id]:
        action = actions.ACTIONS[actions.ACTION_BY_NAME[primitive_name]]
        if action.kind == "select":
            selected[0] = action.fn(grid, *args)
        elif action.kind == "select_region":
            slot = args[0]
            selected[slot] = action.fn(grid, *args[1:])
            selected_region[slot] = args[1]
        elif action.kind == "combine_selection":
            op = args[0]
            selected[0] = action.fn(op, selected[0], selected[1])
            selected[1] = None
            selected_region[1] = None
        elif action.kind == "act_on_selection":
            grid = action.fn(grid, selected[0], *args)
        elif action.kind == "act_on_region_selection":
            grid = action.fn(grid, selected[0], selected_region[0], *args)
        elif action.kind == "fill_slot_onto_region":
            slot = args[0]
            grid = action.fn(grid, selected[slot], *args[1:])
        else:
            grid = action.fn(grid, *args)
    return grid


@pytest.mark.parametrize("task_id", sorted(CURATED_TASK_IDS))
def test_solver_reproduces_expected_output(task_id):
    task = load_task(task_id)
    for pair in (*task.train, *task.test):
        assert replay(task_id, pair.input) == pair.output


@pytest.mark.parametrize("task_id", sorted(CURATED_TASK_IDS))
def test_solver_replays_through_env_action_executor(task_id):
    """Same check, but through `actions.execute` (raw-args + validity path)
    rather than calling the primitive directly - this is what the rollout
    script and any future trainer actually go through."""

    task = load_task(task_id)
    for pair in (*task.train, *task.test):
        grid = pair.input
        selected = {0: None, 1: None}
        selected_region = {0: None, 1: None}
        for primitive_name, args in CURATED_TASK_IDS[task_id]:
            primitive_index = actions.ACTION_BY_NAME[primitive_name]
            action = actions.ACTIONS[primitive_index]
            raw_args = tuple(_encode(spec, value) for spec, value in zip(action.args, args))
            grid, selected, selected_region, decoded, valid = actions.execute(
                primitive_index, raw_args, grid, selected, selected_region
            )
            assert valid, f"{task_id}: {primitive_name}{args} was rejected as invalid"
            assert tuple(decoded.values()) == args
        assert grid == pair.output


def _encode(spec: actions.ArgSpec, value: int) -> int:
    """Inverse of `spec.decode`, for driving `actions.execute`'s raw-args
    interface from a solver's real argument values in this test only."""

    if spec.kind == "color":
        return value % 10
    if spec.kind == "factor":
        return value - 2
    if spec.kind == "dim":
        return value - 1
    if spec.kind == "size":
        return value - 1  # ADR-0021: same decode as "dim" (raw + 1)
    if spec.kind in ("slot", "region", "combine_op"):
        return value  # ADR-0020: all decode via `raw % N`, identity here too
    return value  # "coord"/"direction": decode is the identity (mod 4 for direction)
