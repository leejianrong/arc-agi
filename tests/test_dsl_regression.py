"""V1 integration test (SLICES.md): replay each curated task's known-correct
`arc-dsl` solver program through the env's action executor and check it
reproduces the task's exact expected output, for every train and test pair.

This is the "riskiest-mechanism-first" check for V1 (PLAN.md Testing
approach / ADR-0001 consequences): it's free ground truth, since the ARC
dataset + `arc-dsl`'s solvers already establish what "correct" is.
"""

import pytest

from arc_env import actions
from arc_env.task_loader import CURATED_TASK_IDS, load_task


def replay(task_id: str, grid: tuple) -> tuple:
    for primitive_name, args in CURATED_TASK_IDS[task_id]:
        action = actions.ACTIONS[actions.ACTION_BY_NAME[primitive_name]]
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
        for primitive_name, args in CURATED_TASK_IDS[task_id]:
            primitive_index = actions.ACTION_BY_NAME[primitive_name]
            action = actions.ACTIONS[primitive_index]
            raw_args = tuple(_encode(spec, value) for spec, value in zip(action.args, args))
            grid, decoded, valid = actions.execute(primitive_index, raw_args, grid)
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
    return value  # "coord": decode is the identity
