"""E2E regression (SLICES.md V5): replay every hand-written object-space program
through `object_env` and check it reproduces the task's exact train+test output
— the object-track analogue of `tests/test_dsl_regression.py`, and the concrete
evidence the typed grammar spans task families (set-op + move/recolor/crop/
canvas/transform), not just the cluster it was born on.
"""

import pytest

from arc_env.task_loader import load_task
from object_env.programs import PROGRAMS, SET_OP_TASK_IDS, TARGET_TASK_IDS, build


@pytest.mark.parametrize("task_id", sorted(PROGRAMS))
def test_program_reproduces_expected_output(task_id):
    program = build(task_id)  # constructs + type-checks
    task = load_task(task_id)
    for pair in (*task.train, *task.test):
        assert program.run(pair.input) == pair.output


def test_basket_spans_families():
    """The fixture must actually reach past the 8 set-op tasks — the whole point
    of V5 (fork 2, moderate basket)."""
    assert len(SET_OP_TASK_IDS) == 8
    non_set_op = [t for t in TARGET_TASK_IDS if t not in SET_OP_TASK_IDS]
    assert len(non_set_op) >= 6  # move / recolor / crop / canvas / transform
