"""Deterministic self-check for the object-POC harness (not part of `make
test` — `testpaths=["tests"]` excludes `research/`). Run manually:

    uv run pytest research/arc-object-poc/tests/ -q

Avoids the stochastic re-arc path; checks the executor and the hand programs
against the fixed real train/test pairs.
"""

import sys
from pathlib import Path

_POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_POC))
sys.path.insert(0, str(_POC.parents[1]))

from arc_env.task_loader import load_task  # noqa: E402

from gp_object import evaluate  # noqa: E402
from object_actions import ObjState, execute  # noqa: E402
from programs import PROGRAMS  # noqa: E402
from verify import run_program  # noqa: E402


def test_hand_programs_solve_real_pairs():
    """Gate 1 expressibility: every hand program is exact on all real train
    AND test pairs of its task."""
    for task_id, program in PROGRAMS.items():
        task = load_task(task_id)
        for pair in list(task.train) + list(task.test):
            assert run_program(program, pair.input) == pair.output, task_id


def test_hand_programs_score_perfect_fitness():
    for task_id, program in PROGRAMS.items():
        assert evaluate(program, load_task(task_id)) == (1.0, 1.0), task_id


def test_execute_invalid_preconditions_are_noops():
    grid = ((0, 1), (1, 0))
    state = ObjState(grid=grid)
    # combine before any select -> invalid, state unchanged
    new_state, _decoded, valid = execute(2, (0, 0, 0), state)
    assert valid is False
    assert new_state is state
    # out-of-range index -> invalid
    _s, _d, valid = execute(99, (0, 0, 0), state)
    assert valid is False


def test_split_then_select_is_valid():
    grid = ((3, 3), (0, 0), (3, 0), (0, 3))  # 4x2, top/bottom halves each 2x2
    state = ObjState(grid=grid)
    state, _d, valid = execute(0, (0, 0, 0), state)  # split top/bottom
    assert valid and state.region_a is not None and state.region_b is not None
    state, _d, valid = execute(1, (0, 0, 0), state)  # select color 0 in region a
    assert valid and state.set_a is not None
