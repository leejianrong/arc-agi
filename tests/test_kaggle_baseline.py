import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "arc2_solve", Path(__file__).resolve().parent.parent / "kaggle/arc2_baseline/solve.py"
)
solve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(solve)


def _task(pairs, test_inputs):
    return {
        "train": [{"input": i, "output": o} for i, o in pairs],
        "test": [{"input": g} for g in test_inputs],
    }


def test_geometry_and_color_map_solved():
    pairs = [([[1, 2], [3, 4]], [[2, 1], [4, 3]])]
    rows = solve.attempts(_task(pairs, [[[5, 6], [7, 8]]]))
    assert rows[0]["attempt_1"] == [[6, 5], [8, 7]]


def test_unsolvable_still_yields_two_attempts_per_test_input():
    sub = solve.solve_all({"abcd1234": _task([([[1]], [[1, 1]])], [[[1]], [[2]]])})
    assert len(sub["abcd1234"]) == 2
    assert all({"attempt_1", "attempt_2"} <= row.keys() for row in sub["abcd1234"])
