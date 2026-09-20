"""Official-style pass@1/pass@2 evaluation for ARC-AGI.

The solver-facing :class:`Challenge` contains demonstration input/output
pairs and test inputs only. Hidden test outputs remain in the evaluator and
are consulted only after the solver has returned one or two concrete grids
per test input. This is an API isolation boundary, not a hostile-process
sandbox: solver code running in this repository could still open dataset
files itself, so benchmark orchestration must separately control filesystem
access when evaluating untrusted or leakage-prone solvers.

Submission shape follows the ARC Prize convention::

    {
      "task_id": [
        {"attempt_1": [[...]], "attempt_2": [[...]]}
      ]
    }

Unlike the competition file format, ``attempt_2`` may be omitted so the same
scorer can report both a genuine one-attempt baseline and a two-attempt one.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_env.task_loader import Pair, Task

Grid = tuple[tuple[int, ...], ...]


class SubmissionValidationError(ValueError):
    """The submission cannot be scored without guessing its intended shape."""


@dataclass(frozen=True)
class Challenge:
    """The complete information a conforming ARC solver may observe."""

    task_id: str
    train: tuple[Pair, ...]
    test_inputs: tuple[Grid, ...]


def _grid_to_lists(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def challenge_from_task(task: Task) -> Challenge:
    """Drop hidden test outputs before a task crosses the solver boundary."""

    return Challenge(
        task_id=task.task_id,
        train=task.train,
        test_inputs=tuple(pair.input for pair in task.test),
    )


def challenge_to_dict(challenge: Challenge) -> dict:
    """Return the standard public challenge JSON shape, without test outputs."""

    return {
        "task_id": challenge.task_id,
        "train": [
            {"input": _grid_to_lists(pair.input), "output": _grid_to_lists(pair.output)}
            for pair in challenge.train
        ],
        "test": [{"input": _grid_to_lists(grid)} for grid in challenge.test_inputs],
    }


def _task_map(tasks: Iterable[Task]) -> dict[str, Task]:
    by_id: dict[str, Task] = {}
    for task in tasks:
        if task.task_id in by_id:
            raise ValueError(f"duplicate task id: {task.task_id}")
        if not task.test:
            raise ValueError(f"task {task.task_id} has no test cases")
        by_id[task.task_id] = task
    if not by_id:
        raise ValueError("at least one task is required")
    return by_id


def _parse_grid(value: Any, path: str) -> Grid:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 30:
        raise SubmissionValidationError(f"{path}: grid must have 1..30 rows")

    width = None
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= 30:
            raise SubmissionValidationError(f"{path}[{row_index}]: row must have 1..30 cells")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise SubmissionValidationError(f"{path}: grid must be rectangular")

        parsed_row = []
        for column_index, cell in enumerate(row):
            cell_path = f"{path}[{row_index}][{column_index}]"
            if type(cell) is not int:
                raise SubmissionValidationError(f"{cell_path}: cell must be an integer")
            if not 0 <= cell <= 9:
                raise SubmissionValidationError(f"{cell_path}: cell must be in 0..9")
            parsed_row.append(cell)
        rows.append(tuple(parsed_row))

    return tuple(rows)


def _parse_submission(
    submission: Any,
    tasks: Mapping[str, Task],
) -> dict[str, tuple[tuple[Grid, ...], ...]]:
    if not isinstance(submission, Mapping):
        raise SubmissionValidationError("submission must be an object keyed by task id")

    expected_ids = set(tasks)
    submitted_ids = set(submission)
    missing = sorted(expected_ids - submitted_ids)
    extra = sorted(submitted_ids - expected_ids)
    coverage_errors = []
    if missing:
        coverage_errors.append(f"missing task ids: {', '.join(missing)}")
    if extra:
        coverage_errors.append(f"unexpected task ids: {', '.join(extra)}")
    if coverage_errors:
        raise SubmissionValidationError("submission: " + "; ".join(coverage_errors))

    parsed: dict[str, tuple[tuple[Grid, ...], ...]] = {}
    for task_id in sorted(tasks):
        raw_test_predictions = submission[task_id]
        if not isinstance(raw_test_predictions, (list, tuple)):
            raise SubmissionValidationError(f"submission[{task_id!r}] must be a list")
        expected_count = len(tasks[task_id].test)
        if len(raw_test_predictions) != expected_count:
            raise SubmissionValidationError(
                f"submission[{task_id!r}]: expected {expected_count} test predictions, "
                f"got {len(raw_test_predictions)}"
            )

        test_predictions = []
        for test_index, raw_attempts in enumerate(raw_test_predictions):
            path = f"submission[{task_id!r}][{test_index}]"
            if not isinstance(raw_attempts, Mapping):
                raise SubmissionValidationError(f"{path} must be an object")
            keys = set(raw_attempts)
            if "attempt_1" not in keys:
                raise SubmissionValidationError(f"{path}: attempt_1 is required")
            unexpected = sorted(keys - {"attempt_1", "attempt_2"})
            if unexpected:
                raise SubmissionValidationError(
                    f"{path}: unexpected keys: {', '.join(str(key) for key in unexpected)}"
                )

            attempts = [_parse_grid(raw_attempts["attempt_1"], f"{path}.attempt_1")]
            if "attempt_2" in raw_attempts:
                attempts.append(_parse_grid(raw_attempts["attempt_2"], f"{path}.attempt_2"))
            test_predictions.append(tuple(attempts))
        parsed[task_id] = tuple(test_predictions)

    return parsed


def evaluate_submission(tasks: Iterable[Task], submission: Any) -> dict:
    """Validate and exactly score one/two attempts for every hidden test grid.

    ``pass_at_1`` and ``pass_at_2`` are test-case accuracies, matching the
    competition's output-level metric. ``task_pass_at_*`` are stricter
    diagnostics: a task passes only when all of its test inputs pass.
    """

    tasks_by_id = _task_map(tasks)
    parsed = _parse_submission(submission, tasks_by_id)

    correct_cases_at_1 = 0
    correct_cases_at_2 = 0
    correct_tasks_at_1 = 0
    correct_tasks_at_2 = 0
    task_results = {}

    for task_id in sorted(tasks_by_id):
        task = tasks_by_id[task_id]
        case_results = []
        for test_index, (pair, attempts) in enumerate(zip(task.test, parsed[task_id], strict=True)):
            pass_at_1 = attempts[0] == pair.output
            pass_at_2 = any(attempt == pair.output for attempt in attempts[:2])
            correct_attempt = next(
                (index for index, attempt in enumerate(attempts, start=1) if attempt == pair.output),
                None,
            )
            correct_cases_at_1 += int(pass_at_1)
            correct_cases_at_2 += int(pass_at_2)
            case_results.append({
                "attempts_submitted": len(attempts),
                "correct_attempt": correct_attempt,
                "pass_at_1": pass_at_1,
                "pass_at_2": pass_at_2,
                "test_index": test_index,
            })

        task_at_1 = all(case["pass_at_1"] for case in case_results)
        task_at_2 = all(case["pass_at_2"] for case in case_results)
        correct_tasks_at_1 += int(task_at_1)
        correct_tasks_at_2 += int(task_at_2)
        task_results[task_id] = {
            "pass_at_1": task_at_1,
            "pass_at_2": task_at_2,
            "test_cases": case_results,
        }

    n_tasks = len(tasks_by_id)
    n_test_cases = sum(len(task.test) for task in tasks_by_id.values())
    return {
        "correct_tasks_at_1": correct_tasks_at_1,
        "correct_tasks_at_2": correct_tasks_at_2,
        "correct_test_cases_at_1": correct_cases_at_1,
        "correct_test_cases_at_2": correct_cases_at_2,
        "n_tasks": n_tasks,
        "n_test_cases": n_test_cases,
        "pass_at_1": correct_cases_at_1 / n_test_cases,
        "pass_at_2": correct_cases_at_2 / n_test_cases,
        "schema_version": 1,
        "task_pass_at_1": correct_tasks_at_1 / n_tasks,
        "task_pass_at_2": correct_tasks_at_2 / n_tasks,
        "tasks": task_results,
    }


def evaluate_solver(
    tasks: Iterable[Task],
    solver: Callable[[Challenge], Sequence[Mapping[str, Any]]],
) -> dict:
    """Call ``solver`` with blind challenges, then score its returned grids."""

    tasks_by_id = _task_map(tasks)
    submission = {
        task_id: solver(challenge_from_task(tasks_by_id[task_id]))
        for task_id in sorted(tasks_by_id)
    }
    return evaluate_submission(tasks_by_id.values(), submission)


def report_json(report: Mapping[str, Any]) -> str:
    """Canonical JSON: stable key order, separators, and trailing newline."""

    return json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"


def _load_pair(raw: Any, path: str) -> Pair:
    if not isinstance(raw, Mapping) or set(raw) != {"input", "output"}:
        raise ValueError(f"{path} must contain exactly input and output")
    try:
        input_grid = _parse_grid(raw["input"], f"{path}.input")
        output_grid = _parse_grid(raw["output"], f"{path}.output")
    except SubmissionValidationError as exc:
        raise ValueError(str(exc)) from exc
    return Pair(input=input_grid, output=output_grid)


def load_tasks(tasks_dir: Path) -> tuple[Task, ...]:
    """Load scorer-owned task files containing hidden test outputs."""

    paths = sorted(tasks_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no task JSON files found in {tasks_dir}")

    tasks = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, Mapping) or set(raw) != {"train", "test"}:
            raise ValueError(f"{path}: task must contain exactly train and test")
        if not isinstance(raw["train"], list) or not raw["train"]:
            raise ValueError(f"{path}: train must be a non-empty list")
        if not isinstance(raw["test"], list) or not raw["test"]:
            raise ValueError(f"{path}: test must be a non-empty list")
        tasks.append(Task(
            task_id=path.stem,
            train=tuple(_load_pair(pair, f"{path}.train[{index}]") for index, pair in enumerate(raw["train"])),
            test=tuple(_load_pair(pair, f"{path}.test[{index}]") for index, pair in enumerate(raw["test"])),
        ))
    return tuple(tasks)
