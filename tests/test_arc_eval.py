"""Scientific-validity contracts for the official-style ARC evaluator."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from arc_env.task_loader import Pair, Task
from arc_eval import (
    SubmissionValidationError,
    challenge_to_dict,
    evaluate_solver,
    evaluate_submission,
    report_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _task(task_id="fixture") -> Task:
    return Task(
        task_id=task_id,
        train=(
            Pair(input=((1, 0),), output=((0, 1),)),
            Pair(input=((2,), (0,)), output=((0,), (2,))),
        ),
        test=(
            Pair(input=((3, 0),), output=((0, 3),)),
            Pair(input=((4,), (0,), (4,)), output=((4, 4, 4),)),
        ),
    )


def test_solver_receives_demonstrations_and_test_inputs_but_no_test_outputs():
    task = _task()
    seen = []

    def solver(challenge):
        seen.append(challenge_to_dict(challenge))
        assert challenge.task_id == "fixture"
        assert challenge.train[0].output == ((0, 1),)
        assert challenge.test_inputs == (((3, 0),), ((4,), (0,), (4,)))
        assert not hasattr(challenge, "test")
        return [
            {"attempt_1": [[0, 3]]},
            {"attempt_1": [[4, 4, 4]]},
        ]

    report = evaluate_solver([task], solver)

    assert report["pass_at_1"] == 1.0
    assert seen == [{
        "task_id": "fixture",
        "train": [
            {"input": [[1, 0]], "output": [[0, 1]]},
            {"input": [[2], [0]], "output": [[0], [2]]},
        ],
        "test": [{"input": [[3, 0]]}, {"input": [[4], [0], [4]]}],
    }]


def test_pass_at_1_and_2_score_each_test_input_in_official_order():
    task = _task()
    submission = {
        "fixture": [
            {"attempt_1": [[0, 3]], "attempt_2": [[9]]},
            {"attempt_1": [[0]], "attempt_2": [[4, 4, 4]]},
        ]
    }

    report = evaluate_submission([task], submission)

    assert report["n_tasks"] == 1
    assert report["n_test_cases"] == 2
    assert report["correct_test_cases_at_1"] == 1
    assert report["correct_test_cases_at_2"] == 2
    assert report["pass_at_1"] == 0.5
    assert report["pass_at_2"] == 1.0
    assert report["correct_tasks_at_1"] == 0
    assert report["correct_tasks_at_2"] == 1
    assert report["task_pass_at_1"] == 0.0
    assert report["task_pass_at_2"] == 1.0
    assert report["tasks"]["fixture"]["test_cases"] == [
        {
            "attempts_submitted": 2,
            "correct_attempt": 1,
            "pass_at_1": True,
            "pass_at_2": True,
            "test_index": 0,
        },
        {
            "attempts_submitted": 2,
            "correct_attempt": 2,
            "pass_at_1": False,
            "pass_at_2": True,
            "test_index": 1,
        },
    ]


def test_one_attempt_is_valid_and_pass_at_2_does_not_invent_a_second_guess():
    task = _task()
    submission = {
        "fixture": [
            {"attempt_1": [[0, 3]]},
            {"attempt_1": [[0]]},
        ]
    }

    report = evaluate_submission([task], submission)

    assert report["pass_at_1"] == report["pass_at_2"] == 0.5
    assert report["tasks"]["fixture"]["test_cases"][0]["attempts_submitted"] == 1


def test_wrong_shape_is_a_valid_but_incorrect_prediction():
    task = Task(
        task_id="variable-shape",
        train=(Pair(input=((1,),), output=((1, 1), (1, 1))),),
        test=(Pair(input=((2,),), output=((2, 2), (2, 2))),),
    )

    report = evaluate_submission(
        [task],
        {"variable-shape": [{"attempt_1": [[2]], "attempt_2": [[2, 2], [2, 2]]}]},
    )

    assert report["pass_at_1"] == 0.0
    assert report["pass_at_2"] == 1.0


@pytest.mark.parametrize(
    ("submission", "message"),
    [
        ([], "submission must be an object"),
        ({}, "missing task ids: fixture"),
        ({"fixture": [], "extra": []}, "unexpected task ids: extra"),
        ({"fixture": [{}, {}]}, "attempt_1 is required"),
        ({"fixture": [{"attempt_2": [[0]]}, {"attempt_1": [[0]]}]}, "attempt_1 is required"),
        ({"fixture": [{"attempt_1": [[0]], "attempt_3": [[0]]}, {"attempt_1": [[0]]}]}, "unexpected keys"),
        ({"fixture": [{"attempt_1": []}, {"attempt_1": [[0]]}]}, "grid must have 1..30 rows"),
        ({"fixture": [{"attempt_1": [[0], [0, 1]]}, {"attempt_1": [[0]]}]}, "grid must be rectangular"),
        ({"fixture": [{"attempt_1": [[True]]}, {"attempt_1": [[0]]}]}, "cell must be an integer"),
        ({"fixture": [{"attempt_1": [[10]]}, {"attempt_1": [[0]]}]}, "cell must be in 0..9"),
    ],
)
def test_malformed_submissions_fail_with_stable_diagnostics(submission, message):
    with pytest.raises(SubmissionValidationError, match=message):
        evaluate_submission([_task()], submission)


def test_wrong_number_of_test_predictions_is_rejected():
    with pytest.raises(SubmissionValidationError, match="expected 2 test predictions, got 1"):
        evaluate_submission([_task()], {"fixture": [{"attempt_1": [[0, 3]]}]})


def test_report_json_is_canonical_and_contains_no_grids():
    tasks = [_task("z-task"), _task("a-task")]
    submission = {
        task.task_id: [
            {"attempt_1": [[0, 3]]},
            {"attempt_1": [[4, 4, 4]]},
        ]
        for task in reversed(tasks)
    }

    encoded = report_json(evaluate_submission(tasks, submission))

    assert encoded == report_json(evaluate_submission(reversed(tasks), submission))
    assert encoded.endswith("\n")
    assert "[[" not in encoded
    assert list(json.loads(encoded)["tasks"]) == ["a-task", "z-task"]


def test_cli_emits_the_same_deterministic_report(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "fixture.json").write_text(json.dumps({
        "train": [{"input": [[1, 0]], "output": [[0, 1]]}],
        "test": [
            {"input": [[3, 0]], "output": [[0, 3]]},
            {"input": [[4], [0], [4]], "output": [[4, 4, 4]]},
        ],
    }))
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(json.dumps({
        "fixture": [
            {"attempt_1": [[0, 3]]},
            {"attempt_1": [[4, 4, 4]]},
        ]
    }))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "arc_eval",
            "--tasks-dir",
            str(tasks_dir),
            "--submission",
            str(submission_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["pass_at_1"] == 1.0
    assert result.stdout == report_json(json.loads(result.stdout))
