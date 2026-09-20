"""Leak-resistant, exact-match evaluation for ARC-AGI submissions."""

from arc_eval.evaluator import (
    Challenge,
    SubmissionValidationError,
    challenge_from_task,
    challenge_to_dict,
    evaluate_solver,
    evaluate_submission,
    load_tasks,
    report_json,
)

__all__ = [
    "Challenge",
    "SubmissionValidationError",
    "challenge_from_task",
    "challenge_to_dict",
    "evaluate_solver",
    "evaluate_submission",
    "load_tasks",
    "report_json",
]
