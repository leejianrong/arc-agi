"""Score an ARC submission without exposing hidden outputs to solver code."""

import argparse
import json
import sys
from pathlib import Path

from arc_eval.evaluator import evaluate_submission, load_tasks, report_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True, help="Directory of scorer-owned task JSON files.")
    parser.add_argument("--submission", type=Path, required=True, help="Official-style submission JSON file.")
    args = parser.parse_args(argv)

    try:
        tasks = load_tasks(args.tasks_dir)
        with open(args.submission, encoding="utf-8") as file:
            submission = json.load(file)
        report = evaluate_submission(tasks, submission)
    except (OSError, ValueError) as exc:
        error = {"error": {"message": str(exc), "type": type(exc).__name__}}
        sys.stderr.write(report_json(error))
        return 2

    sys.stdout.write(report_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
