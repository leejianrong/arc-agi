# MIT No Attribution (MIT-0) — see solve.py for the full notice.
"""Score a Kaggle-format submission against the public evaluation solutions.

    python score_local.py SUBMISSION CHALLENGES SOLUTIONS

Task score = fraction of test inputs where either attempt matches exactly.
"""
import json
import sys


def score(sub, challenges, solutions):
    total = 0.0
    for tid, task in challenges.items():
        rows = sub[tid]
        assert len(rows) == len(task["test"]), f"{tid}: wrong number of test rows"
        hits = sum(
            1
            for row, sol in zip(rows, solutions[tid])
            if sol in (row["attempt_1"], row["attempt_2"])
        )
        total += hits / len(task["test"])
    return total, len(challenges)


if __name__ == "__main__":
    loaded = []
    for path in sys.argv[1:4]:
        with open(path) as f:
            loaded.append(json.load(f))
    sub, ch, sol = loaded
    s, n = score(sub, ch, sol)
    print(f"{s:.2f} / {n} tasks = {100 * s / n:.2f}%")
