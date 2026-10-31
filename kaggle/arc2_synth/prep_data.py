# MIT No Attribution (MIT-0) — see ../arc2_baseline/solve.py for the full notice.
"""Turn a directory of per-task ARC JSON files (arcprize/ARC-AGI-2 layout) into
Kaggle-format challenges + solutions files.

    python prep_data.py DIR CHALLENGES_OUT SOLUTIONS_OUT
"""
import glob
import json
import os
import sys


def convert(directory):
    challenges, solutions = {}, {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        tid = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            t = json.load(f)
        challenges[tid] = {"train": t["train"], "test": [{"input": x["input"]} for x in t["test"]]}
        solutions[tid] = [x["output"] for x in t["test"]]
    return challenges, solutions


if __name__ == "__main__":
    c, s = convert(sys.argv[1])
    for path, obj in ((sys.argv[2], c), (sys.argv[3], s)):
        with open(path, "w") as f:
            json.dump(obj, f)
    print(f"{len(c)} tasks", file=sys.stderr)
