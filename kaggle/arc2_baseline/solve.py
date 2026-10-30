# MIT No Attribution (MIT-0)
#
# Copyright 2026 arc-agi contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Stdlib-only ARC-AGI-2 baseline: geometry + color bijection, with fallbacks.

Self-contained on purpose (no repo imports) so the whole file can be the body
of a Kaggle notebook. Adapted from ``legacy/baseline.py``. Writes the
two-attempts-per-test-input ``submission.json`` the competition expects:
``{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}``.

    python solve.py [--challenges PATH] [--out PATH]
"""
import argparse
import json
import os
import sys
from collections import Counter

Grid = list

CHALLENGES_NAME = "arc-agi_test_challenges.json"
INPUT_ROOT = "/kaggle/input"
DEFAULT_OUT = "/kaggle/working/submission.json"


def rot90(g):
    h = len(g)
    return [[g[h - 1 - y][x] for y in range(h)] for x in range(len(g[0]))]


def rot180(g):
    return [row[::-1] for row in g[::-1]]


def rot270(g):
    return rot90(rot180(g))


def flip_h(g):
    return [row[::-1] for row in g]


def flip_v(g):
    return g[::-1]


def transpose(g):
    return [[g[y][x] for y in range(len(g))] for x in range(len(g[0]))]


def identity(g):
    return g


GEOM_FUNCS = [identity, rot90, rot180, rot270, flip_h, flip_v, transpose]


def color_bijection(src, dst):
    if len(src) != len(dst) or len(src[0]) != len(dst[0]):
        return None
    mapping, used = {}, set()
    for rs, rd in zip(src, dst):
        for s, d in zip(rs, rd):
            if s in mapping:
                if mapping[s] != d:
                    return None
            else:
                if d in used:
                    return None
                mapping[s] = d
                used.add(d)
    return mapping


def apply_map(g, mp):
    return [[mp.get(c, c) for c in row] for row in g]


def synthesize(train):
    """Return a grid->grid callable consistent with every train pair, or None."""
    for gf in GEOM_FUNCS:
        if all(gf(p["input"]) == p["output"] for p in train):
            return gf
    for gf in GEOM_FUNCS:
        mp = color_bijection(gf(train[0]["input"]), train[0]["output"])
        if mp is None:
            continue
        if all(apply_map(gf(p["input"]), mp) == p["output"] for p in train):
            return lambda g, gf=gf, mp=mp: apply_map(gf(g), mp)
    return None


def constant_output(train):
    first = train[0]["output"]
    return first if all(p["output"] == first for p in train) else None


def most_common_color_canvas(train, test_input):
    """If every train output is a single-color grid of the input's shape, guess
    the input's most common color; a weak fallback, but better than a copy."""
    for p in train:
        out = p["output"]
        if len({c for r in out for c in r}) != 1:
            return None
    flat = [c for r in test_input for c in r]
    color = Counter(flat).most_common(1)[0][0]
    return [[color] * len(test_input[0]) for _ in test_input]


def find_challenges(root=INPUT_ROOT):
    """Locate the test file wherever the competition data is mounted."""
    for dirpath, _, files in os.walk(root):
        if CHALLENGES_NAME in files:
            return os.path.join(dirpath, CHALLENGES_NAME)
    listing = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
    raise FileNotFoundError(f"{CHALLENGES_NAME} not under {root}: {listing[:20]}")


def attempts(task):
    """Two distinct-as-possible candidate grids per test input."""
    train = task["train"]
    fn = synthesize(train)
    const = constant_output(train)
    out = []
    for t in task["test"]:
        g = t["input"]
        cands = []
        if fn is not None:
            try:
                cands.append(fn(g))
            except Exception:  # noqa: BLE001, S110 - a bad transform must not drop the task
                pass
        if const is not None:
            cands.append(const)
        canvas = most_common_color_canvas(train, g)
        if canvas is not None:
            cands.append(canvas)
        cands.append(g)
        cands.append(flip_h(g))
        uniq = []
        for c in cands:
            if c not in uniq:
                uniq.append(c)
        out.append({"attempt_1": uniq[0], "attempt_2": uniq[1] if len(uniq) > 1 else uniq[0]})
    return out


def solve_all(challenges):
    sub = {}
    for tid, task in challenges.items():
        try:
            sub[tid] = attempts(task)
        except Exception:  # noqa: BLE001
            # Never drop a task: a missing or malformed row invalidates the file.
            sub[tid] = [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]]
    return sub


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    with open(args.challenges or find_challenges()) as f:
        challenges = json.load(f)
    sub = solve_all(challenges)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(sub, f)
    print(f"wrote {len(sub)} tasks to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
