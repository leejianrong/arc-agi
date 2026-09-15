"""Gate 2, part 2 — the same object actions, but searched through a *typed
grammar* instead of a free-form flat genome.

`gp_object.py` searches programs as an unstructured list of `(action, args)`
genes (exactly how the shipped `trainers/gp` genome works) and finds 0/8 — the
dense-similarity fitness plateaus at a deceptive ~0.88 local optimum. This
module fixes the *skeleton* the set-op family always follows —
`split -> select -> select -> combine -> paint` — and searches only the typed
args (axis, two colors, set-op, and the paint colors). That collapses the
search to a few thousand evaluations per task.

The contrast is the POC's headline: the object representation is only
searchable once the action space is *structured*. Flatness, not pixels-vs-
objects, is the real blocker.
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arc_env.task_loader import load_task  # noqa: E402

from gp_object import evaluate  # noqa: E402
from object_actions import ACTION_BY_NAME  # noqa: E402
from programs import TARGET_TASK_IDS  # noqa: E402

_SPLIT = ACTION_BY_NAME["split"]
_SELECT = ACTION_BY_NAME["select_color"]
_COMBINE = ACTION_BY_NAME["combine"]
_PAINT_CANVAS = ACTION_BY_NAME["paint_canvas"]
_PAINT_REGION = ACTION_BY_NAME["paint_onto_region"]


def _sample(rng):
    """Draw one program from the set-op grammar: a fixed 5-step skeleton with
    randomized typed args, and a choice of the two paint verbs."""
    axis = rng.randint(0, 1)
    c0, c1 = rng.randint(0, 9), rng.randint(0, 9)
    op = rng.randint(0, 3)
    if rng.random() < 0.5:
        paint = (_PAINT_CANVAS, (rng.randint(0, 9), rng.randint(0, 9), 0))
    else:
        paint = (_PAINT_REGION, (rng.randint(0, 9), 0, 0))
    return [
        (_SPLIT, (axis, 0, 0)),
        (_SELECT, (0, 0, c0)),
        (_SELECT, (1, 1, c1)),
        (_COMBINE, (op, 0, 0)),
        paint,
    ]


def search(task, budget, seed):
    rng = random.Random(seed)
    best, best_prog = (-1.0, -1.0), None
    for i in range(budget):
        prog = _sample(rng)
        f = evaluate(prog, task)
        if f > best:
            best, best_prog = f, prog
        if best[0] >= 1.0:
            return True, i + 1, best, best_prog
    return False, budget, best, best_prog


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print(f"Gate 2b — typed object-grammar arg search (budget={args.budget} seeds={args.seeds})\n")
    solved = 0
    for task_id in TARGET_TASK_IDS:
        task = load_task(task_id)
        t0 = time.time()
        outcomes = [search(task, args.budget, s) for s in range(args.seeds)]
        wins = [o for o in outcomes if o[0]]
        if wins:
            solved += 1
            evs = min(o[1] for o in wins)
            print(f"  SOLVED  {task_id}  {len(wins)}/{args.seeds} seeds  (min {evs} evals, {time.time() - t0:.1f}s)")
        else:
            bf = max(o[2] for o in outcomes)
            print(f"  ----    {task_id}  0/{args.seeds}  best={bf[0]:.2f}/{bf[1]:.3f}")

    print(f"\n  typed-grammar solved: {solved}/{len(TARGET_TASK_IDS)}")
    print("\nGATE 2b: " + ("PASS (structured search discovers)" if solved == len(TARGET_TASK_IDS) else "REVIEW"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
