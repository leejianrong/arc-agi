"""The SLICES.md V5 demo entrypoint: replay a hand-written object-space program
through `object_env` and show it reproducing a task — a set-op task *and* a
non-set-op (move/crop/…) task, evidence the one grammar spans families.

    uv run python -m object_env.cli replay 6430c8c4
    uv run python -m object_env.cli replay 25ff71a9
"""

import argparse

from arc_env.tasks import load_task
from object_env.programs import PROGRAMS, build


def _fmt(grid) -> str:
    return "\n".join(" ".join(f"{v}" for v in row) for row in grid)


def replay(task_id: str) -> int:
    if task_id not in PROGRAMS:
        print(f"no object-space program for {task_id!r}; known: {', '.join(PROGRAMS)}")
        return 1
    program = build(task_id)  # type-checks at construction
    task = load_task(task_id)
    steps = " -> ".join(name for name, _ in PROGRAMS[task_id])
    print(f"task {task_id}\nprogram: {steps}\n")
    ok = 0
    pairs = list(task.train) + list(task.test)
    for i, pair in enumerate(pairs):
        got = program.run(pair.input)
        match = got == pair.output
        ok += match
        print(f"  pair {i}: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            print("  expected:\n" + _fmt(pair.output) + "\n  got:\n" + _fmt(got))
    print(f"\n{ok}/{len(pairs)} pairs reproduced exactly")
    return 0 if ok == len(pairs) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="replay an object-space program for a task")
    r.add_argument("task_id")
    args = ap.parse_args()
    if args.cmd == "replay":
        return replay(args.task_id)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
