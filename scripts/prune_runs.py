#!/usr/bin/env python3
"""Prunes `runs/` so it doesn't silently re-accumulate (KAN-1183 cleanup).

Ad-hoc rollouts/training passes pile up fast (a full curated-task pass alone
is 50+ directories) and `runs/` is gitignored, so nothing else ever cleans it
up. Retention rule: keep every run whose `run_meta.json` `created_at` falls
on one of the `--keep_days` most recent distinct calendar dates seen across
`runs/` (default 1 - "today's" runs, by `created_at`'s UTC date, survive;
everything older is a prune candidate). This is a date-bucket rule, not a
run-count rule, so a whole same-day batch (e.g. a 26-task training pass, or a
handful of GP+PPO pairs from one investigation) survives or is pruned
together - one run from a batch surviving while its siblings vanish would
leave `runs/` in a half-consistent state.

Defaults to a dry run (prints what would be deleted); pass --yes to actually
delete. Run directories with no readable `run_meta.json` are left alone
(reported, never auto-deleted - could be a run mid-write).

    uv run python scripts/prune_runs.py              # dry run
    uv run python scripts/prune_runs.py --yes         # actually delete
    uv run python scripts/prune_runs.py --keep_days 3 --yes
"""

import argparse
import json
import shutil
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _created_date(run_dir: Path) -> str | None:
    """Returns the UTC calendar-date portion of `created_at` (e.g.
    `"2026-09-04"`), or `None` if `run_meta.json` is missing/unreadable."""

    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        return None
    created_at = meta.get("created_at")
    if not created_at:
        return None
    return created_at.split("T", 1)[0]


def plan_prune(runs_dir: Path, keep_days: int) -> tuple[list[Path], list[Path], list[Path]]:
    """Returns `(keep, delete, unreadable)` run directories under `runs_dir`.

    `unreadable` (no parseable `created_at`) is always kept out of `delete` -
    never auto-remove a run we can't date."""

    if not runs_dir.is_dir():
        return [], [], []

    dated: list[tuple[Path, str]] = []
    unreadable: list[Path] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        date = _created_date(run_dir)
        if date is None:
            unreadable.append(run_dir)
        else:
            dated.append((run_dir, date))

    kept_dates = sorted({date for _, date in dated}, reverse=True)[:keep_days]
    keep = [run_dir for run_dir, date in dated if date in kept_dates]
    delete = [run_dir for run_dir, date in dated if date not in kept_dates]
    return keep, delete, unreadable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--keep_days", type=int, default=1,
        help="Keep runs from the N most recent distinct created_at dates (default: 1).",
    )
    parser.add_argument("--yes", action="store_true", help="Actually delete (default: dry run).")
    args = parser.parse_args()

    keep, delete, unreadable = plan_prune(args.runs_dir, args.keep_days)

    print(f"keeping {len(keep)} run(s):")
    for run_dir in keep:
        print(f"  {run_dir.name}")
    if unreadable:
        print(f"skipping {len(unreadable)} run(s) with no readable run_meta.json (never auto-deleted):")
        for run_dir in unreadable:
            print(f"  {run_dir.name}")
    print(f"{'deleting' if args.yes else 'would delete'} {len(delete)} run(s):")
    for run_dir in delete:
        print(f"  {run_dir.name}")

    if not args.yes:
        print("\ndry run - pass --yes to actually delete")
        return

    for run_dir in delete:
        shutil.rmtree(run_dir)
    print(f"\ndeleted {len(delete)} run(s)")


if __name__ == "__main__":
    main()
