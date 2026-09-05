"""Tests for `scripts/prune_runs.py`'s retention logic."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prune_runs import plan_prune


def _make_run(runs_dir: Path, run_id: str, created_at: str | None) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    if created_at is not None:
        (run_dir / "run_meta.json").write_text(json.dumps({
            "schema_version": 1, "run_id": run_id, "algo": "gp",
            "created_at": created_at, "task_ids": [], "config": {},
        }))
    return run_dir


def test_plan_prune_on_missing_dir_is_all_empty(tmp_path):
    keep, delete, unreadable = plan_prune(tmp_path / "does-not-exist", keep_days=1)
    assert (keep, delete, unreadable) == ([], [], [])


def test_keeps_only_most_recent_date_by_default(tmp_path):
    runs_dir = tmp_path / "runs"
    old_a = _make_run(runs_dir, "old-a", "2026-08-30T18:53:04Z")
    old_b = _make_run(runs_dir, "old-b", "2026-08-30T19:22:42Z")
    new = _make_run(runs_dir, "new", "2026-09-04T15:53:52Z")

    keep, delete, unreadable = plan_prune(runs_dir, keep_days=1)

    assert keep == [new]
    assert sorted(delete) == sorted([old_a, old_b])
    assert unreadable == []


def test_keep_days_widens_the_surviving_window(tmp_path):
    runs_dir = tmp_path / "runs"
    day1 = _make_run(runs_dir, "day1", "2026-09-02T00:00:00Z")
    day2 = _make_run(runs_dir, "day2", "2026-09-03T00:00:00Z")
    day3 = _make_run(runs_dir, "day3", "2026-09-04T00:00:00Z")

    keep, delete, unreadable = plan_prune(runs_dir, keep_days=2)

    assert sorted(keep) == sorted([day2, day3])
    assert delete == [day1]
    assert unreadable == []


def test_whole_same_day_batch_survives_together(tmp_path):
    # A same-day training pass (many run dirs, one shared date) must not be
    # split - either the whole batch is within the keep window or none of it is.
    runs_dir = tmp_path / "runs"
    batch = [_make_run(runs_dir, f"training-pass-{i}", "2026-08-31T02:53:00Z") for i in range(5)]

    keep, delete, unreadable = plan_prune(runs_dir, keep_days=1)

    assert sorted(keep) == sorted(batch)
    assert delete == []
    assert unreadable == []


def test_run_with_no_run_meta_is_never_deleted(tmp_path):
    runs_dir = tmp_path / "runs"
    good = _make_run(runs_dir, "good", "2026-09-04T00:00:00Z")
    unwritten = _make_run(runs_dir, "mid-write", created_at=None)

    keep, delete, unreadable = plan_prune(runs_dir, keep_days=1)

    assert keep == [good]
    assert delete == []
    assert unreadable == [unwritten]


def test_run_with_malformed_run_meta_is_never_deleted(tmp_path):
    runs_dir = tmp_path / "runs"
    bad_dir = runs_dir / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "run_meta.json").write_text("{not valid json")

    keep, delete, unreadable = plan_prune(runs_dir, keep_days=1)

    assert keep == []
    assert delete == []
    assert unreadable == [bad_dir]
