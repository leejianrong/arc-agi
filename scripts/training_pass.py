#!/usr/bin/env python3
"""Serial batch training-pass driver over the curated tasks (the F11 training
pass: measure how well GP / PPO-cold / PPO-warm actually *learn* the curated
tasks, not just whether a known dsl solver maps onto the action space).

For each task, runs up to three arms, each in its OWN SUBPROCESS:
  1. gp        -> runs/<pass_id>/<task>-gp/
  2. ppo-cold  -> runs/<pass_id>/<task>-ppo-cold/
  3. ppo-warm  -> runs/<pass_id>/<task>-ppo-warm/  (--warm_start_from the gp run)

Why subprocesses, and why strictly serial: this machine has very little RAM
(~7.8 GiB, swap usually near-full) and a past pass OOM-crashed it. A fresh
subprocess per run means each run's torch/numpy memory is fully reclaimed on
exit instead of accumulating across the ~270 runs a full pass makes; running
them one at a time (never in parallel) keeps peak memory to a single training
process. This script NEVER runs two trainings at once. See the repo's
`env-ram-constraint` agent-memory note.

Resumable: a run whose dir already holds a `run_meta.json` + non-empty
`metrics.jsonl` is treated as done and skipped, so a crash/interrupt only
costs the run in flight. Per-run wall-time and peak RSS (polled from /proc)
are recorded, and a solved/not summary is written to
`docs/results/training-pass-<pass_id>.md` after every run (the raw `runs/`
tree is gitignored; this committed summary is the durable record — the last
pass's results survived only in a memory note because nothing committed them).

'Solved' means:
  - gp:  final metrics row `success_rate` (GP's best_fitness) == 1.0
  - ppo: final metrics row `eval_success` True — the fixed development-train pair's
         greedy-policy `info["exact_match"]` (per CLAUDE.md), NOT the noisy
         per-rollout `success_rate`. 'ever' columns also report whether any
         eval row was ever solved (the last pass saw PPO reach a working
         policy then regress, so both are worth tracking).

Usage:
  uv run python scripts/training_pass.py --tasks calib          # ~5-task calibration
  uv run python scripts/training_pass.py --tasks all            # full curated set
  uv run python scripts/training_pass.py --tasks 67a3c6ac,44f52bb0
  uv run python scripts/training_pass.py --tasks calib --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from arc_env.splits import SEARCH_TASK_IDS

RUNS_DIR = REPO / "runs"
RESULTS_DIR = REPO / "docs" / "results"

# A small, deliberately mixed calibration set: the easy PPO sanity fixture,
# an object-selection task (heavier env, PPO struggled on this class last
# pass), and three ADR-0026 additions spanning a 1x1 canvas, a variable-shape
# canvas, and one of the heavier same-shape classifiers. Filtered to whatever
# is actually curated at runtime.
CALIB_PREFERENCE = ["67a3c6ac", "1f85a75f", "44f52bb0", "d0f5fe59", "b548a754"]

ARMS = ("gp", "ppo-cold", "ppo-warm")


def _python() -> str:
    """Prefer the venv interpreter directly (no `uv run` wrapper child) so the
    subprocess we launch IS the training process, keeping the RSS measurement
    and process-group bookkeeping simple; fall back to the current one."""
    venv_py = REPO / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def _group_rss_kb(pgid: int) -> int:
    """Sum VmRSS (kB) across every process in group `pgid` — the launched
    child (a group leader, via start_new_session) plus any descendants."""
    total = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                stat = f.read()
            after = stat[stat.rindex(")") + 2:].split()  # skip 'comm' (may hold spaces)
            if int(after[2]) != pgid:  # after = [state, ppid, pgrp, ...]
                continue
            with open(f"/proc/{entry}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            continue
    return total


def _run(cmd: list[str], log_path: Path) -> tuple[int, float, float]:
    """Run `cmd` serially to completion, polling peak RSS. Returns
    (returncode, wall_seconds, peak_rss_mb)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    peak_kb = 0
    start = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(REPO)
        )
        while proc.poll() is None:
            peak_kb = max(peak_kb, _group_rss_kb(proc.pid))
            time.sleep(0.5)
    return proc.returncode, time.monotonic() - start, peak_kb / 1024.0


def _is_done(run_dir: Path, arm: str, n_updates: int) -> bool:
    """A run counts as done only if it actually *completed*, so a resume
    re-runs any run interrupted mid-training (e.g. by a machine sleep or
    reboot) instead of treating its partial output as the final result. GP
    writes its metrics only after the whole search finishes, so any metrics
    row means done; PPO streams one row per update, so require the last row to
    be the final update (a run stopped at update 15/30 has 15 rows and must be
    re-run, not skipped)."""
    meta = run_dir / "run_meta.json"
    metrics = run_dir / "metrics.jsonl"
    if not (meta.exists() and metrics.exists() and metrics.stat().st_size > 0):
        return False
    if arm == "gp":
        return True
    rows = _final_metrics(run_dir)
    return bool(rows) and rows[-1].get("update") == n_updates - 1


def _write_stats(run_dir: Path, wall_s: float, peak_mb: float, rc: int) -> None:
    (run_dir / "driver_stats.json").write_text(
        json.dumps({"wall_s": wall_s, "peak_mb": peak_mb, "rc": rc})
    )


def _read_stats(run_dir: Path) -> dict:
    """Per-run wall/RSS/rc persisted by this driver, so a resumed pass still
    reports them for runs it skipped this invocation. Zeros if absent (e.g. a
    run completed by a different invocation that didn't record them)."""
    path = run_dir / "driver_stats.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"wall_s": 0.0, "peak_mb": 0.0, "rc": 0}


def _final_metrics(run_dir: Path) -> list[dict]:
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return []
    rows = []
    for line in metrics.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _gp_solved(run_dir: Path) -> bool | None:
    rows = _final_metrics(run_dir)
    if not rows:
        return None
    return any(r.get("success_rate") == 1.0 for r in rows)


def _ppo_result(run_dir: Path) -> tuple[bool | None, bool]:
    """(final_eval_success, ever_eval_success)."""
    rows = _final_metrics(run_dir)
    evals = [r for r in rows if r.get("eval_success") is not None]
    if not evals:
        return None, False
    return bool(evals[-1]["eval_success"]), any(bool(r["eval_success"]) for r in evals)


def _build_cmd(arm: str, task: str, pass_dir: Path, args) -> list[str]:
    run_id = f"{task}-{arm}"
    base = [
        _python(), "train.py", "--task_id", task, "--run_id", run_id,
        "--runs_dir", str(pass_dir), "--seed", str(args.seed), "--max_steps", str(args.max_steps),
    ]
    if arm == "gp":
        return base + [
            "--algo", "gp", "--n_generations", str(args.gp_generations),
            "--population_size", str(args.gp_population),
        ]
    cmd = base + [
        "--algo", "ppo", "--n_updates", str(args.n_updates),
        "--rollout_steps", str(args.rollout_steps), "--eval_every", str(args.eval_every),
    ]
    if arm == "ppo-warm":
        cmd += ["--warm_start_from", str(pass_dir / f"{task}-gp")]
    return cmd


def _write_summary(path: Path, pass_id: str, args, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    solved = {arm: sum(1 for r in records if r["arm"] == arm and r["solved"] is True) for arm in ARMS}
    total = {arm: sum(1 for r in records if r["arm"] == arm) for arm in ARMS}
    lines = [
        f"# Training pass `{pass_id}`",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}",
        (
            f"- Settings: PPO n_updates={args.n_updates}, rollout_steps={args.rollout_steps}, "
            f"eval_every={args.eval_every}; GP generations={args.gp_generations}, "
            f"population={args.gp_population}; max_steps={args.max_steps}, seed={args.seed}"
        ),
        f"- Arms: {', '.join(args.arms)}",
        "",
        "## Solved totals",
        "",
        "| Arm | Solved | Runs |",
        "|-----|--------|------|",
    ]
    for arm in ARMS:
        if total[arm]:
            lines.append(f"| {arm} | {solved[arm]} | {total[arm]} |")
    lines += [
        "",
        "## Per-task",
        "",
        "| Task | Arm | Solved | Ever | Wall (s) | Peak RSS (MB) | rc |",
        "|------|-----|--------|------|----------|---------------|----|",
    ]
    for r in records:
        solved_s = {True: "yes", False: "no", None: "?"}[r["solved"]]
        ever_s = {True: "yes", False: "no", None: "-"}[r.get("ever")]
        lines.append(
            f"| {r['task']} | {r['arm']} | {solved_s} | {ever_s} | "
            f"{r['wall_s']:.0f} | {r['peak_mb']:.0f} | {r['rc']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pass-id", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--tasks", default="calib", help="'calib', 'all', or a comma-separated task list")
    p.add_argument("--arms", default=",".join(ARMS), help="comma-separated subset of gp,ppo-cold,ppo-warm")
    p.add_argument("--n-updates", dest="n_updates", type=int, default=30)
    p.add_argument("--rollout-steps", dest="rollout_steps", type=int, default=256)
    p.add_argument("--eval-every", dest="eval_every", type=int, default=5)
    p.add_argument("--gp-generations", dest="gp_generations", type=int, default=100)
    p.add_argument("--gp-population", dest="gp_population", type=int, default=200)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args()

    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in args.arms:
        if a not in ARMS:
            p.error(f"unknown arm {a!r} (choose from {ARMS})")
    if "ppo-warm" in args.arms and "gp" not in args.arms:
        p.error("ppo-warm needs gp in --arms (it warm-starts from the gp run)")

    if args.tasks == "all":
        tasks = sorted(SEARCH_TASK_IDS)
    elif args.tasks == "calib":
        tasks = [t for t in CALIB_PREFERENCE if t in SEARCH_TASK_IDS]
        dropped = [t for t in CALIB_PREFERENCE if t not in SEARCH_TASK_IDS]
        if dropped:
            print(f"note: calib tasks not curated, dropped: {dropped}")
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [t for t in tasks if t not in SEARCH_TASK_IDS]
    if bad:
        p.error(f"not curated: {bad}")

    pass_dir = RUNS_DIR / args.pass_id
    summary_path = RESULTS_DIR / f"training-pass-{args.pass_id}.md"
    n_runs = len(tasks) * len(args.arms)
    print(f"pass {args.pass_id}: {len(tasks)} tasks x {len(args.arms)} arms = {n_runs} runs (serial)")
    print(f"tasks: {tasks}")
    print(f"runs -> {pass_dir}")
    print(f"summary -> {summary_path}")
    if args.dry_run:
        for t in tasks:
            for arm in args.arms:
                print("  " + " ".join(_build_cmd(arm, t, pass_dir, args)))
        return

    records: list[dict] = []
    done = 0
    for task in tasks:
        for arm in args.arms:
            done += 1
            run_dir = pass_dir / f"{task}-{arm}"
            tag = f"[{done}/{n_runs}] {task} {arm}"
            if _is_done(run_dir, arm, args.n_updates):
                print(f"{tag}: already done, skip")
            else:
                if arm == "ppo-warm" and not (pass_dir / f"{task}-gp" / "episodes" / "best-program.jsonl").exists():
                    print(f"{tag}: SKIP — no gp best-program to warm-start from")
                    continue
                cmd = _build_cmd(arm, task, pass_dir, args)
                print(f"{tag}: running…", flush=True)
                rc, wall_s, peak_mb = _run(cmd, run_dir / "driver.log")
                _write_stats(run_dir, wall_s, peak_mb, rc)
                print(f"{tag}: rc={rc} wall={wall_s:.0f}s peak={peak_mb:.0f}MB")
            if arm == "gp":
                solved, ever = _gp_solved(run_dir), None
            else:
                solved, ever = _ppo_result(run_dir)
            stats = _read_stats(run_dir)
            records.append({
                "task": task, "arm": arm, "solved": solved, "ever": ever,
                "wall_s": stats["wall_s"], "peak_mb": stats["peak_mb"], "rc": stats["rc"],
            })
            _write_summary(summary_path, args.pass_id, args, records)

    print(f"\ndone. summary written to {summary_path}")
    for arm in args.arms:
        s = sum(1 for r in records if r["arm"] == arm and r["solved"] is True)
        n = sum(1 for r in records if r["arm"] == arm)
        print(f"  {arm}: {s}/{n} solved")


if __name__ == "__main__":
    main()
