#!/usr/bin/env python3
"""Harness for F12's "LLM-seeded search" refinement (`docs/QUESTIONS.md`,
ADR-0018): an LLM proposes a candidate action sequence, drawn only from the
existing curated DSL (`arc_env.actions`, never arbitrary code), for a task
GP and/or PPO currently struggle with. That sequence feeds GP's initial
population (`trainers.gp.evolve.run_gp`'s `seed_programs`) and/or PPO's
ADR-0009 warm-start (`train.py --algo ppo --warm_start_from`) as a *second*
demonstration source, alongside GP's own search - not instead of it.

**Honesty note, load-bearing for how this harness's output is read**: this
does NOT test "can an LLM discover solutions to previously-unreached tasks".
Every task this harness is meant to be pointed at is already in
`arc_env/task_loader.py`'s `CURATED_TASK_IDS`, with a known-correct sequence
already on file - a proposed sequence that reproduces every train/test pair
exactly is not a new discovery, it's a candidate demonstration. What this
harness (and the experiment built on it) actually tests is narrower and
still useful: does seeding GP's population / PPO's warm-start with an
independently-proposed demonstration help those trainers succeed more
reliably/cheaply than their own unseeded search, on tasks where that search
currently struggles.

Four pieces, each usable standalone:

- `describe_task(task_id)` - a task's train/test pairs plus the full
  curated action menu, in a form a human or LLM proposer can read and
  reason from.
- `verify(task_id, sequence)` - replays a candidate `[(name, args), ...]`
  sequence (the exact shape `CURATED_TASK_IDS` already uses) through
  `arc_env.actions.execute`'s raw-args path against every train + test pair,
  mirroring `tests/test_dsl_regression.py::
  test_solver_replays_through_env_action_executor` exactly (including reuse
  of that test module's own `_encode` helper, rather than a second
  reimplementation of "invert an ArgSpec's decode").
- `sequence_to_program(sequence)` - converts a verified sequence into a GP
  `Program` (`trainers.gp.genome.Program`), for use as a
  `run_gp(..., seed_programs=[...])` entry. Reuses the same `_encode`
  inversion `verify` and `trainers/ppo/warm_start.py`'s `_DECODE_INVERSE`
  both already solve, rather than a third implementation.
- `write_seed_episode(task_id, sequence, run_dir)` - once verified, writes
  the sequence out as a real `runs/<run_id>/episodes/best-program.jsonl` +
  `run_meta.json`, via the *unmodified* `arc_env.episode_log.EpisodeWriter`/
  `write_run_meta` (mirroring `viz/backend/play.py`'s ADR-0017 pattern of
  reusing this schema unchanged) with `algo="llm-seed"` - the value
  `train.check_warm_start_compatible`'s allowlist accepts per ADR-0018.
  Replays through `ArcEnv.step` (`trainers.gp.replay.
  program_to_episode_trace`, unmodified) rather than the raw executor, so
  the logged reward/exact_match/selected fields come from the exact same
  step-by-step semantics PPO/GP episodes already use.

CLI:

    python scripts/llm_seed_search.py describe --task_id ea32f347
    python scripts/llm_seed_search.py verify --task_id ea32f347
    python scripts/llm_seed_search.py write-episode --task_id ea32f347 --run_id llm-seed-ea32f347

`verify`/`write-episode` with no explicit `--sequence` use this module's own
`PROPOSED_SEQUENCES` table (the 3 sequences derived for F12's small
experiment - see `docs/research/llm-seeded-search-experiment.md`); pass
`--sequence '[["vmirror", []], ...]'` (JSON) to check/write a different one.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_env import actions
from arc_env.env import ArcEnv
from arc_env.episode_log import RunMeta, write_run_meta
from arc_env.task_loader import Task, load_task
from tests.test_dsl_regression import _encode
from train import _write_episode
from trainers.gp.genome import Program
from trainers.gp.replay import program_to_episode_trace

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

Sequence = list  # list[tuple[str, tuple]] - [(action_name, real_args), ...], CURATED_TASK_IDS's own shape

# The 3 sequences this pass's small experiment proposes and verifies (F12's
# target tasks - see `docs/QUESTIONS.md` and the research doc for how each
# was derived from its task's train pairs, and whether it matches
# `arc_env.task_loader.CURATED_TASK_IDS`'s existing entry for that task).
PROPOSED_SEQUENCES: dict[str, Sequence] = {
    "ea32f347": [
        ("replace", (5, 4)),
        ("select_largest", ()),
        ("recolor_selected", (1,)),
        ("select_smallest", ()),
        ("recolor_selected", (2,)),
    ],
    "5bd6f4ac": [("commit", (0, 6, 3, 3))],
    "5614dbcf": [("replace", (5, 0)), ("downscale", (3,))],
}


def describe_task(task_id: str) -> dict:
    """A task's train/test pairs plus the curated action menu (name/kind/
    args), in a form suitable for a human or LLM proposer to reason from -
    no derivation logic here, just the raw material for reasoning."""

    task: Task = load_task(task_id)
    menu = [
        {"name": a.name, "kind": a.kind, "args": [{"name": s.name, "kind": s.kind} for s in a.args]}
        for a in actions.ACTIONS
    ]
    return {"task": task, "menu": menu}


def verify(task_id: str, sequence: Sequence) -> dict:
    """Replays `sequence` through `arc_env.actions.execute`'s raw-args path
    against every train + test pair of `task_id`, mirroring
    `tests/test_dsl_regression.py::
    test_solver_replays_through_env_action_executor` exactly. Returns
    `{"task_id", "sequence", "pairs": [{"split", "index", "passed",
    "reason"}, ...], "all_passed": bool}` - a per-pair report, not just a
    single pass/fail bit, so a partial failure is diagnosable."""

    task = load_task(task_id)
    results = []
    all_passed = True

    for split, pairs in (("train", task.train), ("test", task.test)):
        for i, pair in enumerate(pairs):
            grid = pair.input
            selected = None
            passed = True
            reason = None

            for step_i, (primitive_name, args) in enumerate(sequence):
                if primitive_name not in actions.ACTION_BY_NAME:
                    passed, reason = False, f"step {step_i}: unknown primitive {primitive_name!r}"
                    break
                primitive_index = actions.ACTION_BY_NAME[primitive_name]
                action = actions.ACTIONS[primitive_index]
                raw_args = tuple(_encode(spec, value) for spec, value in zip(action.args, args))
                grid, selected, decoded, valid = actions.execute(primitive_index, raw_args, grid, selected)
                if not valid:
                    passed, reason = False, f"step {step_i} ({primitive_name}{tuple(args)}) was invalid"
                    break
                if tuple(decoded.values()) != tuple(args):
                    passed = False
                    reason = f"step {step_i}: decoded args {tuple(decoded.values())} != requested {tuple(args)}"
                    break

            if passed and grid != pair.output:
                passed, reason = False, "final grid does not match the expected output"

            results.append({"split": split, "index": i, "passed": passed, "reason": reason})
            all_passed = all_passed and passed

    return {"task_id": task_id, "sequence": sequence, "pairs": results, "all_passed": all_passed}


def sequence_to_program(sequence: Sequence) -> Program:
    """Converts a verified `[(name, args), ...]` sequence into a GP
    `Program` (`trainers.gp.genome.Program` - `list[(primitive_index,
    raw_args)]`), for use as a `run_gp(..., seed_programs=[...])` entry.
    Inverts each arg's `kind`-specific `decode` back to a raw value via
    `tests/test_dsl_regression.py`'s `_encode` helper - the same inversion
    `trainers/ppo/warm_start.py`'s `_DECODE_INVERSE` solves for a
    differently-shaped input (a logged episode's decoded-args dict, vs. this
    module's `(name, args)` tuple shape) - reused here rather than
    reinvented a third time. Does not itself verify the sequence; callers
    should `verify` first."""

    program: Program = []
    for primitive_name, args in sequence:
        primitive_index = actions.ACTION_BY_NAME[primitive_name]
        action = actions.ACTIONS[primitive_index]
        raw_args = [_encode(spec, value) for spec, value in zip(action.args, args)]
        raw_args += [0] * (actions.MAX_ARITY - len(raw_args))
        program.append((primitive_index, tuple(raw_args)))
    return program


def write_seed_episode(task_id: str, sequence: Sequence, run_dir: Path, pair_index: int = 0) -> dict:
    """Writes a verified `sequence` out as a real `runs/<run_id>/episodes/
    best-program.jsonl` + `run_meta.json`, via the unmodified
    `arc_env.episode_log.EpisodeWriter`/`write_run_meta` (mirroring
    `viz/backend/play.py`'s ADR-0017 pattern) - `algo="llm-seed"` (ADR-0018).
    Replays through `ArcEnv.step` (`trainers.gp.replay.
    program_to_episode_trace`, unmodified) so the logged
    reward/terminated/exact_match/selected fields match real step-by-step
    env semantics exactly, not just a bare-function replay. Returns
    `{"run_dir", "success", "total_reward"}`."""

    task = load_task(task_id)
    program = sequence_to_program(sequence)
    env = ArcEnv()
    pair = task.train[pair_index]

    write_run_meta(run_dir, RunMeta(
        run_id=run_dir.name, algo="llm-seed", task_ids=[task_id],
        config={"pair_index": pair_index, "sequence": sequence},
    ))
    trace = program_to_episode_trace(env, program, task_id, pair)
    _write_episode(run_dir, "best-program", env, task_id, pair, trace)

    return {"run_dir": str(run_dir), "success": trace["success"], "total_reward": trace["total_reward"]}


def _print_verify_report(report: dict) -> None:
    status = "ALL PASSED" if report["all_passed"] else "SOME FAILED"
    print(f"{report['task_id']}: {status}")
    for p in report["pairs"]:
        mark = "ok" if p["passed"] else f"FAIL ({p['reason']})"
        print(f"  {p['split']}[{p['index']}]: {mark}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["describe", "verify", "write-episode"])
    parser.add_argument("--task_id", required=True)
    parser.add_argument(
        "--sequence", default=None,
        help='JSON [(name, args), ...] - defaults to PROPOSED_SEQUENCES[task_id] if omitted.',
    )
    parser.add_argument("--run_id", default=None, help="write-episode only; defaults to llm-seed-<task_id>.")
    parser.add_argument("--pair_index", type=int, default=0, help="write-episode only.")
    parser.add_argument("--runs_dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    if args.sequence is not None:
        sequence = [(name, tuple(seq_args)) for name, seq_args in json.loads(args.sequence)]
    elif args.task_id in PROPOSED_SEQUENCES:
        sequence = PROPOSED_SEQUENCES[args.task_id]
    else:
        parser.error(f"no --sequence given and {args.task_id!r} has no PROPOSED_SEQUENCES entry")

    if args.command == "describe":
        described = describe_task(args.task_id)
        print(f"task {args.task_id}: {len(described['task'].train)} train pair(s), "
              f"{len(described['task'].test)} test pair(s)")
        for i, pair in enumerate(described["task"].train):
            print(f"  train[{i}] input {len(pair.input)}x{len(pair.input[0])} "
                  f"-> output {len(pair.output)}x{len(pair.output[0])}")
        print(f"curated action menu: {len(described['menu'])} actions")
    elif args.command == "verify":
        _print_verify_report(verify(args.task_id, sequence))
    else:
        report = verify(args.task_id, sequence)
        if not report["all_passed"]:
            parser.error(f"sequence does not verify for {args.task_id!r} - run `verify` first")
        run_id = args.run_id or f"llm-seed-{args.task_id}"
        result = write_seed_episode(args.task_id, sequence, args.runs_dir / run_id, args.pair_index)
        print(f"wrote {result['run_dir']} (success={result['success']}, total_reward={result['total_reward']})")


if __name__ == "__main__":
    main()
