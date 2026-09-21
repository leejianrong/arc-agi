"""The grammar-aware discovery proof (V5 fork 3, recommended).

Two demonstrations, mirroring the POC's two results, both driven by the
grammar's own type-valid enumeration and scored with the shipped dense reward
(`arc_env.reward`) — no hardcoded skeletons, no per-task bespoke search code:

1. `discover` — type-directed search *from scratch*. Enumerate type-valid
   skeletons shortest-first (the skeleton emerges from the types) and arg-search
   each; a param-less skeleton costs one eval, not a wasted batch. This solves
   the shallow families (move / crop / canvas / transform) in well under a
   second — where a flat pick-any-of-N genome plateaus.

2. `--fixture-arg-search` within a grammar-*derived* skeleton — an explicitly
   non-blind diagnostic and faithful re-proof of
   the POC's set-op 8/8, generalized to every fixture family. A deep 5-step
   program is a needle in ~170k type-valid depth-5 skeletons, so uniform
   from-scratch discovery of it is the deceptive-landscape problem ADR-0029
   hands to V6's biased/learned search (commitment #4). But given the skeleton
   (obtained from `grammar.skeleton_of` over a fixture program — its verbs and
   a/b wiring, with the value args stripped and re-searched), arg-search
   rediscovers the exact colors/ops/directions in a few thousand samples.

    uv run python -m object_env.search [--budget N] [--seeds K]
    uv run python -m object_env.search --fixture-arg-search  # non-blind diagnostic

Runs in seconds locally, serially, within the RAM budget.
"""

import argparse
import random
import statistics
import time

from arc_env.reward import compute_diff_mask, similarity
from arc_env.tasks import load_search_task
from object_env.actions import ACTIONS
from object_env.grammar import enumerate_skeletons, iter_fills, skeleton_of

# shallow families discoverable from scratch (depth <= 3, low arity); the rest
# are the deep multi-step programs proven via skeleton arg-search below.
FROM_SCRATCH_TASK_IDS = ["25ff71a9", "1f85a75f", "23b5c85d", "5582e5ca", "b1948b0a"]


def evaluate(program, task) -> tuple:
    """(mean exact-match over train pairs, mean similarity) — the fitness the
    shipped GP uses (`trainers/gp/fitness`), so a discovered program is
    'solved' exactly when the trainers would count it."""
    exact = 0
    sims = []
    for pair in task.train:
        diff = compute_diff_mask(pair.input, pair.output)
        try:
            out = program.run(pair.input)
            sims.append(similarity(out, pair.output, diff))
            exact += out == pair.output
        except Exception:  # noqa: BLE001 - a malformed program can raise any DSL error
            sims.append(0.0)
    n = len(task.train)
    return (exact / n if n else 0.0, statistics.mean(sims) if sims else 0.0)


def _grid_terminal(skeleton) -> bool:
    action, sargs = skeleton[-1]
    return "grid" in action.writes(sargs)


def arg_search(task, skeleton, budget: int, seed: int) -> tuple:
    """Search a skeleton's value args (exhaustively when the space is small,
    else sampled) until exact or budget runs out. (solved, evals, best)."""
    rng = random.Random(seed)
    best, used = (-1.0, -1.0), 0
    for prog in iter_fills(skeleton, rng, budget):
        f = evaluate(prog, task)
        used += 1
        best = max(best, f)
        if best[0] >= 1.0:
            return True, used, best
    return False, used, best


def discover(task, budget: int, seed: int, max_depth: int = 3, per_skeleton: int = 400) -> tuple:
    """From-scratch type-directed search: shortest-first over type-valid
    grid-terminal skeletons, arg-searching each (small spaces exhaustively).
    (solved, evals, best, program)."""
    rng = random.Random(seed)
    best, best_prog, used = (-1.0, -1.0), None, 0
    for skeleton in enumerate_skeletons(ACTIONS, max_depth):
        if not _grid_terminal(skeleton):
            continue
        for prog in iter_fills(skeleton, rng, per_skeleton):
            if used >= budget:
                return False, used, best, best_prog
            f = evaluate(prog, task)
            used += 1
            if f > best:
                best, best_prog = f, prog
            if f[0] >= 1.0:
                return True, used, best, prog
    return False, used, best, best_prog


def _report(label, task_ids, run, seeds):
    solved = 0
    for tid in task_ids:
        task = load_search_task(tid)
        t0 = time.time()
        outs = [run(task, s) for s in range(seeds)]
        wins = [o for o in outs if o[0]]
        if wins:
            solved += 1
            print(f"  SOLVED  {tid}  {len(wins)}/{seeds}  (min {min(o[1] for o in wins)} evals, {time.time() - t0:.1f}s)")
        else:
            bf = max(o[2] for o in outs)
            print(f"  ----    {tid}  best={bf[0]:.2f}/{bf[1]:.3f}  ({time.time() - t0:.1f}s)")
    print(f"\n  {label}: {solved}/{len(task_ids)}\n")
    return solved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=80_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--fixture-arg-search",
        action="store_true",
        help="also run the non-blind task-specific fixture-skeleton diagnostic",
    )
    args = ap.parse_args()

    print("[1] type-directed discovery from scratch (shallow families)\n")
    _report("discovered from scratch", FROM_SCRATCH_TASK_IDS,
            lambda task, s: discover(task, args.budget, s), args.seeds)

    if not args.fixture_arg_search:
        return 0

    # Import task-specific known programs only after the caller explicitly
    # requests fixture-guided diagnostics. Blind discovery never loads them.
    from object_env.programs import SET_OP_TASK_IDS, TARGET_TASK_IDS, build

    print("[2] NON-BLIND fixture arg-search within grammar-derived skeletons (all families)\n")
    solved = 0
    for tid in TARGET_TASK_IDS:
        skeleton = skeleton_of(build(tid))  # grammar-valid by construction (build type-checks)
        task = load_search_task(tid)
        t0 = time.time()
        outs = [arg_search(task, skeleton, args.budget, s) for s in range(args.seeds)]
        wins = [o for o in outs if o[0]]
        fam = "set-op " if tid in SET_OP_TASK_IDS else "generic"
        if wins:
            solved += 1
            print(f"  SOLVED  {tid} [{fam}]  {len(wins)}/{args.seeds}  (min {min(o[1] for o in wins)} evals, {time.time() - t0:.1f}s)")
        else:
            bf = max(o[2] for o in outs)
            print(f"  ----    {tid} [{fam}]  best={bf[0]:.2f}/{bf[1]:.3f}  ({time.time() - t0:.1f}s)")
    print(f"\n  rediscovered via grammar skeleton: {solved}/{len(TARGET_TASK_IDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
