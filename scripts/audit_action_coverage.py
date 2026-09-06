#!/usr/bin/env python3
"""Ranks the ARC-AGI-1 training tasks not yet in `arc_env.task_loader.
CURATED_TASK_IDS` by how many *new* `arc-dsl` primitives their known-correct
`third_party/arc-dsl/solvers.py` solver would need to become curatable -
step 0 of the task-coverage-growth loop (`docs/QUESTIONS.md` F11): re-run
this after every new ADR lands to get a fresh, ranked worklist instead of
re-reading solver bodies by hand.

Method (a name-level check, not a full reachability oracle - see Caveats):

1. Classify every top-level `dsl.py` primitive as "higher-order" if any
   parameter's type annotation mentions `Callable` - these build closures
   rather than transform a grid, and are structurally excluded per ADR-0001
   (`arc_env/actions.py`'s module docstring). This can never be curated
   without abandoning the "flat, one-primitive-per-step" action space.
2. Collect every `dsl.<name>` reference anywhere in `arc_env/actions.py` -
   the "curated by name" set. Actions often fuse several primitives into one
   (e.g. `select_largest` = `objects`+`argmax`+`size`+`toindices`), so this
   is deliberately name-level, not "is there a single action for this."
3. For each uncurated task, AST-walk its solver body (`solvers.py` does
   `from dsl import *`, so primitive calls are bare names) to get the set of
   `dsl.py` primitives it calls, then bucket the task:
   - `excluded`: uses a higher-order primitive - out of reach without a
     different action-space representation entirely.
   - `free_by_name`: every primitive it calls already appears somewhere in
     actions.py - a re-check candidate (NOT a guarantee - see Caveats).
   - `one_new` / `two_new` / `three_plus_new`: needs that many primitives,
     by name, that don't appear in actions.py at all yet - `one_new` is
     grouped by which primitive, ranked by how many tasks it would unlock,
     since one well-designed new action can be shared across a cluster of
     tasks (this is exactly how ADR-0012 covered 3 tasks with one pass).

Caveats (read before treating this as "these tasks are done"):

- Name overlap isn't reachability. A curated action's *specific*
  parameterization (e.g. `objects`'s (univalued, diagonal, without_bg)
  triple, or `argmax`'s compare function) may not match what a given
  solver actually needs - `docs/adr/0013-...md`'s `select_tallest`/
  `1c786137` case is a solver that's reachable at the bare-DSL level but not
  in the real step-by-step episode (an already-curated action ends the
  episode one step before the solver's own final step could run). Every
  `free_by_name`/`one_new`/`two_new` entry still needs a human/agent
  judgment pass before writing an ADR.
- A shared primitive name doesn't mean a shared *action design*. Several
  tasks landing in the same one-new bucket may still need different new
  actions if their required parameterization differs task to task.
- This only covers the 400 *training* tasks (the only ones with a published
  `solvers.py` solution) - the held-out *evaluation* set has no solver to
  derive curation from at all (`docs/QUESTIONS.md` F12).

Usage: `uv run python scripts/audit_action_coverage.py`
"""

import argparse
import ast
import inspect
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARC_DSL_DIR = REPO_ROOT / "third_party" / "arc-dsl"
ACTIONS_PATH = REPO_ROOT / "arc_env" / "actions.py"
SOLVERS_PATH = ARC_DSL_DIR / "solvers.py"


def _dsl_module():
    sys.path.insert(0, str(ARC_DSL_DIR))
    import dsl

    return dsl


def higher_order_primitives(dsl) -> set:
    """Names of `dsl.py` top-level functions with a `Callable`-typed param."""

    result = set()
    for name, fn in vars(dsl).items():
        if not (inspect.isfunction(fn) and fn.__module__ == "dsl"):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        if any("Callable" in str(p.annotation) for p in sig.parameters.values()):
            result.add(name)
    return result


def curated_primitives_by_name(actions_src: str) -> set:
    """Every `dsl.<name>` attribute reference anywhere in `actions.py`."""

    tree = ast.parse(actions_src)
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "dsl"
    }


def solver_primitive_usage(solvers_src: str, dsl_func_names: set) -> dict:
    """`{task_id: {dsl primitive names the solve_<task_id> body calls}}`."""

    tree = ast.parse(solvers_src)
    usage = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("solve_"):
            task_id = node.name[len("solve_"):]
            usage[task_id] = {
                sub.func.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in dsl_func_names
            }
    return usage


def bucket_tasks(remaining: dict, higher_order: set, curated_by_name: set) -> dict:
    buckets = {"excluded": {}, "free_by_name": {}, "one_new": {}, "two_new": {}, "three_plus_new": {}}
    for task_id, primitives in remaining.items():
        used_higher_order = primitives & higher_order
        if used_higher_order:
            buckets["excluded"][task_id] = used_higher_order
            continue
        missing = primitives - curated_by_name
        if not missing:
            buckets["free_by_name"][task_id] = primitives
        elif len(missing) == 1:
            buckets["one_new"][task_id] = missing
        elif len(missing) == 2:
            buckets["two_new"][task_id] = missing
        else:
            buckets["three_plus_new"][task_id] = missing
    return buckets


def report(buckets: dict, total_remaining: int, total_curated: int) -> None:
    excluded, free_by_name, one_new, two_new, three_plus = (
        buckets["excluded"], buckets["free_by_name"], buckets["one_new"], buckets["two_new"],
        buckets["three_plus_new"],
    )

    print(f"curated tasks: {total_curated}, remaining uncurated: {total_remaining}")
    print(f"  excluded (higher-order primitive):    {len(excluded):3d}")
    print(f"  free_by_name (re-check candidates):   {len(free_by_name):3d}")
    print(f"  one_new_primitive:                    {len(one_new):3d}")
    print(f"  two_new_primitives:                   {len(two_new):3d}")
    print(f"  three_plus_new_primitives:             {len(three_plus):3d}")
    print()

    print("=== free_by_name (verify parameterization before trusting these) ===")
    for task_id, prims in sorted(free_by_name.items()):
        print(f"  {task_id}: {sorted(prims)}")
    print()

    print("=== one_new_primitive, grouped by missing primitive, ranked by task count ===")
    by_missing = defaultdict(list)
    for task_id, missing in one_new.items():
        by_missing[next(iter(missing))].append(task_id)
    for prim, task_ids in sorted(by_missing.items(), key=lambda kv: -len(kv[1])):
        print(f"  {prim}: {len(task_ids)} tasks -> {sorted(task_ids)}")
    print()

    print("=== two_new_primitives, grouped by missing pair, ranked by task count ===")
    by_pair = defaultdict(list)
    for task_id, missing in two_new.items():
        by_pair[tuple(sorted(missing))].append(task_id)
    for pair, task_ids in sorted(by_pair.items(), key=lambda kv: -len(kv[1])):
        print(f"  {pair}: {len(task_ids)} tasks -> {sorted(task_ids)}")
    print()

    print("=== excluded, ranked by which higher-order primitive is most common ===")
    by_ho = defaultdict(list)
    for task_id, higher_order_used in excluded.items():
        for prim in higher_order_used:
            by_ho[prim].append(task_id)
    for prim, task_ids in sorted(by_ho.items(), key=lambda kv: -len(kv[1])):
        print(f"  {prim}: {len(task_ids)} tasks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from arc_env.task_loader import CURATED_TASK_IDS

    dsl = _dsl_module()
    dsl_func_names = {name for name, obj in vars(dsl).items() if inspect.isfunction(obj) and obj.__module__ == "dsl"}
    higher_order = higher_order_primitives(dsl)
    curated_by_name = curated_primitives_by_name(ACTIONS_PATH.read_text())
    all_usage = solver_primitive_usage(SOLVERS_PATH.read_text(), dsl_func_names)
    remaining = {t: p for t, p in all_usage.items() if t not in CURATED_TASK_IDS}

    buckets = bucket_tasks(remaining, higher_order, curated_by_name)
    report(buckets, total_remaining=len(remaining), total_curated=len(CURATED_TASK_IDS))


if __name__ == "__main__":
    main()
