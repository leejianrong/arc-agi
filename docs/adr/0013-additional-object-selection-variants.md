# ADR-0013: Additional `objects()` connectivity variants

- Status: Accepted
- Date: 2026-09-05
- Deciders: repo owner (delegated design, per ADR-0011/0012's own deferral), via conversation 2026-09-05

## Context

ADR-0011 curated exactly one `dsl.objects(grid, univalued, diagonal,
without_bg)` triple - `(True, True, True)` (univalued, diagonal-connected,
background-excluded) - and explicitly deferred "multiple curated
`objects(...)` connectivity variants" as future work, naming two known but
unreached fixture tasks: `be94b721` (needs `objects(I, T, F, T)` -
`diagonal=False`) and `1c786137` (needs `objects(I, T, F, F)` -
`diagonal=False` *and* `without_bg=False`, plus `argmax` by `height` rather
than `size`). ADR-0012 landed the rest of ADR-0011's deferred menu
(`select_by_color`, `select_unique_color`, and four `act_on_selection`
actions) but reconfirmed, not resolved, this specific gap: "the audit for
this ADR reconfirmed they remain out of reach." This ADR is that connectivity
-variant pass.

Both tasks' known-correct solvers, from `third_party/arc-dsl/solvers.py`:

- `be94b721`: `subgrid(argmax(objects(I, T, F, T), size), I)` - same
  `argmax(_, size)` → `subgrid` shape as `1f85a75f`/`23b5c85d` (ADR-0011),
  differing only in `diagonal=False` instead of the curated `diagonal=True`.
- `1c786137`: `trim(subgrid(argmax(objects(I, T, F, F), height), I))` - a
  different connectivity triple (`diagonal=False`, `without_bg=False`), a
  different compare function (`height`, not `size`), *and* a `trim` call
  chained after `subgrid`.

## Decision

**Add two new `"select"` actions, each a fixed criterion resolved at
execute-time against the live grid - never an agent-chosen segmentation
variant or compare function as a raw argument, same rule ADR-0011/0012
already established:**

- `select_largest_no_diag` - `dsl.objects(grid, True, False, True)` →
  `dsl.argmax` by `dsl.size` → indices. Verified against `be94b721`.
- `select_tallest` - `dsl.objects(grid, True, False, False)` → `dsl.argmax`
  by `dsl.height` → indices. Verified at the DSL level against `1c786137`,
  but **not** curated as a task - see Consequences for why.

Both were empirically verified before touching `arc_env/task_loader.py`, via
a throwaway script calling `arc_env._dsl.dsl` directly and replaying the
candidate action sequence against every train *and* test pair of both
tasks:

```
be94b721: select_largest_no_diag -> commit_selection      # 4 train + 1 test, all exact
1c786137: select_tallest -> commit_selection -> trim       # 3 train + 1 test, all exact
```

Both sequences reproduce every pair exactly at the DSL level. Only
`be94b721`'s is actually landable, though - see Mechanism and Consequences.

### Mechanism

No `execute()`/`Action` machinery changes at all - unlike ADR-0012 (which
widened `"select"` to accept decoded args), this pass needs nothing new: both
`select_largest_no_diag` and `select_tallest` are zero-arg `"select"`
actions, mechanically identical in shape to `select_largest`/`select_smallest`
(ADR-0011). Each is a thin wrap of one more `dsl.objects(...)` call with a
different fixed `(univalued, diagonal, without_bg)` triple and, for
`select_tallest`, a different fixed compare function (`dsl.height` instead of
`dsl.size`) - exactly the mechanically-similar extension ADR-0011's
Alternatives considered anticipated ("each additional triple is a
straightforward additional `select_*` action pair once worth curating").

**Why `1c786137` doesn't get a curated task despite verifying at the DSL
level:** `commit_selection` ends the episode at the `subgrid` step, mirroring
`commit`'s own crop-and-end-episode semantics (ADR-0011's Decision, point 2;
`arc_env/env.py`'s `is_commit`-style termination check fires on
`action_name in ("commit", "commit_selection")`). `1c786137`'s solver needs
`trim` to run *after* `subgrid`, but by the time `commit_selection` has run,
the episode is already over - a further `trim` action can never actually
execute inside `ArcEnv`'s real step-by-step loop, or `trainers/gp/fitness.py`'s
`run_program` (which mirrors the same early-termination check). This is
invisible to a bare-function replay (calling `action.fn` directly in a loop,
the way this ADR's own verification script and `tests/test_dsl_regression.py`
both do) because neither models episode termination - only `ArcEnv.step`
and `run_program` do. Concretely checked and ruled out:

- Reordering to `trim` the *input* grid before selecting (`select_tallest`
  on `trim(I)`, then `subgrid`) does not reproduce the expected output for
  any of `1c786137`'s 4 train/test pairs - trimming the whole grid's outer
  border is not equivalent to trimming the extracted sub-object's own
  border, since `trim`'s effect depends on what's actually on the grid's
  edge before vs. after cropping to the object's bounding box.
- `subgrid` alone (no `trim` at all) does not match the expected output for
  any pair either (checked directly): the sub-region reachable by
  `commit_selection` is a strict superset of the correct answer for all 4
  pairs, so `trim` is genuinely load-bearing, not a defensive no-op that
  happens not to matter for this task's actual grids.

Landing `1c786137` for real would need a new fused act-on-selection
primitive (e.g. a `trim`-then-end-episode variant of `commit_selection`) -
out of scope for this pass, which is about connectivity variants, not new
act-on-selection primitives. `select_tallest` itself is still added and
curated as an action (verified by direct unit test against hand-constructed
grids, `tests/test_actions.py`), since it's a correct, generally useful
selector independent of this one fixture task being unreachable - the same
"verify the primitive on its own merits" bar ADR-0012 already applied to
`select_by_color`/`select_unique_color`/`delete_selected`/`paint_selected_at`,
none of which had a curated fixture either.

### Curated menu landed this pass

`CURATED_TASK_IDS` grows 29 → 30 (14 same-shape + 16 variable-shape):

- `be94b721` (variable-shape) - `[select_largest_no_diag, commit_selection]`,
  verified exact-match against all 4 train pairs and the 1 test pair.

`1c786137` is deliberately **not** added - see Mechanism above. Action count
grows 36 → 38.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Land `1c786137` anyway, relying on `tests/test_dsl_regression.py`'s bare-function replay passing | Rejected: that test doesn't model episode termination, so it would pass while the task is actually unreachable by any real trainer (PPO via `ArcEnv.step`, GP via `run_program`) - a curated-task entry that can never be solved in practice is worse than not having it, since it silently burns compute (ADR-0008's per-task training cost) on a task the mechanism can't reach, and misrepresents `CURATED_TASK_IDS`'s own contract ("this is exactly the regression-test fixture set" per `task_loader.py`'s docstring) if the regression test can't actually catch the gap. |
| Add a new fused `commit_selection_trimmed` (or similarly named) act-on-selection action to unblock `1c786137` in this same pass | Consistent with ADR-0012's own "small, mechanically similar extension" framing, and would work - but conflates two separable concerns (connectivity variants vs. a new act-on-selection primitive) in one ADR/PR, and this ticket's own scope is specifically the connectivity-variant gap ADR-0011/0012 named. Left as an honest, well-understood follow-up (unlike ADR-0012's four audit-negative actions, this one has a known concrete design and a ready-made fixture task waiting) rather than bundled in. |
| Curate `select_tallest` only if a reachable fixture task existed, deferring the action itself alongside the task | Rejected for the same reason ADR-0012 shipped `select_by_color`/`select_unique_color`/`delete_selected`/`paint_selected_at` without curated fixtures: `select_tallest` is unambiguous and fully checkable by direct unit test, and per-task reachability (an episode-termination artifact) is orthogonal to whether the selector itself is correctly implemented. |
| Fold `without_bg=False` support into `select_largest_no_diag` as a second variant, rather than a separate `select_tallest` action with a different compare function too | `be94b721` only ever needs `diagonal=False` with `without_bg` still `True` and `size` still the criterion - conflating the `without_bg` and compare-function changes into the same action `select_largest_no_diag` doesn't need would widen its behavior beyond what's verified, the same "don't speculatively widen a verified action" discipline ADR-0011's own Slice 1 already applied. |

## Consequences

- `arc_env.actions.execute`'s signature and behavior are **unchanged** by
  this pass - unlike ADR-0012 (which widened `"select"` to thread decoded
  args), both new actions are zero-arg and mechanically identical to
  `select_largest`/`select_smallest`. This is the smallest-footprint of the
  three object-selection ADRs so far.
- Action count grows 36 → 38 (`SELECT` gains `select_largest_no_diag`,
  `select_tallest`); `N_ACTIONS`/`MAX_ARITY` in `trainers/ppo/network.py` and
  `trainers/gp/genome.py` derive from `len(actions.ACTIONS)`/
  `actions.MAX_ARITY` already, so no code change needed there (reconfirms,
  same as ADR-0011/0012's own note, rather than revises).
- `CURATED_TASK_IDS` grows 29 → 30 (14 same-shape + 16 variable-shape);
  compute cost still scales linearly with curated task count (ADR-0008), a
  small addition consistent with the existing accepted tradeoff (`PLAN.md`
  Open risks).
- `select_tallest` ships without a curated-task-level correctness signal the
  way `select_largest`/`select_smallest`/`select_largest_no_diag` have one -
  same accepted tradeoff ADR-0012 already made for
  `select_by_color`/`select_unique_color`/`delete_selected`/
  `paint_selected_at`, not a new category of risk.
- `1c786137` remains a known, unreached task - now for a *structural* reason
  (an episode-termination ordering constraint), not a missing-primitive one.
  A future slice could close this by adding a fused "commit selection, then
  trim, then end episode" act-on-selection action (see Alternatives
  considered) - a small, well-scoped, low-risk follow-up, deliberately left
  out of this pass rather than bundled in.
- This ADR's own verification methodology - checking whether a task's
  candidate program is reachable *inside the actual episode-termination
  model*, not just whether a bare-function replay reproduces the expected
  output - is worth reusing for any future task-reachability audit
  (ADR-0011's and ADR-0012's own audits didn't need to distinguish these,
  since none of their landed sequences had a step after `commit`/
  `commit_selection`).
