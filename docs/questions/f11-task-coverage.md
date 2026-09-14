# F11: task-coverage scaling past the curated 16

**Status:** decided (user, 2026-08-29), landed incrementally through ADR-0019.

The question: how do we grow past the 16 curated tasks the original action
space covered? Broaden the primitive set, add object selection, hold at 16,
or lean on `re-arc`-generated variations of the same 16 tasks?

**Decision:** do both, in sequence. Phase 1 (near term) extends the curated
action space with more scalar-arg-only `arc-dsl` primitives, no new
mechanism. Phase 2 (fast follow) adds an object-selection mechanism to
reach the larger object-manipulation bucket Phase 1 can't touch.

As of ADR-0019, 40 tasks are curated (19 same-shape, 21 variable-shape).
The 2026-09-06 follow-up section below concludes that 400/400 is out of
reach under this architecture.

## The original audit (2026-08-29)

Parsed `third_party/arc-dsl/solvers.py`'s 400 `solve_<task_id>` bodies for
calls to higher-order-combinator vs. object-manipulation function names:

- 260/400 (65%) need a higher-order combinator, excluded per ADR-0001.
- 79/400 (20%) need object selection or manipulation, no representation yet.
- 61/400 (15%) need neither, and of those 61, only 16 were curated at the
  time.

This is the basis for ADR-0010's task-coverage-scaling decision above.

## Phase 1: broadening primitives (2026-08-29)

Landed 4 new self-concatenation actions: `hconcat_self`,
`hconcat_self_vmirror`, `vconcat_self_hmirror_top`,
`vconcat_self_hmirror_bottom`. Curated tasks went from 16 to 24.

## Phase 2: object selection (2026-08-31 onward)

Slice 1 (2026-08-31) added a "currently selected patch" side-channel:
`select_largest`, `select_smallest`, `commit_selection`. 24 → 26 curated
tasks.

The deferred full menu landed 2026-09-04 as ADR-0012: `select_by_color`,
`select_unique_color`, `delete_selected`, `recolor_selected`,
`move_selected`, `paint_selected_at`. 26 → 29 curated tasks.

ADR-0013 (2026-09-05) added two more `objects(...)` connectivity variants:
`select_largest_no_diag` and `select_tallest`. 29 → 30 curated tasks.
`select_largest_no_diag` has a verified fixture (`be94b721`).
`select_tallest` doesn't: its fixture (`1c786137`) needs a `trim` step
after `commit_selection`, but `commit_selection` already ends the episode
at the crop, so that `trim` can never actually run in the real
step-by-step loop, even though a bare replay outside `ArcEnv` looks like it
reproduces the answer. Recorded as a well-scoped follow-up, not landed yet
(ADR-0015 picks it back up below).

## The 2026-09-05 refresh: how far can this actually go?

The user asked to make this more urgent and push toward all 400 training
tasks. We re-ran the coverage audit (`scripts/audit_action_coverage.py`,
which checks `dsl.py` type hints for a `Callable`-typed parameter rather
than a fixed name list, so it stays accurate as new higher-order primitives
are or aren't added) against the current, post-ADR-0013 action space and
all 370 still-uncurated tasks:

- 269/370 (73%) call a higher-order combinator (`mapply`, `compose`,
  `fork`, `apply`, `lbind`, `rbind`, `chain`, `sfilter`, and others),
  structurally excluded per ADR-0001, unchanged from before.
- 10 look reachable today by primitive name alone. These are re-check
  candidates, not confirmed: name overlap doesn't guarantee the actual
  parameterization or composition is curated, see `select_tallest`/
  `1c786137` above.
- 21 need exactly one new primitive by name. Top clusters: `ofcolor`
  unlocks 5, `first` unlocks 5, `sizefilter` unlocks 4.
- 18 need exactly two new primitives.
- 52 need three or more.

The revised ceiling: 400/400 is very unlikely under this architecture. 73%
excluded is higher than the original audit's 65%, meaning growth so far has
disproportionately drained the "neither" bucket, leaving what's left
skewed even harder toward the structurally excluded. Continuing
incrementally, highest-leverage near-miss clusters first (`ofcolor` and
`first` next), remains the right near-term move. But reaching all 400
training tasks, let alone the harder, solver-free evaluation set, would
need a fundamentally different representation. See
[F12](f12-llm-driven-approaches.md).

## ADR-0015: the ofcolor + first pass (2026-09-06)

A mixed result, worth recording precisely.

`first`'s 5 flagged tasks turned out to share one root cause: the same
"`commit_selection` ends the episode too early" wall ADR-0013 already hit
for `1c786137`. Fixed generically with a new, non-terminal
`crop_to_selection` act-on-selection action (the identical crop
`commit_selection` does, under a different name, so the termination check
by literal action name doesn't fire) plus one new `objects(...)`
connectivity variant, `select_largest_multicolor` (`univalued=False`).

This landed 6 tasks at once: `1c786137`, plus `a79310a0` (which needed zero
new primitives; the audit's name-level check had simply missed that its
solver already maps onto curated actions), `28bf18c6`, `f25fbde4`,
`2013d3e2`, and `7468f01a` (which needed the new mechanism). Curated tasks
went 30 → 36.

`ofcolor`'s 5 flagged tasks were a different story. Each needed a different
additional capability the audit's name-level check couldn't see: two need
multiple simultaneous selections across different coordinate spaces or
grids, two need a selection reused across several "stamp a shifted copy"
operations (which needs a directional act-on-selection primitive that
didn't exist yet), and one needs a selection checked against an unrelated
grid. None of these is a bare "add one primitive" fix, so none landed this
pass.

The next concrete candidate, named in ADR-0015 itself: a directional
"stamp" act-on-selection primitive, which would unlock 2 of those 5
(`a9f96cdd`, `d364b489`) on its own.

## ADR-0016: the directional stamp (2026-09-06)

Landed that candidate. `stamp_selected`
(`dsl.fill(grid, color, dsl.shift(selected, DIRECTION))`) plus a new,
separate 8-direction menu (4 cardinal, 4 diagonal). `a9f96cdd` needs
diagonals `move_selected`'s existing cardinal-only menu doesn't have, so
this got a fresh menu rather than widening that one and risking its own
`25ff71a9` fixture.

Unlocked `a9f96cdd` and `d364b489` exactly as scoped, both verified by
direct replay against every train and test pair. Curated tasks went 36 →
38.

The remaining 3 `ofcolor`-flagged tasks (`7c008303`, `9ecd008a`,
`a68b268e`) still needed their own separate capabilities, unaddressed by
this pass.

## ADR-0019: two more derived actions (2026-09-06)

Landed two more one-new-primitive tasks from the same 2026-09-05 audit's
21-task bucket: `5582e5ca` (`mostcolor`) and `aabf363d` (`leastcolor`).
Landed as two more derived actions, in the same "derived, not drawn from a
solver 1:1" family `fill_cell`/`canvas` already established:
`canvas_mostcolor` (`canvas` fed a grid-derived color instead of an
agent-chosen one) and `swap_two_least_colors` (a fully self-contained
zero-arg composition of `leastcolor`/`replace`). No new mechanism. Curated
tasks went 38 → 40 (19 same-shape, 21 variable-shape; both new tasks
turned out to be same-shape on computational re-check, correcting an
initial guess that `5582e5ca` was almost certainly variable-shape).

## The 3 remaining ofcolor tasks (2026-09-06 follow-up)

Independent re-verification of the 3 remaining `ofcolor`-flagged tasks
(`7c008303`, `9ecd008a`, `a68b268e`), plus an explicit go/no-go call on
`9ecd008a`, prompted by the coverage audit still flagging all 3 as open.

All 3 known-correct solvers were replayed independently (bare `dsl` calls,
not `actions.execute`, since none is curated) against every train and test
pair. All reproduce the expected output exactly, which confirms rather than
just restates ADR-0015's characterization:

- `7c008303` computes one `Indices` set (`ofcolor(I,3)` → `subgrid`), then
  a second, unrelated `Indices` set in that sub-grid's own coordinate space
  (`ofcolor` on the crop), and fills a third grid built through a wholly
  different pipeline (`replace` twice, `compress`, `upscale(3)`). We
  confirmed computationally that the crop's shape (6x6 in the checked
  fixture) and the final result's shape (also 6x6) only coincide because
  of this task's specific geometry, not any general relationship.
- `a68b268e` computes three separate `ofcolor` calls against three
  different quadrant sub-grids (`lefthalf(tophalf(I))`,
  `righthalf(tophalf(I))`, `lefthalf(bottomhalf(I))`, each independently
  confirmed 4x4 in the fixture), and fills their indices onto a fourth
  quadrant. A genuine simultaneous-multi-selection-across-grids
  requirement.

Both are reconfirmed as needing the multi-selection/cross-grid-index
mechanism the original scope boundary named: PPO's factored action head,
GP's flat gene list, episode logging, and the visualizer's selection
overlay would all need to represent more than one live named selection at
once. That's a redesign bigger than anything ADR-0011 through ADR-0019
made. Correctly out of scope for an incremental pass, and not attempted.

`9ecd008a` got traced further this pass. `vmirror(I)` is computed once;
`ofcolor(I,0)` is computed against the original, pre-mirror `I`; `subgrid`
crops the mirrored grid using those stale, pre-transform indices. We found
that `ofcolor(I,0)` selects a compact 9-cell, 3x3 hole (color 0 is the
least-common color in the checked fixture, occupying a tight bounding box)
in an otherwise-dense 16x16 grid, and the output is exactly that 3x3 patch.
This is the classic ARC "symmetry repair" motif: a masked region's true
content recovered from the grid's own mirror-symmetric counterpart, not an
arbitrary coincidence.

That makes the task well-motivated. But the mechanism it needs is the
specific one ADR-0015 flagged as tension with ADR-0011: a selection
computed before a transform, deliberately kept alive, and reused against
the post-transform grid.

**Explicit go/no-go: no-go, not implemented.** The only way to reach this
inside the current one-selection-slot model is an `act_on_selection`-kind
action that performs an ordinary whole-grid transform (say, `vmirror`)
while requiring, but never actually using, the current selection, purely
to exploit `execute()`'s existing rule that only the generic "transform"
kind clears the selection (`arc_env/actions.py`, ADR-0011). Every actual
`act_on_selection` action shipped so far (`crop_to_selection`,
`delete_selected`, `stamp_selected`, and the rest) consumes `selected` as a
real input to its computation. This hypothetical action's function body
would ignore `selected` entirely: a selection would be a precondition for
no reason connected to what the action does, a shape no existing action
has and a confusing precedent to set.

Worse, landing it as a general menu entry, not a fixture-only special case,
would put a "transform the whole grid, keep whatever was selected before,
no matter how the transform changed the grid" action into the same policy
and genome action space every other task's search draws from. That
revives, generally and permanently, exactly the "silently wrong stale
selection surviving an unrelated edit" failure mode ADR-0011 was written to
rule out, on the strength of one task where the staleness happens to be
exactly correct.

This is a single-task special case dressed up as a general mechanism, not
a generally useful addition, consistent with the `select_tallest`/
`1c786137` precedent above and ADR-0015's own `ofcolor`-cluster precedent:
both cases where a well-reasoned "don't land this" was the right call.

All 3 tasks remain uncurated. No code changed this pass.

## Where this feeds next

`7c008303` and `a68b268e` still needed the multi-selection/cross-grid
mechanism this file kept deferring. F13's Stage 1 pass
(`docs/questions/f13-interactive-editor.md`) independently found more
evidence for a broader version of the same gap: the real requirement looked
like "more than one live grid state, addressed and recombined somehow,"
not the narrower "more than one named selection." That fed F14's
region-scoped dual-selection design (ADR-0020), which explicitly scoped
`7c008303`/`a68b268e` out as needing a different, heavier capability
("hold the pre-crop original grid alongside a derived crop").

**2026-09-14 resolution (ADR-0023):** a broader audit + rigorous
`re-arc`-generalization check found that framing was wrong - neither task
needs a new mechanism at all. `a68b268e` landed as one ordinary derived
action (`fill_quadrant_from_colors`, 3 `COLOR_ARG`s, verified 30/30 against
fresh `re-arc` instances); `7c008303` turned out to be a no-go for a
different reason than originally thought - not "needs more state," but
"its known-correct solver is narrower than `re-arc`'s own generative
concept of the task" (1/30 even under brute-force arg search), the same
disposition as `select_tallest`/`1c786137` and `9ecd008a`. See ADR-0023 for
the full audit and verification.

## ADR-0024: the `free_by_name` re-check (2026-09-14)

A same-day follow-up: re-ran the audit against the post-ADR-0023 baseline
(60 curated) and re-checked its 20 `free_by_name` tasks by hand (the
bucket the audit's own docstring flags as "reachable by primitive name
alone, not a guarantee"). 7 shared one of three fixed geometric-tiling
shapes (2 identical-solver pairs plus 2 singletons), landed as 4 new
zero-arg derived actions, verified 30/30 general against fresh `re-arc`
instances each: `quad_rotate_tile` (`46442a0e`, `7fe24cdd`),
`quad_mirror_tile` (`3af2c5a8`, `62c24649`, `67e8384a`),
`stack3_vmirror_tile` (`8d5021e8`), and `left_third` (`2dee498d` - this one
needed zero new logic at all, just exposing ADR-0022's existing internal
`_left_third` region helper as its own standalone action). Two more
candidates (`0520fde7`, `a699fb00`) were checked and found not to
generalize (1/30, 5/30) - same disposition as this file's earlier no-gos,
not pursued further. Curated tasks went 60 → 67. See ADR-0024 for the full
account, including the 11 `free_by_name` tasks still unexamined past a
first read.

**Landed by:** ADR-0010, ADR-0011, ADR-0012, ADR-0013, ADR-0015, ADR-0016,
ADR-0019, ADR-0023, ADR-0024.
