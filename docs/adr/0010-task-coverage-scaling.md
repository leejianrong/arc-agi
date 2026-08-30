# ADR-0010: Task-coverage scaling past the curated 16 — broaden primitives, then object selection

- Status: Accepted
- Date: 2026-08-29
- Deciders: repo owner, via conversation 2026-08-29

## Context

`arc_env/task_loader.py`'s `CURATED_TASK_IDS` (16 tasks: 11 same-shape, 5
variable-shape) was derived, not hand-picked: a task qualifies iff its
`third_party/arc-dsl/solvers.py` known-correct solver calls only primitives in
`arc_env/actions.py`'s curated groups - `Grid [, scalar args] -> Grid` primitives,
excluding higher-order combinators (`compose`/`chain`/`fork`/etc., ADR-0001) and
anything needing an `Object`/`Indices`/`Callable` argument.

A repo-audit script (2026-08-29, parsing `solvers.py`'s 400 `solve_<task_id>`
function bodies for calls to a higher-order-combinator name list and a separate
object-manipulation-function name list) found:

| Bucket | Count | % of 400 |
|--------|-------|----------|
| Calls a higher-order combinator (`apply`/`mapply`/`compose`/`chain`/`fork`/`rbind`/`lbind`/`power`/etc.) | 260 | 65% |
| Calls an object-manipulation primitive (`objects`/`partition`/`colorfilter`/`argmax` over objects/etc.), no higher-order combinator | 79 | 20% |
| Neither | 61 | 15% |

The "neither" bucket (61) is the ceiling of tasks reachable by ADR-0001's
scalar-args-only restriction *in principle* - only 16 of those 61 are actually
curated today, because the curated action space (23 actions) is a strict subset of
arc-dsl's full scalar-arg-primitive catalog (arc-dsl has on the order of 150
primitives total, most of them not yet wrapped as an action here). The 79-task
object-manipulation bucket needs a genuinely new mechanism (some way to represent
"the object currently selected/being acted on") that doesn't exist in this
env/action space yet.

This was flagged by the repo owner as a decision needing its own ADR, not a code
change to just start on - resolved in the 2026-08-29 conversation as: pursue **both**
directions, sequenced.

## Decision

Two phases, mirroring ADR-0003's "cheaper/lower-risk first, bigger/riskier as
fast-follow" build-order pattern:

**Phase 1 - broaden curated primitives (near-term, no new mechanism).** Extend
`arc_env/actions.py`'s curated action groups (`ZERO_ARG`/`ONE_ARG`/`TWO_ARG`/
`THREE_ARG`/`FOUR_ARG`) with more of arc-dsl's `Grid [, scalar args] -> Grid`
primitives - still zero-to-four scalar typed arguments, still no `Object`/
`Indices`/`Callable` arguments, same restriction ADR-0001 already established. No
change to the observation space, the action `Dict` shape, `trainers/ppo/network.py`'s
factored action head, or `trainers/gp/genome.py`'s flat gene-list representation -
just a longer `ACTIONS` list, `MAX_ARITY`/`RAW_ARG_RANGE` unaffected or adjusted
mechanically. Once a concrete expanded primitive set is chosen, re-run the same
solver-body filter `task_loader.py`'s module docstring already describes over the
full 400 solvers to compute the new curated task count (expected to move toward, not
necessarily all the way to, the 61-task ceiling - some of those 61 may use scalar-arg
primitives not worth curating for other reasons, e.g. redundant with existing
actions).

**Phase 2 - object selection (fast-follow, real design work, not yet specified).**
Add a mechanism for selecting/manipulating a specific object, to reach the larger
79-task object-manipulation bucket: e.g., a discrete "select object index" action
over objects detected by arc-dsl's `objects`/`partition`-style segmentation
(univalued/diagonal-connectivity/without-background variants already exist in the
vendored DSL), exposed as a new observation channel (a per-cell "which detected
object is this cell part of" index, capped at some maximum object count) plus new
actions parameterized by that selection instead of raw `(row, col)` coordinates.
This phase is explicitly **not** designed by this ADR - open questions for its own
future design pass include: how variable per-grid object counts get represented in
a fixed-shape observation/action space, how object selection interacts with the
existing 30x30 scratch-canvas/`commit` mechanism (ADR-0002), and how both PPO's
factored action head and GP's flat gene list represent "act on object K" consistently
with each other (ADR-0003's shared-substrate requirement).

## Alternatives considered

| Option | Why not |
|--------|---------|
| Hold at 16 tasks this milestone (R0's success criterion is measurable learning, not breadth) | Rejected as the sole direction - Phase 1 is cheap (no new mechanism, bounded, mechanical) relative to the coverage it can add, so there's no reason to leave it on the table even if ambition elsewhere is otherwise capped. |
| Deepen via `re-arc` curriculum only (harder/more varied instances of the same 16 tasks, no new task IDs) | Rejected as the sole direction, for the same reason - it doesn't address breadth at all, and it's already a partially-independent lever (`arc_env/re_arc.py`, `SLICES.md` V2 step 3) that composes with either phase here rather than substituting for them. |
| Object selection only, skip broadening primitives | Rejected - Phase 1's ~61-task ceiling is reachable with zero new representational risk; sequencing it first captures easy coverage before spending design effort on the harder mechanism, exactly the ADR-0003 precedent for how this repo sequences cheap-first vs. risky-later work. |

## Consequences

- Phase 1 is implementable without any further design pass: pick additional
  arc-dsl primitives, add them to `arc_env/actions.py`'s groups, re-run the solver
  filter, update `CURATED_TASK_IDS`, extend `tests/test_dsl_regression.py`'s fixture
  coverage accordingly (ADR-0001's existing regression-test pattern already handles
  new fixture tasks with zero new test-writing beyond adding rows to the table).

### Phase 1 implementation (2026-08-29)

Landed: 4 new zero-arg actions - `hconcat_self`, `hconcat_self_vmirror`,
`vconcat_self_hmirror_top`, `vconcat_self_hmirror_bottom` - each fixing
`dsl.hconcat`/`dsl.vconcat`'s second `Grid` argument as a function of the
first (self, or a mirror of self), the same "derived, not a 1:1 solver call"
pattern `fill_cell` already established. A repo-audit script (AST-parsing
`solvers.py`, tracking which extra, non-curated calls a candidate solver
needs beyond the existing 23 actions) found these 4 primitives fully cover 7
solvers that only differ in call order/mirroring, plus one already-expressible
`commit` call (`5bd6f4ac`: `crop(I, tojvec(SIX), THREE_BY_THREE)`, a
constant-only literal never touching the grid, exactly `d10ecb37`'s pattern) -
8 new tasks, verified exact-match against every train *and* test pair before
adding: `a416b8f3`, `6d0aefbc`, `c9e6f938`, `4c4377d9`, `6fa7a44f`,
`8be77c9e`, `f25ffba3`, `5bd6f4ac`. `CURATED_TASK_IDS` grew 16 → 24 (12
same-shape + 12 variable-shape - `f25ffba3` nets back to its input shape
despite using a shape-changing action mid-sequence, so it counts as
same-shape, not variable). `arc_env/actions.py::execute` also gained a
generic oversized-grid guard (`new_h/new_w > MAX_GRID_DIM` → invalid/no-op)
since these new actions can double a dimension past the 30x30 canvas -
previously only `upscale`/`hupscale`/`vupscale`/`commit` had bounds checks
because no other existing action could grow a grid at all.

The audit also surfaced solvers needing exactly one more primitive
(`mostcolor`, `leastcolor`, `cellwise`, `ofcolor`) that were deliberately
**not** added this pass: each depends on the grid's actual content
(`mostcolor(I)`/`leastcolor(I)` compute a color that varies by instance,
`ofcolor`/`cellwise` build/compare `Indices`/multi-grid state), so a single
literal transcription into `CURATED_TASK_IDS` (one fixed action sequence
applied identically to every train/test pair) either isn't safe in general
or strays into Phase 2's object/indices territory - out of scope for a
"broaden with simple primitives" pass. Remains available as a smaller
fast-follow if worth revisiting.

- Phase 2 needs its own design pass (observation-space change, action-space change,
  both trainers' representations must agree) before implementation - this ADR scopes
  the target and constraints, not the mechanism.
- The 260/79/61-out-of-400 split is a repo-audit fact computed 2026-08-29 by a
  simple call-name filter over `solvers.py`, not a guarantee about exactly which or
  how many tasks Phase 1 will actually add - that depends on which additional
  primitives get curated and is only known precisely once that set is chosen and the
  filter re-run.
- Compute cost still scales linearly with curated task count (ADR-0008's per-task
  training) - adding tasks via either phase multiplies total training time/runs
  accordingly, unchanged from the existing accepted tradeoff (`PLAN.md` Open risks).
