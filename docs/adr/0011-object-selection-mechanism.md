# ADR-0011: Object-selection mechanism (ADR-0010 Phase 2)

- Status: Accepted
- Date: 2026-08-31
- Deciders: repo owner (delegated design, per ADR-0010's own deferral), via conversation 2026-08-31

## Context

ADR-0010 sequenced task-coverage scaling into two phases and explicitly declined to
design Phase 2: "Add a mechanism for selecting/manipulating a specific object... This
phase is explicitly **not** designed by this ADR - open questions for its own future
design pass include: how variable per-grid object counts get represented in a
fixed-shape observation/action space, how object selection interacts with the
existing 30x30 scratch-canvas/`commit` mechanism (ADR-0002), and how both PPO's
factored action head and GP's flat gene list represent 'act on object K' consistently
with each other (ADR-0003's shared-substrate requirement)." This ADR is that design
pass, informed by a fresh repo-audit of `third_party/arc-dsl/solvers.py`'s 400
solvers (`solve_<task_id>`) restricted to their non-lambda `Call` targets, to ground
the design in what object-manipulation solvers actually do rather than designing in
the abstract.

**Audit finding 1: `objects` → `argmax`/`argmin` → `subgrid`/`paint`/`merge` is the
dominant pattern, and the compare-function argument is almost always a plain named
unary function, not a built closure.** Across the 79-task object-manipulation
bucket, call frequency: `objects` (57), `subgrid` (31), `merge` (22), `paint` (18),
`argmax`/`argmin` (24 combined), `colorfilter`/`sizefilter` (21 combined). Of the ~35
`argmax`/`argmin`/`valmax`/`valmin` call sites across all 400 solvers, the
overwhelming majority pass the literal named function `size` as the compare
function (`argmax(x1, size)`, `argmin(x1, size)`) - not an inline `lambda` or a
`compose`/`rbind`-built closure. A handful use `height`/`width`/`numcolors`/
`rightmost`/`shape` the same way. This matters directly for ADR-0001's restriction:
`argmax`/`argmin` take a `Callable` argument in general, which is exactly the shape
ADR-0001 excluded (a primitive needing another unpicked primitive to construct) -
but when that Callable is always one of a small, fixed set of named unary functions
(`size`, `height`, `width`, ...), the "closure" is trivial and requires no
composition mechanism at all: it's curatable as a menu of concrete, zero-arg
**selector** actions (`select_largest` = "argmax by size", `select_smallest` =
"argmin by size", etc.), each hardcoding one fixed criterion internally, exactly the
same "derived, not drawn from a solver 1:1" pattern `fill_cell`/the ADR-0010 Phase 1
self-concatenation actions already established.

**Audit finding 2: a small but real slice of the object-manipulation bucket is
*fully* expressible with curated(27) + this small selector/action-on-selection
menu, no other higher-order combinator at all.** Checking all 400 solvers' call
sets (lambda-free) against curated(27) ∪ {`objects`, `argmax`, `argmin`, `size`,
`subgrid`, `cover`, `paint`, `mostcolor`, `colorfilter`, `numcolors`, `height`,
`width`, `uppermost`, `lowermost`, `leftmost`, `rightmost`}, 21 solvers are fully
contained (vs. 24 today) - 3 genuinely new beyond ADR-0010 Phase 1's set, all a
segment-then-extract shape: `1f85a75f` (`objects(I,T,T,T)` → `argmax(_,size)` →
`subgrid`), `23b5c85d` (`objects(I,T,T,T)` → `argmin(_,size)` → `subgrid`), and
`be94b721`/`1c786137` (same shape, different `objects` connectivity args or a
different compare function - see Consequences for why these two are deferred, not
landed, this pass).

## Decision

**A "selected patch" side-channel, threaded through the existing single-grid
`Grid -> Grid` action loop, resolved by a small fixed menu of criterion functions -
not a raw object index, and not a general object/indices observation.**

This directly answers ADR-0010's three open questions:

1. **Variable per-grid object count in a fixed-shape observation/action space:**
   dissolved, not solved head-on. The agent never needs to represent or choose
   among "all N objects" - only ever "what's currently selected", which is a fixed
   30x30 binary mask regardless of how many objects exist or which grid size is in
   play. `objects(grid, ...)`'s actual cardinality is never exposed to the policy or
   encoded in the action `Dict`; it's resolved internally by the environment at
   execute-time, exactly like every existing curated action already resolves its
   own DSL call against the current grid.
2. **Interaction with the 30x30 scratch-canvas/`commit` mechanism (ADR-0002):** a
   new `commit_selection` action mirrors `commit`'s "crop-and-end-episode" semantics,
   using the selected patch's bounding box instead of 4 literal coordinate/dimension
   args (`dsl.subgrid(selected, grid)` = `dsl.crop(grid, ulcorner(selected),
   shape(selected))`) - the natural "extract this one object as my final answer"
   move the audit's dominant `objects → argmax/argmin → subgrid` pattern already
   wants. Any other action that successfully mutates the grid (including the
   existing `canvas`/`commit`/rotation/concatenation actions) clears the selection
   back to "none selected" - a rotation or resize can silently invalidate what a
   stale index set used to point at, and re-selecting is cheap (one more curated
   action), so the safe default is always to force re-selection rather than risk a
   silently-wrong stale selection surviving an unrelated edit.
3. **PPO/GP shared-substrate consistency (ADR-0003):** selector and
   act-on-selection actions are ordinary entries in `arc_env/actions.py`'s
   `ACTIONS` list, with the same `(primitive_index, raw_args)` shape as every
   existing action. `trainers/gp/genome.py`'s flat gene list needs **zero** changes:
   `random_gene` already samples uniformly over `len(actions.ACTIONS)`, and a
   zero-arg selector/act-on-selection action is handled exactly like `vmirror`/
   `identity` already are. PPO's factored action head similarly needs no new head or
   architecture, only a wider `N_ACTIONS`. This reconfirms, rather than revises,
   ADR-0003's shared-substrate premise.

### Mechanism

- `arc_env.actions.Action` gains a `kind: str = "transform"` field (default keeps
  every existing action unchanged): `"transform"` (today's `Grid [, args] -> Grid`,
  unaffected), `"select"` (`Grid -> Indices`, updates the selection, doesn't touch
  the grid), or `"act_on_selection"` (`Grid, Indices [, args] -> Grid`, requires a
  non-empty selection, mirrors an ordinary transform otherwise).
- `arc_env.actions.execute`'s signature grows one parameter and one return value:
  `execute(primitive_index, raw_args, grid, selected=None) -> (new_grid,
  new_selected, decoded_args, valid)`. A `"select"` action's `fn(grid) -> Indices`
  is invalid (no-op, `new_selected` = unchanged `selected`) if it finds no objects
  to select. An `"act_on_selection"` action is invalid if `selected` is falsy. A
  `"transform"` action's successful case now also resets `new_selected` to `None`
  (see point 2 above); its failure case leaves `selected` untouched, matching how
  its failure case already leaves `grid` untouched.
- `ArcEnv` gains `self._selected: frozenset | None` episode state (reset to `None`
  in `reset()`, threaded through `step()`'s call to `actions.execute`). The
  observation grows from a single `(30, 30)` grid channel to a `(2, 30, 30)` stack:
  channel 0 unchanged (color id, 0-9, `PAD_VALUE`=10 for padding), channel 1 a
  binary "is this cell currently selected" mask, 0 everywhere when nothing is
  selected. `is_commit`-style episode termination (currently gated on
  `action_name == "commit"`) also fires on `commit_selection`.
- `trainers/ppo/network.py`'s `ActorCritic._encode` concatenates one more channel
  (the raw 0/1 selected mask, no embedding needed - it's already numeric) before
  `input_proj`; `input_proj`'s input channel count grows from `EMBED_DIM + 1` to
  `EMBED_DIM + 2`. No other network change: the factored action head's arity
  masking already generalizes to zero-arg selector/act-on-selection actions for
  free, the same way it already does for `vmirror`/`identity`.
- `trainers/ppo/rollout.py` needs no change at all - it treats the observation as an
  opaque array shape throughout (`np.stack`, `torch.as_tensor(...).unsqueeze(0)`),
  never hardcoding `(30, 30)`.
- `trainers/gp/fitness.py`'s `run_program` (the fast fitness-evaluation path, which
  calls `actions.execute` directly rather than going through `ArcEnv.step`) threads
  `selected` the same way `ArcEnv.step` does, and its early-termination check
  (currently `action.name == "commit"`) also fires on `commit_selection`.
  `trainers/gp/replay.py`'s `program_to_episode_trace` needs no change - it already
  goes through `ArcEnv.step`, which owns the new state internally.
- Reward (`arc_env/reward.py`) is unaffected: still purely a function of `grid`
  before/after vs. `target`. The selection is control/scratch state, not part of
  what's being scored - scoring the selection itself would conflate "did the agent
  pick a good editing target" with "does the grid match", two different signals ADR
  -0005's dense reward was never designed to entangle.

### Curated menu landed this pass (Slice 1)

A deliberately small, literally-verified menu - proving the mechanism end-to-end
over a large speculative one, the same call ADR-0010 Phase 1 made for
self-concatenation:

- `select_largest` / `select_smallest` (0-arg): `dsl.objects(grid, True, True,
  True)` (univalued, diagonal-connected, background-excluded - the one segmentation
  variant curated this pass, chosen because it's what both new fixture tasks below
  use verbatim) → `dsl.argmax`/`dsl.argmin` by `dsl.size` → `dsl.toindices` of the
  chosen object.
- `commit_selection` (0-arg): `dsl.subgrid(selected, grid)`, then ends the episode
  like `commit`.

Two new curated tasks, verified exact-match against every train *and* test pair:
`1f85a75f` (`[select_largest, commit_selection]`, solver: `subgrid(argmax(objects
(I,T,T,T), size), I)`) and `23b5c85d` (`[select_smallest, commit_selection]`,
solver: `subgrid(argmin(objects(I,T,T,T), size), I)`) - both variable-shape.
`CURATED_TASK_IDS` grows 24 → 26 (12 same-shape + 14 variable-shape).

## Alternatives considered

| Option | Why not |
|--------|---------|
| Raw "select object index K" action (ADR-0010's own original sketch: a discrete index into `objects(grid, ...)`'s output, plus a "which object is this cell part of" observation channel) | The audit shows solvers essentially never pick objects by an arbitrary index - they pick by a computed criterion (`argmax(_, size)`, `colorfilter(_, color)`). An index-based action would need the agent to *first* infer int-to-object correspondence from an unordered-set enumeration that changes shape/order across grids, solving a harder, less-grounded problem than the one 79-task bucket's own solvers actually pose. It also reintroduces exactly the "variable per-grid object count in a fixed space" difficulty ADR-0010 flagged as unresolved, rather than dissolving it. |
| Represent every segmented object as its own observation channel/slot (e.g. a fixed max-object-count encoding) | Same "variable count in fixed shape" problem, just pushed into the observation instead of the action; also multiplies observation size for no payoff this slice's mechanism needs, since only *one* patch is ever acted on at a time. |
| Curate the full intended menu now (all criterion selectors: `select_by_color`, `select_unique_color`, `select_tallest`/`select_widest`, plus act-on-selection actions `delete_selected`/`recolor_selected`/`move_selected`/`paint_selected_at`) | Rejected for this pass, not rejected outright - see Consequences. A large untested menu risks the same failure mode Phase 1 explicitly avoided (a `mostcolor`/`leastcolor`-style primitive whose correctness can't be spot-checked against a literal, verified fixture task). Landing the minimal 3-action mechanism first, end-to-end and test-covered, de-risks the larger menu's implementation (execute()'s signature change, the new observation channel, GP/PPO glue) before spending design/verification effort on more selectors. |
| Multiple curated `objects(...)` connectivity variants (to also literally reach `be94b721`'s `objects(I,T,F,T)` and `1c786137`'s `objects(I,T,F,F)` + `argmax(_,height)`) | Deferred, not designed away - each additional `(univalued, diagonal, without_bg)` triple is a straightforward additional `select_*` action pair once worth curating, same mechanical pattern as this pass's two. Holding at one variant keeps this pass's verified surface small; revisit once the single-variant menu's practical training payoff is measured (this ADR's Consequences). |

## Consequences

- `arc_env.actions.execute`'s signature and return shape change (new `selected`
  parameter and return value) - every direct caller updated in this pass:
  `arc_env/env.py` (`ArcEnv.step`), `trainers/gp/fitness.py` (`run_program`), and
  `tests/test_dsl_regression.py`'s replay harness. This is the one genuine breaking
  change Phase 2 makes that Phase 1 didn't need (Phase 1 was purely additive to
  `ACTIONS`, no signature change) - worth flagging since it's the main way this
  phase differs in kind, not just scale, from Phase 1.
- The observation shape changes from `(30, 30)` to `(2, 30, 30)` - a breaking
  change to any saved checkpoint's `ActorCritic.input_proj` weights (input channel
  count changed). No migration path is provided; this milestone's per-task,
  solve-time training (ADR-0008) already never shares a checkpoint across a
  training-code change, so this is consistent with existing practice, not a new
  gap.
- The visualizer is **not** updated to render the selection mask this pass -
  episode replay still shows only `grid_before`/`grid_after` per step (ADR-0006's
  existing schema), so a `select_largest` step will visibly do nothing to the
  rendered grid even though it changed hidden state. Documented as a deliberate,
  known gap (mirroring Phase 1's "the visualizer needs no changes to replay it"
  claim, which no longer fully holds for *this* phase) rather than silently
  shipped - a future slice should add a selection-overlay render, low-risk since
  the episode log's `grid_before`/`grid_after` pairing already has everywhere it
  would need to plug in.
- `mostcolor`-style content-dependent primitives, which ADR-0010 Phase 1 explicitly
  deferred as "unsafe to curate as one fixed literal action sequence" (a single
  static program applied identically to every train/test pair, per the curated-task
  -audit method), are **not** subject to that same objection inside a `"select"`
  action's `fn`: `dsl.objects(..., without_bg=True)` already calls `dsl.mostcolor`
  internally to find the background color, computed fresh from whatever grid the
  policy is looking at *this step* - safe here because Phase 2's actions are chosen
  step-by-step by a learned policy conditioned on the live grid, not transcribed
  once as a fixed sequence for literal replay. This resolves, rather than
  reopens, Phase 1's deferral for the object-selection context specifically.
- Compute cost still scales linearly with curated task count (ADR-0008); 24 → 26
  tasks is a small addition, consistent with the existing accepted tradeoff
  (`PLAN.md` Open risks).
- Deferred, not designed: the larger act-on-selection menu (`delete_selected`,
  `recolor_selected`, `move_selected`, `paint_selected_at`), a `select_by_color`/
  `select_unique_color` pair (covering the `colorfilter`-based selection pattern,
  the second-most-common one in the audit), additional `objects(...)` connectivity
  variants, and any visualizer selection-overlay work. Each is a small, mechanically
  similar extension of this ADR's mechanism (same `"select"`/`"act_on_selection"`
  `Action.kind`s, no further architecture change) - intentionally left as follow-up
  slices rather than bundled into this one, per the same "prove the mechanism
  first" reasoning in Alternatives considered.
