# ADR-0021: six more derived actions, landing 7 of F13 Stage 1's cheap findings

- Status: Accepted
- Date: 2026-09-13
- Deciders: repo owner, via conversation 2026-09-13

## Context

F13 Stage 1 (`docs/research/f13-stage1-play-audit.md`) and the broader
structural audit behind ADR-0020 (`docs/questions/f14-multi-selection-
mechanism.md`) between them surfaced a batch of tasks reachable through the
same low-risk "one more derived action" pattern already used 10 times
(`fill_cell`, `canvas`, `canvas_mostcolor`, `swap_two_least_colors`, the 4
self-concat actions) - no new mechanism, no PPO/GP/logging changes, just
new `Action` entries.

Presented as "Bucket A" of the F13 Stage 2 backlog: `c909285e`, `b94a9452`,
`a740d043`, `42a50994` (Stage 1 findings) plus `007bbfb7`, `80af3007`,
`8f2ea7aa` (found during ADR-0020's own audit, previously miscategorized -
see below). Every one of the 7 verified end to end with real proposed
action implementations against every train/test pair before this ADR was
written, not just plausibility-checked.

**One correction to the record.** F13 Stage 1's write-up filed `007bbfb7`
as a "Family 2" case (needs to hold two live grid states). That
characterization was based on testing it as two separate mid-episode
steps. Its actual solver is a pure function of the input grid alone - no
selection, no dependency on any other step - and is fully reproducible as
one bundled derived action. Its two siblings `80af3007`/`8f2ea7aa` share
the same "fractal expansion via self-cellwise-combine" shape (confirmed:
both have exactly the object-count structure needed for existing/new
selectors to reach them). F13's Family 2 roster shrinks from 3 tasks to 2
(`928ad970`, `017c7c7b`) - the real "hold the pre-crop original grid" need
is those 2 plus F11's original `7c008303`/`a68b268e`, unchanged and still
open.

## Decision

Six new actions:

- **`select_leastcolor()`** (`kind="select"`, zero args) - `canvas_mostcolor`/
  `swap_two_least_colors`'s "derived, not agent-chosen" pattern applied to
  `select_by_color`: `dsl.toindices(dsl.merge(dsl.colorfilter(_objects
  (grid), dsl.leastcolor(grid))))`. Needed because the target color varies
  per train/test pair (verified: `c909285e`'s 4 pairs have least-common
  colors 3, 2, 6, 8 - a literal `select_by_color(color)` argument cannot
  reproduce all of them with one fixed sequence, the same constraint
  `CURATED_TASK_IDS`'s fixed-sequence-per-task convention already imposes
  on every other entry).
- **`select_all()`** (`kind="select"`, zero args) - the "always select
  everything" selector Stage 1 named for `a740d043`, reusing the existing
  `(True, True, True)` connectivity: `dsl.toindices(dsl.merge(_objects
  (grid)))`. Confirmed connectivity-invariant for this "merge everything"
  use (also lands `8f2ea7aa`, whose own solver uses `(True, False, True)` -
  merging every object yields the same union of non-background cells
  regardless of diagonal connectivity, verified directly).
- **`select_by_size(size)`** (`kind="select"`, one `SIZE_ARG` - new, decodes
  `raw + 1` into `[1, 30]`, mirroring `DIM_ARG`) - `dsl.toindices(dsl.merge
  (dsl.sizefilter(_objects(grid), size)))`. Lands `42a50994` at `size=1`.
- **`select_largest_multicolor_no_diag()`** (`kind="select"`, zero args) -
  a fifth `objects(...)` connectivity variant, `(univalued=False,
  diagonal=False, without_bg=True)`, argmax by size. `b94a9452`'s solver
  uses this connectivity specifically - it does **not** coincide with the
  existing `select_largest` (`True, True, True`), `select_largest_no_diag`
  (`True, False, True`), or `select_largest_multicolor` (`False, True,
  True`); confirmed by direct comparison this is a genuinely distinct
  fifth variant, not reachable via any already-curated selector despite
  this task having exactly one object either way.
- **`switch_least_most_colors()`** (`kind="transform"`, zero args) -
  `dsl.switch(grid, dsl.leastcolor(grid), dsl.mostcolor(grid))`. The narrow
  near-term candidate F13 Stage 1 named directly: `b94a9452` is fully
  reachable except for this exact last step.
- **`fractal_expand_cellwise(factor)`** (`kind="transform"`, one existing
  `FACTOR_ARG`) - tiles the grid `factor` times in both dimensions (repeated
  `hconcat`/`vconcat`, mirroring the self-concat actions' own style) and
  `cellwise`-combines that with an `upscale(grid, factor)` copy, background
  `0`. A pure function of the current grid alone, the classic ARC
  "fractal expansion" motif. Lands `007bbfb7` by itself; `80af3007`/
  `8f2ea7aa` combine it with a preceding select + `crop_to_selection` (both
  already curated).

**Verified action sequences** (all replayed against every train/test pair
with the exact proposed implementations above):

| Task | Sequence | Shape |
|------|----------|-------|
| `c909285e` | `select_leastcolor` → `crop_to_selection` | variable |
| `b94a9452` | `select_largest_multicolor_no_diag` → `crop_to_selection` → `switch_least_most_colors` | variable |
| `a740d043` | `select_all` → `crop_to_selection` → `replace(1, 0)` | variable |
| `42a50994` | `select_by_size(1)` → `delete_selected` | same |
| `007bbfb7` | `fractal_expand_cellwise(3)` | variable |
| `80af3007` | `select_largest` → `crop_to_selection` → `fractal_expand_cellwise(3)` → `downscale(3)` | variable |
| `8f2ea7aa` | `select_all` → `crop_to_selection` → `fractal_expand_cellwise(3)` | same |

Curated tasks: 48 → 55 (2 more same-shape: 19 → 21; 5 more variable-shape:
29 → 34) - see `arc_env/task_loader.py`'s own docstring for the exact
running tally, corrected against actuals at implementation time.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Give `select_by_size` a fixed `size=1` (matching only `42a50994`'s exact need) instead of a real `SIZE_ARG` | Rejected: a scalar-parameterized selector, generalizable to any size, costs nothing extra over a hardcoded one and matches `select_by_color`'s own precedent exactly - no reason to curve-fit narrower than the natural generalization. |
| Treat `007bbfb7`/`80af3007`/`8f2ea7aa` as still needing the multi-selection/cross-grid mechanism, per F13 Stage 1's original filing | Rejected on direct re-verification (see Context) - they're pure functions of the current grid/selection, no cross-grid state needed at all. Landing them here rather than leaving them mis-filed under a mechanism they don't actually need. |

## Consequences

- Action count: 47 → 53 (ADR-0020's count plus these 6).
- `docs/questions/f13-interactive-editor.md` and `docs/adr/0020-region-
  scoped-dual-selection.md`'s own characterization of `007bbfb7` are
  corrected to reflect this.
- Remaining F13 Stage 1/F14 backlog, unchanged by this pass: `cf98881b`
  (needs a 3-way region split) and `1b2d62fb` (needs a region-scoped
  recolor that doesn't clear the selection) as small, scoped follow-ups to
  ADR-0020; `7c008303`/`a68b268e`/`928ad970`/`017c7c7b` (the "hold the
  pre-crop original grid" need) as a still-open, separate design question.
