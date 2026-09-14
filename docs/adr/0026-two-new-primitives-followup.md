# ADR-0026: `two_new_primitives` bucket - 13 more derived actions, 10 new `arc-dsl` primitives

- Status: Accepted
- Date: 2026-09-14
- Deciders: repo owner, via conversation 2026-09-14

## Context

Re-ran `scripts/audit_action_coverage.py` fresh against the post-ADR-0025
baseline (77 curated, 323 remaining). `free_by_name` and `one_new_primitive`
were both exhausted by ADR-0025 (everything remaining in either bucket was
already a dispositioned no-go from a prior ADR, except one fresh
`one_new_primitive` task, `67a423a3`/`neighbors` - too small to be its own
pass). The next-highest-value work is `two_new_primitives` (16 tasks, 2
already dispositioned no-go by ADR-0023: `c9f8e694`, `017c7c7b`). This pass
folds `67a423a3` in with the 14 fresh `two_new_primitives` tasks - 15
candidates total, split into two clusters and verified by two parallel
research passes (this repo's 2-concurrent research-subagent cap):

- **Cluster A** (branch/canvas/computed-scalar constructions, expected to
  collapse toward 0-1 new primitives): `44f52bb0`, `7b7f7511`, `ff805c23`,
  `d0f5fe59`, `d631b094`, `1fad071e`, `ac0a08a4`, `b91ae062`, `77fdfe62`.
- **Cluster B** (expected to need a genuinely new geometric primitive):
  `67a423a3`, `5c0a986e`, `88a10436`, `88a62173`, `b548a754`, `1bfc4729`.

Both clusters were verified bare-`dsl` against every train/test pair, then
through the real `arc_env.actions.execute()` -> `ArcEnv.step()` path, then
against 30-300 fresh `re-arc` instances per task, brute-forcing argument
assignments where relevant - the same bar every ADR in this lineage has
held to.

**The dominant finding this pass, more so than any prior one**: for most of
these 15 tasks, the literal `solvers.py` program's read of "what primitive
does this need" was itself misleading, not just its hardcoded constants.
ADR-0025 already found hardcoded constants (a color, a background, a crop
shape) frequently needed correcting to a grid-derived value. This pass
found the same is true of *which primitive the task actually needs* -
`neighbors` (single-hole-cell) turned out wrong for `67a423a3` (the real
generator concept needs `outbox` over a perpendicular-band intersection);
`portrait` turned out wrong for `7b7f7511` (the real rule is a structural
tophalf==bottomhalf equality check, independent of aspect ratio);
`lrcorner` turned out unneeded for `5c0a986e` once processed per-object
instead of per-merged-blob; `astuple`/`neighbors`/`lrcorner` all turned out
to be either trivial Python or simply the wrong primitive for the actual
generative concept. Every landing task below required tracing the actual
`re-arc` generator source, not just re-reading the solver more carefully -
a materially higher verification cost than ADR-0025, and the reason this
pass's yield-per-task-read is lower than prior passes despite a similar
candidate count.

## Decision

**14 of 15 candidates land, as 13 new derived actions** (one action,
`upscale_by_numcolors_minus_one`, is shared by `ac0a08a4`/`b91ae062`), no
new mechanism, no new `Action.kind`. **10 new `arc-dsl` primitives**:
`numcolors`, `backdrop`, `outbox`, `shoot`, `center`, `normalize`,
`leastcommon`, `box`, `hfrontier`, `connect`.

### Cluster A - 7 actions, 8 tasks

- **`canvas_by_symmetry()`** (zero-arg) - a 1x1 canvas colored `1` if the
  grid is symmetric under any of `hmirror`/`vmirror`/`dmirror`/`cmirror`,
  else `7`. Corrects the literal solver's `vmirror`-only check: `re-arc`
  applies a random post-hoc rotation/reflection, so the symmetry can show
  up on any axis. Needs no new primitive. Unlocks `44f52bb0`. 8/8 real,
  60/60 `re-arc`.
- **`tophalf_or_lefthalf_by_equality()`** (zero-arg) - `tophalf(grid)` if
  `tophalf(grid) == bottomhalf(grid)`, else `lefthalf(grid)`. Corrects the
  literal solver's `portrait`/`branch` read entirely: the real invariant is
  a structural equality check between the two candidate halves, not grid
  aspect ratio (naive `portrait()` is 17/30 on `re-arc`). Needs no new
  primitive. Unlocks `7b7f7511`. 4/4 real, 30/30 `re-arc`.
- **`mirror_crop_by_rectangle_marker()`** (zero-arg) - auto-detects the
  marker color as whichever palette color's cells form an exact solid
  rectangle (`ofcolor(grid, c) == backdrop(ofcolor(grid, c))`, new
  `backdrop` primitive - shared with `b548a754` below), crops that color's
  bbox out of both `hmirror(grid)` and `vmirror(grid)`, and picks whichever
  crop no longer contains the marker color. Corrects the literal solver's
  hardcoded marker color `1` (0/30 on `re-arc` uncorrected). Unlocks
  `ff805c23`. 4/4 real, 30/30 `re-arc`.
- **`diagonal_canvas_by_object_count()`** (zero-arg) - builds an NxN
  canvas (N = foreground object count) colored by the grid's own
  `mostcolor`/`leastcolor` (not the literal solver's hardcoded `0`/`8`,
  which only match by fixture coincidence), diagonal-filled via a plain
  Python `frozenset` (no `dsl.shoot` needed here - the diagonal is a
  simple `range`). Unlocks `d0f5fe59`. 4/4 real, 30/30 `re-arc`.
- **`canvas_row_by_foreground_count()`** (zero-arg) - a 1-row canvas, sized
  and colored by the grid's one non-zero foreground color's own cell
  count (`next(v for row... if v != 0)`, the same auto-detect ADR-0025's
  `fill_nonsingleton_foreground` established - not `dsl.other`, which
  isn't needed). Background `0` is a genuine `re-arc` generator invariant
  here (confirmed against generator source), unlike most of this pass's
  other hardcodes. Unlocks `d631b094`. 5/5 real, 30/30 `re-arc`.
- **`bar_chart_canvas_by_size4_count()`** (zero-arg) - a fixed-total-5
  "bar chart" canvas (N cells of color `1` beside `5-N` cells of the
  grid's own `mostcolor`, N = count of size-4 objects of color `1`).
  Corrects the literal solver's hardcoded background `0` to `mostcolor`;
  the fixed total `5` and fill color `1` are genuine generator invariants
  (`randint(0, 5)`), not fixture coincidences. Needs no new primitive.
  Unlocks `1fad071e`. 4/4 real, 30/30 `re-arc`.
- **`upscale_by_numcolors_minus_one()`** (zero-arg) - `dsl.upscale(grid,
  dsl.numcolors(grid) - 1)`. One shared action for both `ac0a08a4` and
  `b91ae062`: `ac0a08a4`'s literal solver (`upscale(I, 9 -
  colorcount(I,0))`) only matches its own real fixture by coincidence (all
  its real grids are 3x3 with a literal-0 background) - 1/28 on fresh
  `re-arc`; the actual shared invariant behind *both* tasks turns out to be
  identical: `numcolors(grid) - 1`. `numcolors` is the only new primitive
  (not `colorcount`/`subtract`/`decrement`, all either wrong or trivial).
  Unlocks `ac0a08a4` (28/28 corrected) and `b91ae062` (30/30, already had
  this formula). 4/4 + 6/6 real.

### Cluster B - 6 actions, 6 tasks

- **`fill_outbox_of_band_intersection()`** (zero-arg) - finds the two
  perpendicular color-bands by column/row bounding-box span (not
  connectivity-object identity, which fragments once one band cuts across
  the other), takes their intersection rectangle, and fills its `outbox`
  (new primitive - the ring of cells one step outside a patch's bbox) with
  `4`. Corrects the literal solver's `neighbors`-of-one-hole-cell read
  entirely (1/60 on `re-arc` - the single-hole-cell framing doesn't
  survive band/band intersections generally). Unlocks `67a423a3`. 4/4
  real, 300/300 `re-arc`.
- **`shoot_diagonals_from_objects()`** (zero-arg) - for every object of
  color 1 and every object of color 2 (not one merged blob per color, the
  literal solver's read), shoots a diagonal (`dsl.shoot`, new primitive)
  from that object's own `ulcorner` (already curated - `lrcorner` turns
  out unneeded once processed per-object). Unlocks `5c0a986e`. 4/4 real,
  300/300 `re-arc`.
- **`stamp_shape_at_singleton_echoes()`** (zero-arg) - auto-detects the
  marker color as any singleton object's color (not the literal solver's
  hardcoded `5`), normalizes the "anchor" shape's indices to the origin
  (`dsl.normalize`, new), and stamps it (shifted by `dsl.center`, new, of
  each singleton echo, offset by one cell) at **every** singleton echo of
  the marker color (not just `first()`, the literal solver's read).
  Unlocks `88a10436`. 4/4 real, 300/300 `re-arc`.
- **`crop_to_leastcommon_quadrant()`** (zero-arg) - splits the grid into 4
  quadrants (via already-curated `lefthalf`/`righthalf`/`tophalf`/
  `bottomhalf`), returns whichever quadrant's own full content is `dsl.
  leastcommon` (new primitive - statistical mode by value, over the 4
  quadrant-grids) among the 4. `astuple`/`combine` (the audit's flagged
  pair) are both trivial/inlinable, not needed as curated primitives.
  Verified `re-arc`'s generator structurally guarantees exactly one
  quadrant differs (no tie case reachable) - not merely assumed. Unlocks
  `88a62173`. 4/4 real, 300/300 `re-arc`.
- **`fill_backdrop_and_box_by_rarity()`** (zero-arg) - auto-detects the
  "dot" marker color and its position (not the literal solver's hardcoded
  `8`), auto-detects background (not hardcoded `0`), determines which of
  the merged-foreground bbox's 4 edges is the "cut" edge (any side post
  rotation, not just bottom - the literal solver's implicit assumption)
  to sample a safe corner, then fills the bbox interior (`dsl.backdrop`,
  new, shared with `mirror_crop_by_rectangle_marker` above) with the
  next-least-common color and the bbox outline (`dsl.box`, new) with the
  third-least-common color. Unlocks `b548a754`. 4/4 real, 290/300 (96.7%)
  `re-arc` - the 3% shortfall is a genuine, narrow degenerate case (not a
  design defect): when the interior region shrinks to its geometric
  minimum (2 cells) *and* the resulting rare-color count ties with the
  dot's own count, `leastcolor` can't unambiguously separate "the
  cutoff-marker dot" from "a legitimately rare interior color" - the same
  class of acceptable, documented shortfall as `f76d97a5`'s 99%
  (ADR-0025).
- **`mirror_border_decoration()`** (zero-arg) - classifies the two marker
  dots by *position* (row `< h/2` vs `>= h/2`, not fixed color identity,
  the literal solver's implicit assumption), draws per-half border
  segments plus through-dot frontiers using `dsl.hfrontier` (new) and
  `dsl.connect` (new - the literal solver's whole-grid `dsl.box` is the
  wrong primitive here; a straight-line `connect` through each dot is
  what the actual generator concept needs). Unlocks `1bfc4729`. 3/3 real,
  300/300 `re-arc`.

## Rejected

| Task | Primitive pair (audit) | Why not |
|------|--------------------------|---------|
| `77fdfe62` | `halve`+`width` | Root-caused by running the *literal official solver* itself against fresh `re-arc` instances: 0/20, mostly crashes (`IndexError`) - its hardcoded marker color `8` is actually a randomly-chosen color per instance. The real generator concept is a 2x2 quadrant grid inside a detected border frame with 4 corner-color markers, each quadrant's noise recolored to its own corner's color - genuine structural detection (locate the frame, the 4 quadrants, read 4 corner colors), not a parameterization fix. Same disposition as `7c008303`/`c9f8e694`/`017c7c7b`/ADR-0025's `3de23699`. |

`c9f8e694` and `017c7c7b` (both already-dispositioned no-gos from ADR-0023)
were not re-examined - no new information changes their disposition.

| Alternative considered | Why not |
|--------------------------|---------|
| Ship `b548a754` at 96.7% without the degenerate-case writeup | Every prior ADR in this lineage documents *why* a sub-100% match rate is acceptable when it is (a genuine, narrow, well-understood degenerate input), rather than treating "close enough" as self-justifying - `f76d97a5` (ADR-0025, 99%) is the precedent this follows, not an exception to it. |
| One combined "auto-detect marker/background color" helper shared across `ff805c23`/`d631b094`/`1fad071e`/`88a10436`/`b548a754`/`1bfc4729` (all correct a hardcoded-color assumption to an auto-detected one) | Each task's detection *rule* is genuinely different (solid-rectangle shape, "the one nonzero value", singleton-object membership, positional half, rarity ranking) - there's no single shared predicate to factor out, only a shared *lesson* (don't trust the literal solver's hardcoded color), which the module docstring/ADR text captures without forcing a shared code path that would have no actual common logic. |

## Consequences

- Unlocks `44f52bb0`, `7b7f7511`, `ff805c23`, `d0f5fe59`, `d631b094`,
  `1fad071e`, `ac0a08a4`, `b91ae062`, `67a423a3`, `5c0a986e`, `88a10436`,
  `88a62173`, `b548a754`, `1bfc4729` - 14 tasks, 13 new actions (all
  zero-arg, `"transform"`-kind). `MAX_ARITY` unaffected. No mechanism,
  selection, episode-log, or visualizer changes.
- 10 new `arc-dsl` primitives get their first curated use: `numcolors`,
  `backdrop`, `outbox`, `shoot`, `center`, `normalize`, `leastcommon`,
  `box`, `hfrontier`, `connect`.
- Action count: 72 -> 85. Curated tasks: 77 -> 91.
- `77fdfe62` checked and left uncurated - reasoned no-go, documented above.
- This pass's verification cost was materially higher than ADR-0025's:
  most candidates needed the actual `re-arc` generator source traced, not
  just a closer solver read, because the literal solver's *choice of
  primitive*, not just its constants, was frequently wrong for the
  broader task concept. Worth flagging for whoever scopes the next pass
  (`three_plus_new_primitives`, 26 tasks, deliberately deferred - see F11):
  expect this cost to keep rising as the remaining uncurated tasks get
  further from what a literal solver read suggests.
