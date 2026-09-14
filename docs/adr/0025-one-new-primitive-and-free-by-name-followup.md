# ADR-0025: 5 new `arc-dsl` primitives (`one_new_primitive` bucket) + 3 more `free_by_name` derived actions

- Status: Accepted
- Date: 2026-09-14
- Deciders: repo owner, via conversation 2026-09-14

## Context

Re-ran `scripts/audit_action_coverage.py` fresh against the post-ADR-0024
baseline (67 curated, 333 remaining) and worked its two next-highest-value
buckets:

- `free_by_name` (13 tasks): 4 already dispositioned by prior ADRs
  (`0520fde7`, `a699fb00` - no-go, ADR-0024; `7c008303`, `9ecd008a` - no-go,
  ADR-0023/F11), leaving 9 genuinely unexamined:
  `11852cab`, `3618c87e`, `47c1f68c`, `67385a82`, `aedd82e4`, `cce03e0d`,
  `e3497940`, `e8593010`, `e98196ab`.
- `one_new_primitive` (9 tasks, each needing exactly one `arc-dsl` primitive
  not yet used anywhere in `arc_env/actions.py`): `32597951` (`delta`),
  `c1d99e64` (`frontiers`), `f76d97a5` (`palette`), `e9afcf9a` (`astuple`),
  `05f2a901` (`gravitate`), `90c28cc7` (`dedupe`), `6d75e8bb` (`asobject`),
  `3de23699` (`fgpartition`), `c3e719e8` (`asindices`).

Each of the 18 was hand-read, verified bare-`dsl` against every train/test
pair, then verified through the real `arc_env.actions.execute()` ->
`ArcEnv.step()` path (not just a bare-function replay - the ADR-0013
`select_tallest`/`1c786137` distinction), then generalization-checked
against 30+ fresh `re-arc` instances, brute-forcing plausible argument
assignments rather than trusting the literal solver's hardcoded constants.
Several tasks only reproduce their own real fixture because the literal
`solvers.py` program's hardcoded scalar (a color, a crop shape, a size
threshold) happens to match that one grid's specific values - not a general
invariant `re-arc`'s broader instance space shares. This pass found that
cuts both ways: some literal solvers are narrower than the task's real
concept (rejected below), and some just needed the hardcoded constant
replaced with a grid-derived value or an agent-chosen arg to reach full
generalization (landed below), the same correction ADR-0024's
`fill_nonsingleton_foreground`-style fixes and ADR-0021's
`canvas_mostcolor`/`swap_two_least_colors` precedent already established.

Two `one_new_primitive` candidates landed at first pass but below this
project's established ~100% re-arc bar (`05f2a901` at ~96%, `3de23699` at
~94%) got a dedicated root-cause follow-up rather than being shipped on a
"close enough" basis - see their own Rejected entries below for the exact
mechanism found.

## Decision

**10 new zero/low-arity `"transform"`-kind actions, no new `Action.kind`, no
new mechanism** - the same "derived, not drawn from a solver 1:1" family
`canvas_mostcolor`/`fractal_expand_cellwise`/ADR-0024's tiling actions
already established. **5 new `arc-dsl` primitives** get their first use in
`arc_env/actions.py`: `delta`, `frontiers`, `palette`, `dedupe`, `asobject`
(not `gravitate`, `fgpartition`, `astuple`, or `asindices` - see Rejected).

All verified bare-`dsl`, through the real `execute()` -> `ArcEnv.step()`
path, and against fresh `re-arc` instances per-task as noted:

### `one_new_primitive` bucket - 7 of 9 land

- `fill_delta_by_color(dot_color, fill_color)` (2 `COLOR_ARG`s) -
  `dsl.fill(grid, fill_color, dsl.delta(dsl.ofcolor(grid, dot_color)))`.
  `delta` is new (indices in a patch's bounding box but not part of the
  patch itself). Unlocks `32597951`. 4/4 real, 30/30 `re-arc`.
- `fill_frontiers()` (zero-arg) - `dsl.fill(grid, 2,
  dsl.merge(dsl.frontiers(grid)))`. `frontiers` is new (full-width/height
  single-color rows/columns). Fill color `2` is a `re-arc` generator
  invariant, not an arbitrary hardcode. Unlocks `c1d99e64`. 4/4 real, 30/30
  `re-arc`.
- `switch_palette_then_zero_five()` (zero-arg) - literal-solver copy:
  `palette(grid)` -> `first`/`last` -> `switch` the two colors -> `replace`
  the surviving `5` with `0`. `palette` is new. Colors `5`/`0` are reserved
  generator invariants, same as `fill_frontiers`'s `2`. Unlocks `f76d97a5`.
  4/4 real, 297/300 (99%) `re-arc` - the 3 misses are a genuinely
  ill-posed degenerate input (the generator occasionally fills the entire
  canvas with the foreground color, leaving no background `5` for the task
  concept to act on at all), the same class of guarded degenerate case
  `_middle_third`'s existing `StopIteration` fix already treats as
  acceptable rather than a design defect.
- `tile_alternating_column_mirror()` (zero-arg) - **needs zero new
  primitives**, superseding its bucket assignment. The literal solver
  hardcodes an `astuple(TWO, ONE)` crop shape and a fixed 6-column-wide
  tiling; that reproduces the real fixture (3/3) but is 0/30 on `re-arc`
  (the true repeating-column unit varies with grid width, not a fixed
  `(2,1)` crop). Re-deriving the tile unit and count from the grid's own
  shape instead of the literal hardcode reaches 30/30, using only
  already-curated `hmirror`/`hconcat`/`crop`. Unlocks `e9afcf9a`.
- `dedupe_grid_both_axes()` (zero-arg) - crop to the bounding box of the
  grid's literal non-zero cells (not `dsl.objects`'s `mostcolor`-based
  background autodetection, which picks the wrong background here the same
  way ADR-0023/24's audits already flagged elsewhere), then
  `dedupe`->`rot90`->`dedupe`->`rot270` to collapse repeated adjacent rows
  and columns. `dedupe` is new. Unlocks `90c28cc7`. 4/4 real, 100/100
  `re-arc`.
- `fill_holes_in_object_bbox()` (zero-arg) - crop to the grid's single
  foreground object's bounding box (`select_largest`-style), replace
  `mostcolor(grid)` (not a literal `0` - the literal solver's hardcode,
  0/30 on `re-arc`) with `2` inside the crop, then `asobject`+`shift`+
  `paint` the recolored crop back onto the original grid at its original
  position (a non-destructive composite, distinct from an in-place
  `fill`/`replace`). `asobject` is new. Unlocks `6d75e8bb`. 4/4 real, 30/30
  `re-arc`.
- `tile_by_mostcolor()` (zero-arg) - tiles the grid `h x w` times (`h`,`w`
  the grid's own dimensions, not the literal solver's hardcoded 3x3 - a
  fixture-specific coincidence where the real grid's own size happened to
  equal 3), pasting a copy of the grid at every position matching
  `mostcolor(grid)` via the same `asobject`+`shift`+`paint` composition as
  `fill_holes_in_object_bbox` (so this needs `asobject`, not the literal
  solver's `asindices`+`difference` route). Unlocks `c3e719e8`. 4/4 real,
  30/30 `re-arc`.

### `free_by_name` bucket - 3 of 9 unexamined land

- `paint_vmirrored_righthalf_onto_lefthalf()` (zero-arg) -
  `dsl.paint(dsl.lefthalf(grid), dsl.merge(dsl.objects(dsl.vmirror(
  dsl.righthalf(grid)), T, F, T)))`, using only already-curated primitives.
  Unlocks `e3497940`. 4/4 real, 60/60 `re-arc`.
- `fill_nonsingleton_foreground()` (zero-arg) - auto-detects the grid's
  single foreground color in plain Python (the grid's one non-background
  value, not a hardcoded `THREE` - the literal solver's hardcode is 0/30
  on `re-arc`, whose generator picks an arbitrary foreground color), fills
  every non-singleton (size > 1) object of that color to `8`. Unlocks
  `67385a82`. 5/5 real, 60/60 `re-arc`.
- `recolor_objects_by_size()` (zero-arg) - fills size-1/2/3 foreground
  objects (auto-detected background, not the literal solver's hardcoded
  `0` - also 0/30-class fragile on `re-arc`, whose generator picks an
  arbitrary background color) to colors `3`/`2`/`1` respectively. Unlocks
  `e8593010`. 4/4 real, 60/60 `re-arc`.

The other 6 unexamined `free_by_name` tasks (`11852cab`, `3618c87e`,
`47c1f68c`, `aedd82e4`, `cce03e0d`, `e98196ab`) are rejected - see below.

## Rejected

| Task | Primitive/bucket | Why not |
|------|-------------------|---------|
| `11852cab` | free_by_name | Literal solver merges *all* objects before mirroring about one shared bbox - correct only when a grid has exactly one point-symmetric cluster. `re-arc` places 1-N independent clusters per grid; merging corrupts each cluster's own local symmetry. ~7/30 (23%). A per-object mirror variant is worse (0/4 even on the real fixture - real objects are scattered single cells whose own-bbox mirror is a no-op). |
| `3618c87e` | free_by_name | Literal solver moves a noise dot by a fixed `TWO_BY_ZERO` offset; `re-arc`'s generator moves the dot to the end of a variable-length line, not a fixed 2-cell hop - the real fixture's line length is a coincidence. 0/30. |
| `47c1f68c` | free_by_name | `compress` strips *every* uniform row/column, not just the intended divider; `re-arc` frequently produces spurious all-background rows/columns from sparse object placement that the real fixtures apparently never hit, over-trimming real content. 18/30 (60%). |
| `aedd82e4` | free_by_name | Genuine structural tension in `dsl.objects`'s single `(univalued, diagonal, without_bg)` knob: the literal hardcoded-color design is 2/30 (arbitrary noise color); a multivalued/background-autodetect redesign is 30/30 `re-arc` but only 3/5 real (majority-color heuristic breaks when noise outnumbers background); a univalued/explicit-color-0-exclusion redesign is 5/5 real but 9/30 `re-arc` (wrongly splits adjacent different-colored noise into separate objects). No single connectivity triple satisfies both. |
| `cce03e0d` | free_by_name | Literal solver's upscale-by-3-plus-3x3-tile only coincides when the grid is exactly 3x3 (real fixture); `re-arc`'s actual concept is "stamp a shifted, scaled copy at each marker cell," structurally different at any other size. 1/30. |
| `e98196ab` | free_by_name | `re-arc`'s generator `dmirror`-transposes ~50% of instances, turning the horizontal top/bottom divider into a vertical one; a fixed `tophalf`/`bottomhalf` split can't detect divider orientation at runtime. 13/30 (43%, consistent with roughly half getting the wrong split axis). |
| `05f2a901` | one_new_primitive (`gravitate`) | Root-caused via direct repro: `dsl.gravitate` stops at the first shift where Manhattan distance to the target reaches 1, but `re-arc`'s own ground truth shifts until true cell-overlap and backs off one step. These coincide only when the moving object is convex along its travel axis; `05f2a901`'s generator explicitly grows concave/gapped objects via random cellular growth, so the distance-1 condition frequently plateaus for 2+ consecutive shift steps before real overlap, and `gravitate` stops early. ~96% (289-290/300), inherent to `gravitate`'s own stopping rule, not fixable by any action-level argument or guard. |
| `3de23699` | one_new_primitive (`fgpartition`) | Root-caused via direct repro, including running the unmodified official `solvers.py` solver itself against fresh `re-arc` instances: it crashes (`StopIteration`) on 5% and is silently wrong on another 0.8% - i.e. the known-correct solver is *itself* not robust against this task's own generator. Root cause: `fgpartition`'s background detection (`palette(grid) - {mostcolor(grid)}`) is a single global-majority heuristic; the generator's independently-sampled canvas margin and interior noise density frequently make the *noise* color the global majority instead of the true background. Needs real structural detection (identify the target rectangle by shape/connectivity, not global color frequency) - out of scope, same disposition as `7c008303`/`c9f8e694`/`017c7c7b`. |

| Alternative considered | Why not |
|--------------------------|---------|
| Ship `05f2a901`/`3de23699` anyway at ~95-99% since real train/test pairs are 100% | This project's task-coverage numbers are meant to reflect genuine task generalization, not overfitting to the one published fixture - every prior ADR in this file's lineage held a strict ~100%-on-`re-arc` bar before curating a task (rejecting `0520fde7` at 1/30 and `a699fb00` at 5/30, for instance). A GP/PPO policy trained against these tasks draws its practice instances from the same `re-arc` generator (`arc_env/re_arc.py`), so a ~5% silent-wrong-answer rate would show up directly as training noise, not just a documentation footnote. |
| One combined "recolor by size" action for `aedd82e4`/`e8593010`'s size-based recoloring | Different connectivity/background requirements per task (see `aedd82e4`'s three-way rejection above) - `e8593010`'s own design happens to generalize cleanly with plain multivalued/no-diag connectivity and an explicit background exclusion that `aedd82e4`'s task shape doesn't tolerate; forcing one shared action would inherit `aedd82e4`'s failure mode. |

## Consequences

- Unlocks `32597951`, `c1d99e64`, `f76d97a5`, `e9afcf9a`, `90c28cc7`,
  `6d75e8bb`, `c3e719e8`, `e3497940`, `67385a82`, `e8593010` - 10 tasks, 10
  new actions (9 zero-arg, 1 two-`COLOR_ARG`). `MAX_ARITY` unaffected
  (still 4, from `commit`). No mechanism, selection, episode-log, or
  visualizer changes.
- 5 new `arc-dsl` primitives get their first curated use: `delta`,
  `frontiers`, `palette`, `dedupe`, `asobject`.
- Action count: 62 -> 72. Curated tasks: 67 -> 77.
- `11852cab`, `3618c87e`, `47c1f68c`, `aedd82e4`, `cce03e0d`, `e98196ab`,
  `05f2a901`, `3de23699` checked and left uncurated - reasoned no-gos,
  documented above so a future pass doesn't re-derive the same ground.
