# F13 Stage 1: playing 8 excluded tasks through `/api/play/*` to observe mechanism gaps

`docs/QUESTIONS.md` F13's Stage 1 plan: run Stage 0 (`viz/backend/play.py`,
ADR-0017) against a sample of tasks outside `arc_env.task_loader.
CURATED_TASK_IDS`, driving `/api/play/*` directly rather than clicking
through a browser, with an agent reasoning through each task's grids the
same way ADR-0018's LLM-seeded-search experiment had an agent act as the
"LLM proposer" - here the agent plays the "human", attempting a handful of
excluded tasks and recording where it gets stuck and what it wishes it
could do. This is direct empirical input for Stage 2, complementing
`scripts/audit_action_coverage.py`'s static solver-text audit (F11) with
actual attempted-play data.

## Method

Drove `viz/backend/play.py`'s functions directly (in-process, the same
pattern `tests/test_viz_play.py::test_start_step_save_solves_67a3c6ac`
uses, not over HTTP - no server needed for this). For each task: read its
known-correct `third_party/arc-dsl/solvers.py` solver to understand the
target transformation, formed a hypothesis for a curated-action sequence
(or an explicit "this can't work because X" prediction when no sequence
seemed possible), then actually called `play.start_session` /
`play.step_session` for **every train pair** and checked `exact_match`.
Where a sequence partially matched or failed, dug further (extra bare-`dsl`
probes) to pin down precisely which piece is missing, rather than stopping
at "no-match". Script: `f13_stage1_play.py` (run via `uv run python`, not
committed - a one-off diagnostic, not a test).

8 tasks sampled across `scripts/audit_action_coverage.py`'s buckets (current
run: 40 curated, 360 remaining - 268 excluded/higher-order, 10
free_by_name, 15 one_new, 19 two_new, 48 three_plus_new): 2 from
`free_by_name`/`one_new` picked to specifically re-check the audit's
name-level caveat, 2 from `one_new`, 2 from `two_new`, 1 `free_by_name`, and
1 from `excluded` for contrast. Not a claim that 8 tasks characterizes all
360 - a first pass, per F13's own "5-10 tasks, not all 269" scope.

## Findings

### 1. `c909285e` (audit bucket: `one_new` / `ofcolor`) - actually fully reachable today, audit false negative

Solver: `ofcolor(I, leastcolor(I))` → `subgrid(..., I)`. Hypothesis: a human
doesn't need an `ofcolor` *primitive* at all - they can just look at the
grid, read off which color is rarest, and call the already-curated
`select_by_color(that literal color)` + `crop_to_selection`.
`select_by_color`'s implementation (`arc_env/actions.py`) is
`dsl.colorfilter(_objects(grid), color)` merged into one selection, which
for a univalued object partition is exactly the same index set as
`dsl.ofcolor(grid, color)` - the two are equivalent whenever every cell of
a given color belongs to some univalued connected object (always true).
**Verified: all 3 train pairs match exactly**, zero new primitives needed.

This is the same "audit's name-level bucketing missed an existing
equivalence" surprise ADR-0015 already logged for `a79310a0` - a second,
independent confirmation that the `one_new`/`free_by_name` buckets need a
human/agent pass before being trusted, not just a re-statement of that
caveat. **Action item for a future incremental pass**: land `c909285e` as a
curated task (no new action needed, `select_by_color` + `crop_to_selection`
already covers it) - out of scope for this Stage-1 observation pass itself,
but a free win worth flagging.

### 2. `42a50994` (audit bucket: `one_new` / `sizefilter`) - needs a "select every object matching a criterion, merged" selector, not just one more object

Solver: `objects(T,T,T)` → `sizefilter(=1)` → `merge` → `cover` (delete).
Probed with the closest existing analogue, `select_smallest` (picks the
single smallest object) + `delete_selected`: **fails on all 4 train
pairs** - `select_smallest` returns exactly one object, but this task's
grids have multiple size-1 "noise" objects that all need deleting at once.
The gap isn't a missing DSL primitive by name (`sizefilter` is a filter, not
a new mechanism) - it's that every curated selector picks **exactly one**
object via some argmax/argmin/unique/color criterion; none picks **an
arbitrary-count subset matching a predicate, merged into one selection**.

### 3. `b94a9452` (audit bucket: `one_new` / `first`) - selection is fine; the missing piece is a "switch"-style derived action

Solver: `objects(F,F,T)` → `first` → `subgrid` → `switch(leastcolor,
mostcolor)`. Checked object counts directly: this task's `(univalued=False,
diagonal=False, without_bg=True)` partition has **exactly 1 object in all 3
train pairs** - so `first`'s "arbitrary element of the set" is trivially the
same object any argmax/argmin selector would also pick. Probed
`select_largest` + `crop_to_selection`: this correctly reaches the
`x3 = subgrid(...)` intermediate for all 3 pairs (confirmed by comparing the
cropped grid to the solver's own `x3` bare-call value, not just guessing
from the no-match). The full sequence still doesn't match only because the
**last** step, `switch(leastcolor(x3), mostcolor(x3))` (swap a grid's least-
and most-common color), has no curated equivalent for the *current* grid.

This is a narrow, concrete near-term candidate: ADR-0019 already landed
`swap_two_least_colors` (swaps a grid's two *least*-common colors) as a
derived, zero-arg action. A sibling `switch_least_most_colors` (swap
least↔most, same "derived from the current grid, no agent-chosen args"
pattern) would land `b94a9452` outright - no new mechanism, one more
zero-arg transform action in the same family as `canvas_mostcolor`/
`swap_two_least_colors`. Not implemented this pass (Stage 1 is observation,
not Stage 2's design/build), but concrete enough to hand to a future
incremental ADR pass directly.

### 4. `a740d043` (audit bucket: `free_by_name`) - needs a "select ALL objects, merged" selector; existing single-object selectors only coincide by luck

Solver: `objects(T,T,T)` → `merge` (every object, not filtered) → `subgrid`
→ `replace(1,0)`. Checked object counts: 2 objects in all 3 train pairs.
Probed `select_largest` + `crop_to_selection` + `replace(1,0)`: **matches
pair 0 only**, fails pairs 1-2. Diagnosis: `select_largest`'s bounding box
only equals the *union* of both objects' bounding box when the smaller
object happens to sit entirely inside the larger one's bbox (true for pair
0's specific geometry, false for pairs 1-2) - exactly the same "a
selection's result coincides with the general case only for one fixture's
specific geometry" trap ADR-0015's `7c008303` analysis and ADR-0019's
`5582e5ca` shape-guess correction both already hit. The real requirement is
a selector that merges **every** object in the partition into one selection
(not filtered by any size/color/uniqueness criterion at all) - a cheap
sibling to #2's "select multiple by predicate, merged": call it
"select-all", the null/always-true predicate case of the same missing
selector family.

### 5. `007bbfb7` (audit bucket: `one_new` / `cellwise`) - the "one new primitive" label understates the gap; needs two live grids at once, not one more op on the current grid

Solver: `hupscale(3)` then `vupscale(3)` on a copy, `hconcat`×2 +
`vconcat`×2 on a *separate* copy (3×3-tiling the original), then
`cellwise(upscaled, tiled, 0)` combines the two **different, independently-
built** grids cell-by-cell. Probed `hupscale(3)` → `vupscale(3)`: produces
the upscaled grid correctly (an intermediate value, not checked against
target - the task's actual output needs the `cellwise` combination), but
there's no way, at any point in the sequence, to also hold the tiled-concat
grid and combine it with the upscaled one - the current action model
threads exactly one "current grid" through every step (`arc_env/actions.py`
module docstring: "Grid [, scalar args] -> Grid"). `cellwise` isn't
reachable by adding one new action shaped like every other transform in
this menu; it needs the same kind of "hold more than one grid state at
once" capability the multi-selection/cross-grid mechanism (F11's
`7c008303`/`a68b268e`) already named as out of scope for an incremental
ADR - this is independent evidence the pattern recurs beyond those 2 tasks,
in a task that wasn't even in F11's `ofcolor`-flagged cluster.

### 6. `928ad970` (audit bucket: `two_new` / `inbox`, `ofcolor`) - needs a selection computed in a cropped coordinate space, applied back to the ORIGINAL grid

Solver: `ofcolor(I,5)` → `subgrid` → `trim` → `leastcolor` (of the trimmed
crop) → `inbox(x1)` (interior region of the *original*, pre-crop indices) →
`fill(I, that color, that interior)` - the fill target is `I`, the
**original, unmodified grid**, not the cropped/trimmed intermediate. Probed
`select_by_color(5)` + `crop_to_selection`: this reaches the crop, but
`crop_to_selection` **replaces the current grid with the crop** - by the
time a `leastcolor`-reading step could run, the original `I` is gone, and
there is no `inbox` action at all regardless. This is the same
"selection/derived-value computed in one coordinate space, needed against a
*different* (here: the original, untransformed) grid" shape ADR-0015 named
in its `9ecd008a` no-go analysis (F11's Register) - a **third**, independent
confirmation of that specific tension, in a task neither F11's `ofcolor`
cluster nor `9ecd008a`'s own analysis covered. Directly relevant to Thread
1's "does the cross-grid-reuse pattern recur beyond the 2-3 already-known
tasks" question.

### 7. `017c7c7b` (audit bucket: `two_new` / `branch`, `equality`) - the conditional itself is a non-issue; the missing piece is appending an arbitrary sub-crop, not a mirrored self-copy

Solver: compares `tophalf(I)` vs `bottomhalf(I)`, `vconcat`s `I` with either
`bottomhalf(I)` (if they're equal) or a fixed `crop(I, (2,0), (3,3))`
otherwise, then `replace(1,2)`. Checked directly: the two halves are equal
in exactly 1 of 3 train pairs, confirming this really is a per-instance
branch a human would decide by eye in about one second - **the `branch`/
`equality` half of the audit's "two new primitives" label is not a real
obstacle for a human player at all**. The actual gap is narrower: every
curated self-concat action (`vconcat_self_hmirror_top`/etc., ADR-0010)
appends a *whole, mirrored copy of the current grid* - there is no action
that appends an arbitrary **sub-crop** (here, `bottomhalf` or a fixed
interior 3×3 crop) onto the grid it was cropped from, because
`crop_to_selection`-style cropping *replaces* the current grid rather than
extending it. Probed isolating `bottomhalf`: confirmed the original 6-row
grid is gone by the time it would need to be `vconcat`'d back onto - same
"crop replaces rather than composes" limitation as #6, in a case that has
nothing to do with selections (no `select_*` action is involved in this
solver at all) - a fourth data point that the underlying gap is broader
than "selection state," more like "no grid, once transformed, can be
recombined with an earlier or sibling grid state."

### 8. `3aa6fb7a` (audit bucket: `excluded` / `mapply`) - a qualitatively different gap: per-object independent iteration, not more grid/selection state

Solver: `objects(T,F,T)` → `mapply(corners, objects)` (apply `corners`
**independently to every object**, merge the results) → `underfill(1,
that)`. Probed `select_largest`: reaches exactly one object, immediately
stuck - there is no `corners`-of-a-selection action, and more fundamentally
no way to apply *any* per-object operation to an unknown, variable number
of objects independently and merge the results, short of literally adding
one step per object (which the action space can't parameterize on, since
the object count varies per task instance). This is the `mapply`/`fork`
higher-order-combinator exclusion ADR-0001 already ruled structurally out
of reach for a flat, one-primitive-per-step action space - included here
for contrast, to confirm Stage 1's method correctly reproduces a known,
already-understood boundary rather than surfacing something new. Distinct
in kind from #5/#6/#7's "need two grids" and #2/#4's "need to select more
than one object, merged": this is "need to iterate a same operation over a
variable number of objects independently", closer to F12's LLM-in-the-loop
scope than to F11/Thread 1's selection-mechanism scope.

## A rough taxonomy of the mechanism gaps this pass actually found

Reading the 6 non-trivial findings (#2, #4, #5, #6, #7, #8) together, they
split into (at least) three qualitatively different families, not one:

1. **"Select more than one object, merged" (#2, #4)** - a cheap, bounded
   extension of the *existing* single-selection model: add one or two more
   selection criteria (an always-true "select-all" predicate for #4; a
   size-equals-N-merged predicate for #2) to the same `select_*` action
   family already curated. No new state channel, no PPO/GP/logging
   redesign - same shape as every `select_*` addition ADR-0011 through
   ADR-0016 already made.
2. **"Hold/recombine more than one grid state at once" (#5, #6, #7)** - a
   genuinely bigger gap: `cellwise` combining two independently-built
   grids, a selection computed pre-crop needed post-crop, and appending an
   arbitrary sub-crop rather than a whole mirrored self-copy are three
   different surface symptoms of the same underlying limitation (the
   action model threads exactly one "current grid" through every step).
   This overlaps with, but is broader than, F11/Thread 1's
   "multi-selection/cross-grid" framing - #7 in particular needs no
   selection at all, just two grid *states* (before- and after-crop) held
   simultaneously. Worth folding into Thread 1's design-options
   conversation as evidence the mechanism gap is "more than one live grid
   state," not narrowly "more than one live *selection*."
3. **"Independent per-object iteration over a variable-count set" (#8)** -
   the already-understood, already-out-of-scope higher-order-combinator
   exclusion (ADR-0001/F12), not new information, included for contrast.

Family 1 is a normal incremental-ADR candidate (like #3's
`switch_least_most_colors`, also cheap and narrow). Family 2 is the one
Stage 2's design work should scope around - it is bigger than "add a
selection slot" and should be framed, per Thread 1's own conversation, as
"how many/which grid states can be held live at once and how are they
addressed/recombined," not narrowly as "how many named selections." Family
3 stays out of scope, same as F12 already concluded.

## What this pass did *not* do

- Did not attempt all 360 uncurated tasks - 8 is a first pass per F13's own
  "5-10, not all 269" scope; a larger sample could turn up gap shapes this
  pass didn't see, or could firm up how common Family 2 (vs. Family 1) is
  across the remaining pool.
- Did not save any of these sessions as `runs/` demonstrations - none of
  the 8 sequences fully solves its task (except `c909285e`, a genuine
  curated-task candidate, not a gap-finding one), so there's nothing to
  warm-start from yet; saving is Stage 0's job for actually-completed
  solves, not this diagnostic pass's.
- Did not build any new mechanism - Stage 1 is observation only, per F13's
  own staging; `switch_least_most_colors` (#3) and the "select-all"/
  "select-by-size-merged" family (#2, #4) are concrete enough to become
  their own small incremental ADR later, and Family 2's broader design is
  explicitly Thread 1/Stage 2's job, not this doc's.
