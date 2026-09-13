# F13: the interactive "Photoshop-like" editor

**Status:** decided (user, 2026-09-06), staged. Stage 0 built 2026-09-06
(ADR-0017). Stage 1 done 2026-09-13. Stage 2 in progress (Buckets A and B
landed 2026-09-13, ADR-0021/ADR-0022; the "hold the pre-crop original
grid" cluster stays open).

Should we build an interactive editor where a human solves ARC-AGI-1 tasks
by hand, with a constrained toolset mirroring the curated action space,
logged as a demonstration trajectory for warm-start/imitation learning?

Assessed as viable, staged to de-risk before committing to the full build.

## Stage 0: the write path (landed 2026-09-06, ADR-0017)

A minimal UI over the existing curated actions (41 as of that pass) plus a
live `ArcEnv` instance, logging through the unmodified `EpisodeWriter`
schema (`arc_env/episode_log.py`) so human solves are automatically
warm-start-compatible (ADR-0009). No schema change.

This needed a real write path into `viz/backend/server.py`, which was
read-only by design (ADR-0006/ADR-0007). A deliberate, explicit
architectural reversal, not an incidental one, scoped as its own small
addition (`viz/backend/play.py`, plus `/api/actions` and `/api/play/*`
routes driving a per-session `ArcEnv`) rather than a rewrite of the
read-only run-browsing path the rest of the visualizer depends on.

What landed: `viz/backend/play.py` (an in-memory
`session_id -> PlaySession` store, guarded by a `threading.Lock()`, no
persistence until an explicit save, an accepted local-single-user
tradeoff), path-traversal validation on both `task_id` and the
client-supplied `run_id` before any file access, and any of the 400
training tasks playable, not restricted to the curated subset (Stage 1
needed that). A new `viz/frontend/src/play.ts` panel wired into the
existing single-page app, with its action picker built generically from
`GET /api/actions` rather than a hardcoded menu.

## Stage 1: playing the excluded tasks (done 2026-09-13)

Run Stage 0 against a sample of tasks outside `CURATED_TASK_IDS`, to see
what a human naturally reaches for and feed that back as design input for
new mechanisms, complementing F11's static solver-text audit with actual
usage data.

Full writeup: `docs/research/f13-stage1-play-audit.md`. Summary: drove
`play.py`'s functions directly (in-process, the same pattern
`tests/test_viz_play.py` uses) against 8 uncurated tasks sampled across
the audit's difficulty buckets. For each, we read the known-correct
solver, hypothesized a curated-action sequence, and verified it against
every train pair rather than eyeballing.

`c909285e` turned out fully reachable today with zero new primitives:
`select_by_color` on the least common color (read by eye), plus
`crop_to_selection`. The audit's name-level `ofcolor` flag was a false
negative, the same class of surprise ADR-0015 logged for `a79310a0`. A
free curation candidate for a future incremental pass.

The other 7 sorted into three distinct gap families:

1. **Select more than one object, merged** (`42a50994`, `a740d043`). A
   cheap, bounded extension of the existing single-selection model: an
   always-true "select-all" predicate, a size-filter-merged predicate.
   Same shape as every prior `select_*` addition. **Landed 2026-09-13,
   ADR-0021.**
2. **Hold or recombine more than one grid state at once** (originally also
   included `007bbfb7`'s `cellwise`; `928ad970`'s selection computed
   pre-crop but needed post-crop against the original grid, a third
   independent confirmation of the exact tension ADR-0015 named in its
   `9ecd008a` no-go, in a task neither F11's `ofcolor` cluster nor that
   analysis covered; `017c7c7b`'s need to append an arbitrary sub-crop
   rather than a whole self-copy, needing no selection at all). Broader
   than F11's "multi-selection" framing suggests: the real requirement is
   "more than one live grid state," a bigger idea than "more than one
   named selection." **Correction (2026-09-13, ADR-0021): `007bbfb7` was
   miscategorized here.** It's a pure function of the current grid alone
   (no selection, no cross-grid state) - fully reachable as one bundled
   derived action, same low-risk pattern as the self-concat actions. Its
   two siblings `80af3007`/`8f2ea7aa` share the same "fractal expansion via
   self-cellwise-combine" shape and landed alongside it. This family's real
   roster is `928ad970`/`017c7c7b` only - 2 tasks, still open (see F11's
   register row).
3. **Independent per-object iteration over a variable-count set**
   (`3aa6fb7a`). The already-understood, already-out-of-scope
   `mapply`/higher-order exclusion from ADR-0001/F12, included for
   contrast, confirming Stage 1's method reproduces a known boundary
   rather than only surfacing new ones.

We also found one narrow, concrete near-term ADR candidate, independent of
the above: `b94a9452` is fully reachable except for its last step (a
`switch(leastcolor, mostcolor)` swap of the current grid, verified by
direct comparison that its own selector + `crop_to_selection` exactly
reproduces the solver's own intermediate value). A
`switch_least_most_colors` derived action, sibling to ADR-0019's
`swap_two_least_colors`, would land it outright. **Landed 2026-09-13,
ADR-0021** (alongside a fifth `objects(...)` connectivity variant this
task turned out to need - its own selector wasn't `select_largest` after
all, see ADR-0021).

**Direct input for the multi-selection/cross-grid mechanism decision:**
gap family 2 is independent empirical evidence that the cross-grid,
stale-value-reuse pattern recurs beyond the 2-3 tasks
[F11](f11-task-coverage.md)'s static audit already named, and reframes the
mechanism's scope as "how many live grid states, addressed and recombined
how," rather than narrowly "how many named selections." To be weighed in
that decision's own design-options conversation.

## Stage 2: in progress

Turn Stage 1's findings into real ADR'd mechanisms, held to the same
review bar as ADR-0011/0012/0013/0015/0016.

**Bucket A landed 2026-09-13 (ADR-0021):** Family 1 (`42a50994`,
`a740d043`), `switch_least_most_colors`/`b94a9452`, and (found during
ADR-0020's own audit, previously mis-filed as Family 2) the "fractal
expansion" cluster `007bbfb7`/`80af3007`/`8f2ea7aa`, plus `c909285e` (a
free win the original coverage audit's name-level check missed). 7 tasks,
6 new actions, all verified end to end before implementation - see
ADR-0021 for the corrected accounting.

**Bucket B landed 2026-09-13 (ADR-0022):** `cf98881b` (a 3-way region
split plus a new `fill_slot_onto_region` action - turned out to need a
genuinely new `Action.kind`, not just more region entries, once the exact
need was worked out) and `1b2d62fb` (a `replace_region_and_fill` action
that fuses `replace` and `fill` into one atomic step, reusing the
existing `act_on_region_selection` kind - cheaper than first framed, no
new kind needed once the selection-clearing question was sidestepped by
fusing rather than adding a bare region-scoped `replace`).

**Still open:** `7c008303`/`a68b268e`/`928ad970`/`017c7c7b` (the "hold the
pre-crop original grid" need, Family 2's real roster) as its own
still-undecided design question, the same weight as the original
multi-selection/cross-grid mechanism decision.

**Landed by:** ADR-0017, `docs/research/f13-stage1-play-audit.md`,
ADR-0021, ADR-0022.
