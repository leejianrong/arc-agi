# ADR-0020: a region-scoped, dual-slot selection mechanism

- Status: Accepted
- Date: 2026-09-13
- Deciders: repo owner, via conversation 2026-09-13

## Context

`docs/QUESTIONS.md` F11 (`docs/questions/f11-task-coverage.md`) named a
multi-selection/cross-grid mechanism as a decision bigger than a normal
incremental ADR, deliberately deferred pending a broader audit and explicit
design sign-off. Two tasks originally motivated it: `7c008303` and
`a68b268e`, both confirmed needing more than one live selection or grid
state at once.

Per that deferral's own instruction, this pass first ran a broader
structural audit of the 92 uncurated tasks *not* excluded by a higher-order
combinator (`scripts/audit_action_coverage.py`'s `free_by_name` + `one_new`
+ `two_new` + `three_plus_new` buckets, current count as of ADR-0019),
before committing to any design.

### The audit, and a reframing it forced

A static classifier (walks each solver's straight-line assignments,
modeling what a single-mutable-grid `ArcEnv` episode can and can't do)
flagged 61/92 tasks as showing some "more than one live grid state"
structure. Reading through all 61 by hand surfaced something that reshapes
the whole cost/benefit question: **because PPO/GP train one dedicated
policy per task (ADR-0008), almost any task whose solver avoids
higher-order combinators is technically reachable by adding one more
bespoke derived action that hardcodes its exact fixed pipeline** - the same
"derived, not drawn 1:1 from a solver" pattern already used 10 times
(`fill_cell`, `canvas`, `canvas_mostcolor`, `swap_two_least_colors`, the 4
self-concat actions). Roughly 40 of the 61 flagged tasks are exactly this:
fixed formulas (quad-mirror tiling, fractal cellwise expansion, canvas-built
geometric patterns) bundleable as one more narrow action - several sharing
enough structure that one action would unlock 2-3 tasks at once (a
"fractal expansion via self-cellwise-combine" shape alone covers
`007bbfb7`/`80af3007`/`8f2ea7aa`).

So the real question was never "is this technically reachable" - it's
whether to keep adding narrower single-purpose derived actions (cheap, but
each barely tests whether GP/PPO can *discover* anything, closer to handing
over the answer) or to invest in a genuinely reusable mechanism that lets
the agent make an *instance-dependent choice* at decision time, the same
distinction ADR-0013's `select_tallest`/`1c786137` deferral and ADR-0015's
`9ecd008a` no-go already drew on ("a single-task special case dressed up
as a general mechanism is not a generally useful addition").

### What actually needs a general mechanism

One coherent, recurring shape does: split the grid into two regions
(halves), pull cell locations of a given color independently from each,
combine the two location-sets with a set operation (intersection, union,
symmetric difference), and paint the result. Confirmed in at least 9 tasks:
`6430c8c4`, `94f9d214`, `ce4f8723`, `f2829549`, `fafffa47`, `99b1bc43`,
`3428a4f5`, `dae9d2b5`, `1b2d62fb` - a materially bigger, better-evidenced
number than the original 2-task motivation, concentrated in one recognizable
pattern rather than scattered one-offs. (`cf98881b` shares the shape with a
3-way split rather than halves - out of this pass's region menu, noted
below as a follow-up, not landed.)

**This is not the same need `7c008303`/`a68b268e` have.** Those need the
*pre-crop original* grid held alongside a derived crop (not two regions of
the still-current grid), and `a68b268e` specifically needs three
simultaneous regions, not two. Verified directly: replaying `a68b268e`
against this ADR's own action set breaks the moment the current grid is
replaced by one quadrant, since the other two quadrants can no longer be
re-derived from the (by-then-gone) original. Explicit scope decision,
confirmed with the user: **build for the 9-task region-vs-region cluster
now; leave `7c008303`/`a68b268e` (and F13 Stage 1's `928ad970`/`017c7c7b`,
which need the same "hold the pre-crop original" capability) an
open, separate question.**

## Decision

Add a **second named selection slot** (`"a"`, `"b"`) and a small
**region-scoping** menu, plus a way to **combine** the two slots into one
result - four new actions, verified against all 9 target tasks by direct
replay against every train/test pair before writing this up (not just
plausibility-checked):

- `select_by_color_in_region(slot, region, color)` - `kind="select_region"`
  (new kind). `region` is one of `tophalf`/`bottomhalf`/`lefthalf`/
  `righthalf` (all four already-curated transform actions, reused here as
  the region-cropping step, not duplicated). Computes
  `dsl.ofcolor(REGION_FN[region](grid), color)` and writes the result into
  `selected[slot]`, tagging `selected_region[slot] = region` (needed by
  `fill_onto_region` below). Writing into one slot never touches the other.
- `combine_slots(op)` - `kind="combine_selection"` (new kind). `op` is one
  of `intersect`/`union`/`symdiff`. Requires both slots populated *and*
  their tagged regions' cropped shapes to match (a and b must be
  comparable positions - true whenever both are halves of the same grid);
  otherwise invalid/no-op, same convention as every other kind. Combines
  `selected["a"]` and `selected["b"]` via the chosen set operation, writes
  the result back into `selected["a"]` (keeping `selected_region["a"]`
  unchanged - the first operand's region is what a following
  `fill_onto_region` targets), and clears slot `"b"`.
- `fill_new_canvas(bg_color, fill_color, height, width)` -
  `kind="act_on_selection"` (the *existing* kind, unmodified: it already
  operates on "the" selection, which is now unambiguously
  `selected["a"]`). `dsl.fill(dsl.canvas(bg_color, (height, width)),
  fill_color, selected["a"])`.
- `fill_onto_region(fill_color)` - `kind="act_on_region_selection"` (new
  kind: `fn(grid, selected, region_tag, *decoded_args) -> Grid`, the one
  place this ADR needs the region tag rather than just the indices).
  `dsl.fill(REGION_FN[selected_region["a"]](grid), fill_color,
  selected["a"])`.

**Slot `"a"` is the existing single-selection channel, unchanged.** Every
one of the 12 already-shipped `select`/`act_on_selection` actions
(`select_largest`, `recolor_selected`, `crop_to_selection`, ...) reads and
writes `selected["a"]` exactly as before - this ADR adds a second,
independent slot alongside it, not a replacement. An ordinary successful
`"transform"`-kind action still clears *both* slots and both region tags
(ADR-0011's staleness rule, now applied to the pair).

**Deliberately not added:** a `"full"` (no-crop) region option, or a
quadrant option - neither is needed by any of the 9 verified tasks, and
adding them speculatively would widen the menu past what's actually
evidenced. `cf98881b`'s 3-way-split variant and `a68b268e`'s
quadrant/pre-crop-original needs stay explicitly out of scope for this
pass.

## Mechanism - why this is cheaper than it first looked

The initial framing (presented to the user before this design was worked
out in full) assumed PPO's factored action head would need widening to
carry a slot index alongside the existing primitive/args. That turned out
to be unnecessary: `slot`, `region`, and `op` are ordinary decoded scalar
arguments, exactly like `COLOR_ARG`/`DIRECTION_ARG` already are, reusing
the *existing* generic "up to `MAX_ARITY` raw int args, each with its own
`decode`" plumbing every action already goes through. `MAX_ARITY` is
already 4 (from `commit`'s four coordinate/dimension args) and none of
these four new actions needs more than 4 args either - so **`MAX_ARITY`
does not change, and neither PPO's action head shape nor GP's flat gene
length changes at all.** The real surface area is: `arc_env/actions.py`
(three new `Action.kind` values, three new `ArgSpec` ordinal encoders,
`execute()` gains matching dispatch branches), `arc_env/env.py` (`selected`
becomes a 2-key dict plus a 2-key region-tag dict; both tracked as private
state, not part of the public step-args shape), `arc_env/episode_log.py`
(the logged `"selected"` field becomes `{"a": [[r,c],...]|None, "b":
[[r,c],...]|None}` instead of a flat list - a schema reshape, not additive;
`runs/` is local and gitignored, so this is a normal breaking evolution of
that schema, not a migration problem, same ethos as ADR-0017's "no
persistence until an explicit save"), `trainers/gp/fitness.py` (its own
copy of the select→act loop, since GP doesn't go through `ArcEnv`),
`trainers/gp/replay.py` and `trainers/ppo/warm_start.py` (both read the
logged `"selected"` shape back out - the observation encoding itself stays
a *single* mask channel, marking slot-`"a"` cells `1` and slot-`"b"` cells
`2`, rather than adding a second channel - cheaper than a channel-count
change and PPO's conv input shape is untouched), and `viz/frontend/src/
grid.ts` (two overlay colors instead of one).

## Alternatives considered

| Option | Why not chosen |
|--------|-----------------|
| Dynamic/variable-length list of selections | Handles arbitrary N regions, not just 2 (would also reach `a68b268e`'s 3-quadrant need) - but PPO's fixed-size action head and GP's flat gene list don't naturally support variable-length state; would need a `MAX_SELECTIONS` cap with masking or a real policy-architecture change. Bigger lift and risk than the evidence (a 2-region cluster) calls for right now. |
| Narrow region-scoped select only (no named slots, still one selection, agent picks a region before selecting) | Cheaper still, but doesn't generalize past this one cluster shape without a second "remember the last region's result while selecting again" side-channel anyway - ends up reinventing most of the 2-slot design without the clean, backward-compatible name for it. |
| More bespoke derived actions instead of a mechanism | Zero architecture risk, and technically *would* also reach these 9 tasks (per this ADR's own audit finding) - but each hardcodes one task's exact color/op/canvas-size constants, reproducing precisely the "single-task special case" pattern ADR-0013/0015 already argued against, and provides no reuse if a similar-shaped task shows up later. Rejected in favor of the more general, still-narrowly-scoped mechanism above. |

## Consequences

- Unlocks (pending implementation and the same fixture-level verification
  every other ADR here requires): `6430c8c4`, `94f9d214`, `ce4f8723`,
  `f2829549`, `fafffa47`, `99b1bc43`, `3428a4f5`, `dae9d2b5`, `1b2d62fb` - 9
  tasks, all already verified by direct bare-`dsl` replay against every
  train/test pair using this exact action set.
- `7c008303`, `a68b268e`, `928ad970`, `017c7c7b` remain uncurated - they
  need a related but different capability (holding the pre-crop original
  grid alongside a derived crop; `a68b268e` also needs three simultaneous
  regions, not two). Recorded as still-open, not silently dropped.
- `cf98881b` (a 3-way-split sibling of the confirmed cluster) is out of
  this pass's region menu (halves only) - a well-scoped, cheap follow-up if
  ever picked up, not attempted here.
- `docs/questions/f14-multi-selection-mechanism.md` records the full
  design/audit trail; `docs/QUESTIONS.md`'s F14 register row points there.
