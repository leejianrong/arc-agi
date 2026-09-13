# F14: the multi-selection/cross-grid mechanism

**Status:** decided (user, 2026-09-13). Design verified; implementation not
yet landed.

F11 named a decision bigger than a normal incremental ADR: two tasks
(`7c008303`, `a68b268e`) need more than one live selection or grid state at
once - PPO's factored action head, GP's flat gene list, episode logging,
and the visualizer's selection overlay would all need to represent more
than one live thing, a redesign F11 explicitly deferred pending a broader
audit and real design sign-off, not built on the spot.

## The broader audit (2026-09-13)

Before designing anything, ran a structural audit of the 92 tasks not
already excluded by a higher-order combinator (`scripts/
audit_action_coverage.py`'s `free_by_name` + `one_new` + `two_new` +
`three_plus_new` buckets). Built a classifier modeling what a single
mutable-grid `ArcEnv` episode can and can't do, validated it against the 10
tasks whose status was already known from F11/F13 (6 confirmed needing
extra state, 4 confirmed not needing it - zero mismatches), then ran it
across all 92.

61/92 flagged. Reading through all 61 by hand forced a reframing that
matters for the cost/benefit case: **because PPO/GP train one dedicated
policy per task (ADR-0008), almost any task whose solver avoids
higher-order combinators is technically reachable by adding one more
bespoke derived action that hardcodes its exact fixed pipeline** - the
same pattern already used 10 times (`fill_cell`, `canvas`,
`canvas_mostcolor`, `swap_two_least_colors`, the 4 self-concat actions).
About 40 of the 61 flagged tasks are exactly this (fixed quad-mirror
tiling, fractal cellwise expansion, canvas-built geometric patterns),
several sharing enough structure that one action would unlock 2-3 tasks at
once. So the real question was never "is this technically reachable" -
it's whether to keep adding narrower single-purpose actions (cheap, but
each barely tests genuine search) or build something reusable that lets
the agent make an instance-dependent choice at decision time.

One recurring shape does need that: split the grid into two halves, pull a
color's cell locations independently from each half, combine the two
location-sets with a set operation (intersection/union/symmetric
difference), paint the result. Confirmed in 8 tasks: `6430c8c4`,
`94f9d214`, `ce4f8723`, `f2829549`, `fafffa47`, `99b1bc43`, `3428a4f5`,
`dae9d2b5` - a materially bigger and better-evidenced number than the
original 2-task motivation. (A ninth, `1b2d62fb`, looked like the same
shape but turned out to need a fifth capability this design doesn't add -
see `docs/adr/0020-region-scoped-dual-selection.md`'s own note on it.)

**Important, checked directly:** this is *not* the same need `7c008303`/
`a68b268e` have. Those need the pre-crop *original* grid held alongside a
derived crop, not two regions of the still-current grid, and `a68b268e`
specifically needs three simultaneous regions. Replaying `a68b268e`
against the design below breaks the moment the current grid is replaced
by one quadrant - the other two can no longer be re-derived from the
(by-then-gone) original.

## Design options presented, and the scope check that followed

Presented four options to the user: (1) two fixed named selection slots
plus a small region-scoping menu, (2) a dynamic/variable-length list of
selections, (3) a narrower region-scoped-select-only variant with no named
slots, (4) skip the mechanism and keep adding bespoke derived actions
instead. **User picked (1).**

That surfaced the scope mismatch above - the chosen design reaches the
8-task region-vs-half cluster, not the 2 tasks that originally motivated
this thread. Checked explicitly with the user before proceeding: **build
it as scoped for the 8-task cluster; leave `7c008303`/`a68b268e`
(and F13 Stage 1's `928ad970`/`017c7c7b`, which need the same
"hold the pre-crop original" capability) an explicitly open, separate
question - not silently dropped, not stretched to cover by force.**

## The verified design

Full design, verification method, and exact new actions: `docs/adr/
0020-region-scoped-dual-selection.md`. Headline: a second named selection
slot (`"a"`/`"b"`), a 4-way region menu (`tophalf`/`bottomhalf`/`lefthalf`/
`righthalf`, all four already-curated transforms reused rather than
duplicated), and 4 new actions (`select_by_color_in_region`,
`combine_slots`, `fill_new_canvas`, `fill_onto_region`). All 8 target
tasks verified by direct bare-`dsl` replay against every train/test pair
before the ADR was written, not just plausibility-checked.

One correction worth recording: the design turned out cheaper than first
previewed to the user. `slot`/`region`/`op` are ordinary decoded scalar
arguments reusing the *existing* generic argument-decoding mechanism every
action already goes through - `MAX_ARITY` doesn't change, so neither
PPO's action head shape nor GP's flat gene length changes at all. The
real surface area is `arc_env/actions.py` (three new action kinds),
`arc_env/env.py` (the selection state becomes a 2-key dict plus a 2-key
region-tag dict), `arc_env/episode_log.py` (the logged `"selected"` field
reshapes from a flat list to a 2-key dict), `trainers/gp/fitness.py`/
`replay.py` and `trainers/ppo/warm_start.py` (read that shape back out),
and `viz/frontend/src/grid.ts` (two overlay colors instead of one) - the
observation encoding itself stays a single mask channel (slot `"a"` marked
`1`, slot `"b"` marked `2`), not a new channel.

## Not landed here

`cf98881b` shares the cluster's shape but splits the grid three ways, not
in half - out of this pass's region menu. `1b2d62fb` shares the shape too
but needs a `replace` sandwiched between selecting and filling, which this
design can't carry a selection through - a fifth capability, not
attempted here. **Both landed 2026-09-13 (ADR-0022):** a 3-way region
split plus a new `fill_slot_onto_region` action for `cf98881b`; a fused
`replace_region_and_fill` action (reusing the existing `act_on_region_
selection` kind, no new selection-survives-a-transform precedent needed)
for `1b2d62fb`. `7c008303`, `a68b268e`, `928ad970`, `017c7c7b` remain
uncurated, needing the still-separate "hold the pre-crop original"
capability - untouched by ADR-0022.

**Landed by:** ADR-0020 (design); implementation tracked on the Pandan
board's multi-selection epic, not yet merged as of this write-up.
