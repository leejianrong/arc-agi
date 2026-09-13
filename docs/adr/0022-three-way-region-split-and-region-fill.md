# ADR-0022: a 3-way region split and an explicit-target region fill

- Status: Accepted
- Date: 2026-09-13
- Deciders: repo owner, via conversation 2026-09-13

## Context

ADR-0020's own Consequences named two near-misses of its region-vs-half
cluster as small, scoped follow-ups: `cf98881b` (shares the cluster's shape
but splits the grid three ways, not in half) and `1b2d62fb` (shares the
shape too, but needs a `replace` sandwiched between selecting and filling
that the original design couldn't carry a selection through). Both
verified end to end below before any design was committed to.

**This turned out a little bigger than "small."** Widening the region menu
to 3-way splits is cheap, but `cf98881b`'s actual need - fill a selection
computed in one region's coordinate frame onto a *different*, explicitly
chosen region, not the region the selection itself came from - needs a
genuinely new `Action.kind`, not just more region entries. `1b2d62fb`
turned out cheaper than first framed: no new kind at all, one action that
fuses `replace` and `fill` into a single atomic step (the same "derived,
bundles more than one `dsl` call" pattern used throughout this project),
so the transform-clears-selection question this file's own earlier
analysis raised never comes up.

## Decision

**Three new region entries**, widening `_REGIONS` from 4 to 7 and
`REGION_ARG`'s decode range from `raw % 4` to `raw % 7` (backward
compatible - `CURATED_TASK_IDS` stores already-decoded region indices, so
existing entries using `0`-`3` are unaffected by more values becoming
reachable):

```python
def _left_third(grid):
    return dsl.first(dsl.hsplit(grid, 3))

def _middle_third(grid):
    parts = dsl.hsplit(grid, 3)
    return dsl.first(dsl.remove(dsl.first(parts), parts))

def _right_third(grid):
    return dsl.last(dsl.hsplit(grid, 3))
```

**One new action, `fill_slot_onto_region(slot, target_region,
fill_color)`** - a new `Action.kind` (`"fill_slot_onto_region"`), since it
needs an *explicit* slot (not always slot `0`/"a", unlike `fill_onto_
region`) and an *explicit* target region independent of that slot's own
tag:

```python
def _fill_slot_onto_region(grid, selected_for_slot, target_region, fill_color):
    return dsl.fill(_REGIONS[target_region](grid), fill_color, selected_for_slot)
```

Invalid if the named slot is empty or untagged, or if its tagged region's
crop and the target region's crop don't have matching shapes (the same
comparability check `combine_slots` already makes). Leaves `selected`/
`selected_region` unchanged on success, matching `act_on_selection`/
`act_on_region_selection`'s existing convention.

**One more new action, `replace_region_and_fill(replacee, replacer,
fill_color)`** - reuses the *existing* `"act_on_region_selection"` kind
(it genuinely consumes `selected`, in its own `fill` step, so there's no
"precondition for no reason" concern the way a bare region-scoped
`replace` alone would raise):

```python
def _replace_region_and_fill(grid, selected, region, replacee, replacer, fill_color):
    base = dsl.replace(_REGIONS[region](grid), replacee, replacer)
    return dsl.fill(base, fill_color, selected)
```

**Verified sequences** (both replayed against every train/test pair with
these exact formulas before this ADR was written):

| Task | Sequence |
|------|----------|
| `cf98881b` | `select_by_color_in_region(a, left_third, 4)` → `select_by_color_in_region(b, middle_third, 9)` → `fill_slot_onto_region(b, right_third, 9)` → `recolor_selected(4)` (existing action - slot `a`'s selection from step 1 is untouched by the intervening steps) |
| `1b2d62fb` | `select_by_color_in_region(a, lefthalf, 0)` → `select_by_color_in_region(b, righthalf, 0)` → `combine_slots(intersect)` → `replace_region_and_fill(9, 0, 8)` |

## Alternatives considered

| Option | Why not |
|--------|---------|
| Make `act_on_region_selection` itself slot-parameterized (add a `slot` arg to every action of that kind) instead of a new kind for `fill_slot_onto_region` | Would change `fill_onto_region`'s existing arg shape, a breaking change to an already-shipped, already-curated action (`dae9d2b5`). A new, additive kind costs nothing extra and leaves ADR-0020's work untouched. |
| A bare region-scoped `replace` action (no fused `fill`), relying on a "this kind doesn't clear the selection" rule | Rejected - this is exactly the shape ADR-0015's `9ecd008a` no-go warned against: a transform-shaped action that doesn't consume the selection it would need to leave alive is a precedent this project has already ruled out once. Fusing `replace` and `fill` into one atomic action sidesteps the question entirely, the same way `fill_new_canvas` already fuses `canvas` and `fill`. |

## Consequences

- Unlocks `cf98881b` and `1b2d62fb` - the last 2 tasks named as follow-ups
  in ADR-0020's own Consequences section.
- Action count: 53 → 55. Curated tasks: 55 → 57 (both variable-shape: 34 →
  36).
- `docs/questions/f13-interactive-editor.md`'s F13 Stage 2 backlog and
  `docs/questions/f14-multi-selection-mechanism.md`'s F14 "not landed
  here" section both update to reflect these as landed.
- Remaining backlog, unchanged: `7c008303`/`a68b268e`/`928ad970`/
  `017c7c7b` (the "hold the pre-crop original grid" need) - still its own
  open design question, not touched by this pass.
