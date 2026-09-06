# ADR-0016: A directional `stamp_selected` act-on-selection primitive

- Status: Accepted
- Date: 2026-09-06
- Deciders: repo owner (delegated design, per ADR-0015's own "next concrete
  candidate" framing), via conversation 2026-09-06

## Context

ADR-0015's Consequences named the next concrete candidate for F11's ongoing
coverage push: two `ofcolor`-flagged tasks, `a9f96cdd` and `d364b489`, that
ADR-0015 itself audited and found architecturally compatible with this
repo's single-selection-slot mechanism but not reachable with any existing
`act_on_selection` primitive. Both solvers share the same shape: a single
`ofcolor` selection, computed once, reused across **several** `shift`-then-
`fill` "stamp" calls - each shifting a copy of the *original* selected cells
by a fixed offset and filling the grid with a fixed color there, without
disturbing the grid at the original selected cells or anywhere else already
stamped.

`third_party/arc-dsl/solvers.py`'s known-correct solvers, transcribed:

```python
def solve_a9f96cdd(I):
    x1 = ofcolor(I, TWO)
    x2 = replace(I, TWO, ZERO)
    x3 = shift(x1, NEG_UNITY)
    x4 = fill(x2, THREE, x3)
    x5 = shift(x1, UP_RIGHT)
    x6 = fill(x4, SIX, x5)
    x7 = shift(x1, DOWN_LEFT)
    x8 = fill(x6, EIGHT, x7)
    x9 = shift(x1, UNITY)
    O = fill(x8, SEVEN, x9)
    return O

def solve_d364b489(I):
    x1 = ofcolor(I, ONE)
    x2 = shift(x1, DOWN)
    x3 = fill(I, EIGHT, x2)
    x4 = shift(x1, UP)
    x5 = fill(x3, TWO, x4)
    x6 = shift(x1, RIGHT)
    x7 = fill(x5, SIX, x6)
    x8 = shift(x1, LEFT)
    O = fill(x7, SEVEN, x8)
    return O
```

Both are, structurally, one `ofcolor` call followed by 4 `shift`+`fill`
pairs against the *same* indices computed once at the top (`x1`) - never a
`shift`/`fill` against a previously-stamped intermediate. `a9f96cdd` also
opens with a `replace(I, TWO, ZERO)` that clears the selected color before
stamping (equivalent to `recolor_selected(0)` once selected).

`a9f96cdd`'s 4 directions are diagonal (`NEG_UNITY`, `UP_RIGHT`,
`DOWN_LEFT`, `UNITY`); `d364b489`'s are the 4 cardinal directions
`move_selected` already supports (`DOWN`, `UP`, `RIGHT`, `LEFT`). No
existing action does "shift the selection and fill, without clearing the
original and without ending the reusability of the selection" -
`move_selected` clears the original (it moves the object, doesn't stamp a
copy) and `paint_selected_at` takes an absolute target coordinate, not a
directional offset.

`select_by_color(color)` was independently re-verified (this pass, not just
inherited) to produce indices identical to `ofcolor(I, color)` for every
train/test pair of both tasks - color 2 in `a9f96cdd` and color 1 in
`d364b489` are each not that task's background/majority color, so
`select_by_color`'s `without_bg=True` never excludes them. No new `select`
action is needed.

## Decision

**Add `stamp_selected`, a new `act_on_selection` action:**

```python
def _stamp_selected(grid, selected, color, direction_index):
    return dsl.fill(grid, color, dsl.shift(selected, _STAMP_DIRECTIONS[direction_index]))
```

This shifts the *selected indices* (an `Indices`, not a colored `Object`) by
a fixed offset and fills `grid` with `color` at those shifted positions.
Two properties fall out of this shape, both required by the fixture tasks:

1. **The original selected cells are not cleared** - `dsl.fill` only writes
   at the shifted indices, leaving every other cell (including the original
   selection) untouched. Unlike `move_selected` (which relocates an object,
   clearing its old position via the underlying `dsl.move`'s `cover`+`paint`
   pair), a "stamp" is additive.
2. **The selection is not invalidated**, letting one selection be reused
   across several stamp calls in a row. This isn't new machinery -
   `execute()`'s existing rule already only invalidates the selection after
   a successful ordinary `"transform"` action (ADR-0011); `act_on_selection`
   actions never invalidate it, which ADR-0015's `crop_to_selection` already
   established works for chaining one act-on-selection call after another.
   `stamp_selected` is simply the first curated task to *need* that chaining
   more than once (4 stamp calls in a row for both fixtures).

**A new, separate 8-direction menu**, since `a9f96cdd` needs 4 diagonal
offsets that `move_selected`'s existing `_DIRECTIONS` (cardinal-only) don't
have:

```python
_STAMP_DIRECTIONS = (
    constants.DOWN, constants.UP, constants.LEFT, constants.RIGHT,
    constants.UNITY, constants.NEG_UNITY, constants.UP_RIGHT, constants.DOWN_LEFT,
)
```

Index order: 0=DOWN, 1=UP, 2=LEFT, 3=RIGHT (same order as `_DIRECTIONS`),
4=UNITY, 5=NEG_UNITY, 6=UP_RIGHT, 7=DOWN_LEFT. Decoded via a new
`_decode_stamp_direction(raw) -> raw % 8`, wired through a new
`STAMP_DIRECTION_ARG` `ArgSpec` constructor - the same `ArgSpec`-based
pattern `DIRECTION_ARG` already uses, just a wider modulus and a separate
backing tuple. `move_selected`'s own `_DIRECTIONS`/`DIRECTION_ARG`/
`_decode_direction` are untouched.

**Curated task count: 36 → 38** (`CURATED_TASK_IDS`, `arc_env/
task_loader.py`), both same-shape:

| Task | Sequence |
|---|---|
| `a9f96cdd` | `select_by_color(2) → recolor_selected(0) → stamp_selected(3, NEG_UNITY) → stamp_selected(6, UP_RIGHT) → stamp_selected(8, DOWN_LEFT) → stamp_selected(7, UNITY)` |
| `d364b489` | `select_by_color(1) → stamp_selected(8, DOWN) → stamp_selected(2, UP) → stamp_selected(6, RIGHT) → stamp_selected(7, LEFT)` |

Both sequences were verified by direct replay against every train + test
pair of both tasks, via both the bare-function path and the raw-args
`actions.execute` path (`tests/test_dsl_regression.py`'s existing two
parametrized checks, auto-covering any new `CURATED_TASK_IDS` entry) -
exact match on every pair, not just a spot check.

## Mechanism

No change to `execute`'s signature or control flow: `stamp_selected` is
mechanically identical in shape to every other `act_on_selection` action
(`fn(grid, selected, *decoded_args) -> Grid`, invalid if nothing is
selected) - only its own function body differs, same as every other action
in that group. The only genuinely new pieces are the function itself, the
second direction menu/decode/`ArgSpec`, and the action's own two-arg
signature (`color`, `direction`) - no new representational mechanism.

`trainers/ppo/network.py`'s `IS_ACT_ON_SELECTION` mask derives from
`Action.kind == "act_on_selection"` generically, so `stamp_selected` is
automatically included - masked out of the policy's sampled distribution
whenever nothing is selected, with no code change needed there either (only
the comments/tests naming "the 6" needed updating to "7", same as ADR-0015
did for "the 5" → "the 6"). `MAX_ARITY` (`trainers/ppo/network.py`,
`trainers/gp/genome.py`) is unaffected - `stamp_selected`'s 2 args don't
exceed the existing max of 3 (`fill_cell`/`canvas`).

## Alternatives considered

| Option | Why not |
|--------|---------|
| Widen `move_selected`'s existing `_DIRECTIONS` tuple to 8 entries, reusing `DIRECTION_ARG`/`_decode_direction` (`raw % 8` instead of `raw % 4`) | Rejected: `move_selected` already has a verified curated fixture (`25ff71a9`, `select_largest → move_selected(DOWN)`) whose direction index (0) is baked into `CURATED_TASK_IDS` and was verified against a 4-entry menu. Changing the modulus changes what every *other* raw direction value (4-29 under `RAW_ARG_RANGE=30`) decodes to for `move_selected` too - a real behavior change for an already-shipped action, for no benefit (`move_selected` itself never needs a diagonal in any curated fixture). Same "don't speculatively widen a verified action" discipline ADR-0013 applied when it added `select_largest_no_diag` as a *new* selector rather than parameterizing `select_largest` with a connectivity flag. A fresh, separate menu for `stamp_selected` is strictly the smaller-footprint change. |
| Give `stamp_selected` a raw signed `(dx, dy)` offset instead of a fixed discrete menu | Rejected for the same reason `move_selected`'s own menu is discrete (see that action's ADR-0012 rationale, restated in `arc_env/actions.py`'s module docstring): an unconstrained signed offset blows up the raw-arg range for no benefit when both known fixtures only ever need one of 8 fixed directions. |
| Fuse "clear original + stamp" into one action (i.e. make `stamp_selected` behave like `move_selected` but keep the selection alive) | Rejected: neither fixture task needs the original cleared as part of the stamp step - `a9f96cdd` clears the original color with a *separate* `recolor_selected(0)` call before any stamping begins, precisely because the clear only needs to happen once, not once per stamp. Fusing it in would force a redundant re-clear on every stamp call and wouldn't match either solver's actual structure. |

## Consequences

- Action count grows 40 → 41 (`ACT_ON_SELECTION` gains `stamp_selected`).
  `N_ACTIONS` (`trainers/ppo/network.py`) and genome arity bounds
  (`trainers/gp/genome.py`) derive from `len(actions.ACTIONS)`/
  `actions.MAX_ARITY` already - no code change needed there (reconfirms,
  same note ADR-0011/0012/0013/0015 each already made).
- `CURATED_TASK_IDS` grows 36 → 38 (17 same-shape + 21 variable-shape - both
  new tasks are same-shape) - compute cost scales linearly with curated task
  count (ADR-0008), an accepted, already-priced-in tradeoff.
- This repo now has two direction menus with different sizes and different
  underlying `_DECODE`/`ArgSpec` pairs (`_DIRECTIONS`/`DIRECTION_ARG`/
  `_decode_direction` for `move_selected`, `_STAMP_DIRECTIONS`/
  `STAMP_DIRECTION_ARG`/`_decode_stamp_direction` for `stamp_selected`) - a
  real but narrow duplication, deliberately chosen (see Alternatives) over
  either widening the existing menu (riskier for an already-shipped fixture)
  or a shared parameterized menu size (no fixture task needs one). Any
  future action needing a directional argument should default to reusing
  whichever of these two menus already fits, rather than adding a third,
  unless a fixture genuinely needs a direction set neither covers.
- `stamp_selected` is the first curated task pattern where a single
  selection is deliberately reused across **more than two** consecutive
  `act_on_selection` calls (4, for both fixtures) - a stronger real-world
  confirmation of ADR-0015's "an act-on-selection action never invalidates
  the selection" design than ADR-0015's own 1-crop-then-1-transform fixture
  tasks exercised.
- This closes out the two `ofcolor`-flagged tasks ADR-0015's Consequences
  named as the next concrete candidate. The remaining 3 `ofcolor`-flagged
  tasks from that same audit pass (`7c008303`, `9ecd008a`, `a68b268e`) each
  need a genuinely different capability (multiple simultaneous selections
  across different coordinate spaces/grids, or a selection checked against
  an unrelated grid) that this ADR does not address - see ADR-0015's Context
  for the per-task detail. `docs/QUESTIONS.md` F11 is updated accordingly.
