# ADR-0012: Object-selection menu expansion (ADR-0011's deferred menu)

- Status: Accepted
- Date: 2026-09-04
- Deciders: repo owner (delegated design, per ADR-0011's own deferral), via conversation 2026-09-04

## Context

ADR-0011 landed a deliberately minimal object-selection menu (`select_largest`,
`select_smallest`, `commit_selection`) to prove the "selected patch" side-channel
mechanism end-to-end, and explicitly deferred the rest: `select_by_color`,
`select_unique_color`, and the act-on-selection actions `delete_selected`,
`recolor_selected`, `move_selected`, `paint_selected_at`. This ADR lands that
deferred menu, informed by the same kind of `third_party/arc-dsl/solvers.py`
audit ADR-0011 used - restricted this time to solvers not already curated,
checking which become fully expressible (no `lambda`/`compose`/`mapply`/
other higher-order combinator) once curated(30) is extended with each
candidate primitive.

**Audit finding: `move_selected` and `recolor_selected` each have a clean,
literal, single-selection fixture task using the exact `objects(I, True,
True, True)` variant ADR-0011 already curates - `select_by_color`,
`select_unique_color`, `delete_selected`, and `paint_selected_at` do not.**
`25ff71a9` (`move(I, first(objects(I,T,T,T)), DOWN)`) and `ea32f347`
(`replace` + two rounds of `argmax`/`argmin`-select + `fill`) both verify
exact-match against every train and test pair using this ADR's new actions.
The audit also turned up a third, free addition needing **no new action at
all**: `1cf80156` (`subgrid(first(objects(I,T,T,T)), I)`) is solvable with
ADR-0011's existing `select_largest`/`select_smallest` + `commit_selection`,
since a lone object is trivially both the largest and the smallest - it was
missed by ADR-0011's own audit because that pass searched for `argmax`/
`argmin` call sites specifically, not `first`.

For `select_by_color`, `select_unique_color`, `delete_selected`, and
`paint_selected_at`, the audit came back empty: every `colorfilter`- or
`cover`-using solver in the corpus that isn't already curated also needs a
combinator this action space excludes by design (`mapply`, `compose`,
`mfilter`, `sfilter`, or a `lambda`), or operates on a *set* of objects at
once (`sizefilter` + `merge`) rather than the single selected patch this
mechanism represents. This is a real, honest gap, not a search-effort
failure - see Consequences.

## Decision

**Land all six actions this pass**, holding the four without a literal
fixture to the same bar `fill_cell`/`canvas` (ADR-0002/ADR-0010 Phase 1)
already established: verified by direct unit test against hand-constructed
grids, not a curated ARC task, when no solver in the corpus needs them
literally. Splitting `select_by_color`/`select_unique_color` from the four
`act_on_selection` additions into two separately-verified slices/PRs, same
"small provable slice" discipline as ADR-0010 Phase 1 and ADR-0011.

### Mechanism changes

- **`select` actions can now take scalar args.** ADR-0011's `execute()`
  called a `"select"` action's `fn` with only `grid`; `select_by_color`
  needs a `color` argument the same way an ordinary `"transform"` action
  already takes one, so `execute()`'s `"select"` branch now calls
  `action.fn(grid, *decoded.values())` - a no-op for the two existing
  zero-arg selectors (`decoded` is empty). The criterion itself is still
  always fixed internally per action (never a `Callable`); only ADR-0011's
  "selectors take no arguments" claim narrows, not the "no closures" rule
  ADR-0001/ADR-0011 actually care about.
- **`move_selected` needs a signed direction, not a coordinate.** Existing
  `ArgSpec` kinds (`color`, `factor`, `coord`, `dim`) all decode to
  non-negative values; a move offset needs sign. Rather than widen the raw
  encoding, a new `DIRECTION_ARG`/`"direction"` kind decodes `raw % 4` into
  an index over a small fixed `_DIRECTIONS = (DOWN, UP, LEFT, RIGHT)` tuple
  (`arc-dsl`'s own constants) - the same "curated discrete menu, not a raw
  general value" choice `FACTOR_ARG`'s `{2, 3, 4}` already made, and
  sufficient for the one verified fixture (`25ff71a9`, which only ever
  needs `DOWN`).
- No other execution-path change: `"act_on_selection"` already accepted
  decoded args (ADR-0011), so `delete_selected` (0-arg), `recolor_selected`
  (`color`), and `paint_selected_at` (`row`, `col`, reusing `fill_cell`'s
  arg style) need nothing new there.

### New actions

All in `arc_env/actions.py`, each a thin wrap of one or two `arc-dsl`
primitives against `grid`/`selected`:

- `select_by_color(grid, color)` - `dsl.toindices(dsl.merge(dsl.colorfilter(objects(grid), color)))`, empty selection if no object has that color. Merges every matching object's indices into one selection (there can be more than one object of a given color) rather than picking just one - the natural generalization of `colorfilter`'s own return type (a set of objects), and the same shape the corpus's own `colorfilter`+`sizefilter`+`merge` solvers already use.
- `select_unique_color(grid)` - selects the object(s) whose color occurs in exactly one object of the grid's segmentation (a color-uniqueness criterion, the same "fixed criterion, resolved at execute-time against the live grid" shape as `select_largest`/`select_smallest`'s size criterion).
- `delete_selected(grid, selected)` - `dsl.cover(grid, selected)` (paints the selection over with the grid's background color).
- `recolor_selected(grid, selected, color)` - `dsl.fill(grid, color, selected)`.
- `move_selected(grid, selected, direction)` - `dsl.move(grid, dsl.toobject(selected, grid), _DIRECTIONS[direction])`.
- `paint_selected_at(grid, selected, row, col)` - stamps the selected object's cells (not the grid) so its upper-left corner lands at `(row, col)`, via `dsl.paint(grid, dsl.shift(toobject(selected, grid), (row - ul_row, col - ul_col)))` - a non-destructive "copy", unlike `move_selected`'s cover-then-paint "cut".

### Curated tasks added this pass

`CURATED_TASK_IDS` grows 26 → 29 (14 same-shape + 15 variable-shape):

- `1cf80156` (variable-shape) - no new action, see audit finding above.
- `25ff71a9` (same-shape) - `[select_largest, move_selected(DOWN)]`.
- `ea32f347` (same-shape) - `[replace(5,4), select_largest, recolor_selected(1), select_smallest, recolor_selected(2)]`.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Only land `move_selected`/`recolor_selected` (the two with a literal fixture), defer the other four again | Rejected: `select_by_color`/`select_unique_color`/`delete_selected`/`paint_selected_at` are simple, unambiguous DSL wraps whose correctness is fully checkable by direct unit test (no ARC-task ambiguity), and the user explicitly asked for the full menu. Deferring them a second time for a search-effort gap rather than a genuine design risk doesn't fit `fill_cell`/`canvas`'s own precedent of shipping a not-solver-drawn action once it's individually verifiable. |
| Give `select_by_color` a raw unconstrained `Indices`-returning closure instead of a fixed color-menu action | Reintroduces exactly the `Callable`-argument shape ADR-0001 excludes - a color is a scalar, so it fits the existing `COLOR_ARG` machinery directly, no closure needed. |
| `move_selected` with two signed coordinate args instead of a 4-direction menu | A general signed 2D offset needs a raw-arg range and sign encoding none of the other actions use, for a fixture that only ever needs one of 4 cardinal directions; the smaller, curated `DIRECTION_ARG` menu mirrors `FACTOR_ARG`'s existing "small fixed set, not a raw range" choice and is trivially widened later if a future fixture needs it. |
| Single combined PR for both slices | Rejected for the same reason ADR-0010 Phase 1 and ADR-0011 kept their slices separate: `select_by_color`/`select_unique_color` are additive to `"select"` (execute() signature change only) while the four `act_on_selection` actions are a larger, distinct risk surface (4 new mutating actions) - separately reviewable PRs, same repo convention. |

## Consequences

- `execute()`'s `"select"` branch signature is unchanged (still `(new_grid, new_selected, decoded_args, valid)`), but its *behavior* changes: a `"select"` action's `fn` now receives decoded args. Every existing zero-arg selector is unaffected (`decoded` is `{}`), so this is additive, not breaking, for ADR-0011's landed actions.
- Action count grows 30 → 36; `N_ACTIONS`/`MAX_ARITY` in `trainers/ppo/network.py` and `trainers/gp/genome.py` derive from `len(actions.ACTIONS)`/`actions.MAX_ARITY` already, so no code change needed there (reconfirms, doesn't revise, ADR-0011's own claim about this).
- As with ADR-0011, no visualizer change ships in this ADR - episode replay still can't show what a `select_*`/`act_on_selection` step did to the hidden selection state. That gap is closed in a following slice (visualizer selection-overlay), not bundled here.
- `select_by_color`/`select_unique_color`/`delete_selected`/`paint_selected_at` ship without a curated-task-level correctness signal the way every other curated action has one; if any of the four turns out to be subtly wrong in a way the unit tests' hand-constructed grids don't exercise, there is no ARC-task regression test to catch it. Accepted as the same tradeoff `fill_cell`/`canvas` already made, not a new category of risk.
- Compute cost still scales linearly with curated task count (ADR-0008); 26 → 29 is a small addition, consistent with the existing accepted tradeoff (`docs/PLAN.md` Open risks).
- Deferred, not designed: additional `objects(...)` connectivity variants (still needed for `be94b721`/`1c786137`, per ADR-0011's own deferral - the audit for this ADR reconfirmed they remain out of reach), and any `select_by_color`/`select_unique_color`/`delete_selected`/`paint_selected_at`-using curated task, should one turn up in a future re-audit or a broader task-generation pass (e.g. `re-arc`-style synthetic variants).
