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
- **KAN-1181 (2026-09-05) re-ran and broadened this audit after ADR-0013 added `select_largest_no_diag`/`select_tallest`, and it still comes back empty for all four actions.** Same AST-based method as ADR-0011/0012/0013 (parse every non-curated `solve_<task_id>` in `solvers.py`, restrict to solvers with no `lambda` and none of `compose`/`chain`/`fork`/`rbind`/`lbind`/`power`/`mapply`/`apply`/`mfilter`/`sfilter`/`extract`/`papply`/`prapply`/`occurrences`/`rapply`), checked this time against the *live* `arc_env.actions.ACTIONS` list (38 actions) rather than a hand-copied count, plus each of the 4 candidate actions individually and in every combination:
  - 63 non-curated solvers call `colorfilter`; only 9 survive the lambda/combinator filter, and every one needs a primitive `select_by_color` can't provide even after adding it - `gravitate` (a computed direction, not one of `move_selected`'s 4 fixed cardinals: `05f2a901`), a size-filter threshold computed from another object's count rather than a literal (`1fad071e`, `6455b5f5`), `difference`/complement-of-a-subset operations (`67385a82`, `aedd82e4`), a non-literal color argument (`leastcolor`: `67a423a3`, `fcb5c309`), or a destination position computed per-grid from an object's center (`88a10436`, which would also need `paint_selected_at` but for a row/col that isn't a fixed literal). `aedd82e4` looked the most promising by call-name matching alone (`colorfilter(objs, TWO)` → `sizefilter(_, ONE)` → `merge` → `fill(I, ONE, _)`) but was empirically refuted: replaying `dsl.colorfilter`/`dsl.sizefilter` against all 5 train/test pairs shows the color-2 objects and the size-1 subset of them are never equal (e.g. train pair 0 has 2 color-2 objects but only 1 of size 1), so `select_by_color(TWO)` selects strictly more cells than the solver's `sizefilter`-narrowed set - `recolor_selected` after it would recolor cells the solver leaves untouched.
  - 23 non-curated solvers call `cover`; only 1 (`42a50994`: `objects(I,T,T,T)` → `sizefilter(_, ONE)` → `merge` → `cover`) survives the filter - the specific "sizefilter matches exactly one object" special case this ticket asked to check. Empirically refuted: every one of its 5 train/test grids has 8-13 distinct size-1 objects, not one, so `select_smallest`/`select_largest` (which each return exactly one object) can only select and delete one of them, not the full set `sizefilter(_, ONE)` matches. The other 22 all still need at least one of `mapply`/`compose`/`mfilter`/`sfilter`/a `lambda` - unchanged from ADR-0012's finding.
  - 25 non-curated solvers call both `paint` and `shift` without also calling `move`; only 6 survive the filter, and all 6 compute their shift/paint position from the grid (`ofcolor`, `normalize`+`center`, `ulcorner`, `hperiod`-based periodicity) rather than using a fixed literal offset - exactly the shape `paint_selected_at`'s fixed `(row, col)` args can't express as one static action sequence replayed identically across every train/test pair. None is a non-destructive "stamp" at a fixed destination.
  - A full closure check (every non-curated, lambda/combinator-free solver's call-name set checked against curated ∪ any subset of the 4 candidates' underlying primitive names) surfaced 4 solvers nominally needing only `select_by_color`+`paint_selected_at` (`11852cab`, `a740d043`, `e3497940`, `e98196ab`) - but all 4 turned out to be false positives on inspection: each calls `merge(objects(grid, ...))` directly (merging *every* object on the grid, or half of it), never `colorfilter`, so `select_by_color`'s literal-color filter never actually matches what these solvers compute; three of the four also apply mirror/half-grid transforms to the merged shape before painting, which no single `select_*` + `paint_selected_at`/`recolor_selected` step can reproduce.
  - No solver anywhere in the corpus computes anything resembling `select_unique_color`'s "the object(s) whose color occurs in exactly one object" criterion - the closest related primitives (`leastcolor`, `mostcommon`/`leastcommon`, `fgpartition`) all operate on pixel-count/color-frequency over the whole grid, not per-object color-uniqueness, so there was no candidate to even empirically test.

  This is the same "no, and here's why" outcome ADR-0013's own precedent (declining to curate `1c786137`) established as acceptable: a real, broadened search, not a re-statement of ADR-0012's original one, still finds nothing. `select_by_color`, `select_unique_color`, `delete_selected`, and `paint_selected_at` remain verified by direct unit test only (`tests/test_actions.py`), not a curated-task fixture.
- `ea32f347`'s 5-step curated program turned out to be GP's other zero-success task alongside `5bd6f4ac` (KAN-1179, 2026-09-05 investigation, see `docs/PLAN.md`'s Open risks for the full writeup) - not a `select`/`act_on_selection`-specific defect: the program does have a real, monotonic partial-credit gradient (0.0 → 0.34 → 0.34 → 0.79 → 0.79 → 1.0 across its five prefixes), but a structurally unrelated single action (`replace(5,1)`/`switch(1,5)`) scores higher immediate similarity (0.4486) than the true program's own first step, purely by coincidence of this task's grid colors. GP's tournament selection converges onto that decoy lineage within ~5 generations and never recovers - a deceptive-local-optimum failure, not a selection-mechanism-specific one. A real but secondary structural risk this ADR's own mechanism does introduce: a successful ordinary `"transform"` action clears the current selection (this ADR's `execute()`), so a mutation/crossover-inserted transform landing between a `select_*` gene and its paired `recolor_selected`/other `act_on_selection` gene silently breaks that segment - confirmed directly, and a real (if secondary) source of fragility for any multi-gene program that chains a select with its act-on-selection step, beyond what zero/one-arg-transform-only curated tasks risk.
