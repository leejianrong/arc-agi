# ADR-0019: `canvas_mostcolor` and `swap_two_least_colors` - two more single-primitive derived actions

- Status: Accepted
- Date: 2026-09-06
- Deciders: repo owner (delegated design, per F11's ongoing incremental-coverage cadence), via conversation 2026-09-06

## Context

F11's 2026-08-29/2026-09-05 coverage audits (`docs/QUESTIONS.md` F11,
`scripts/audit_action_coverage.py`) flagged two ARC-AGI-1 training tasks as
needing exactly one new primitive by name: `5582e5ca` (`mostcolor`) and
`aabf363d` (`leastcolor`). Both were independently re-verified computationally
against every train+test pair before this pass, per the same discipline
ADR-0013/ADR-0015 applied to their own name-flagged candidates (a bare name
match doesn't guarantee the actual parameterization/composition is curated).

`third_party/arc-dsl/solvers.py`'s known-correct solvers, transcribed:

```python
def solve_5582e5ca(I):
    x1 = mostcolor(I)
    O = canvas(x1, THREE_BY_THREE)
    return O

def solve_aabf363d(I):
    x1 = leastcolor(I)
    x2 = replace(I, x1, ZERO)
    x3 = leastcolor(x2)
    O = replace(x2, x3, x1)
    return O
```

Neither fits this module's "Grid [, scalar args] -> Grid" signature as a bare
1:1 wrap of a single `dsl` call: `5582e5ca`'s solver feeds `canvas` a *derived*
color argument (the grid's own most-common color) rather than a
literal/agent-chosen one, and `aabf363d`'s solver is a 4-step composition with
no scalar args at all - it derives everything (`leastcolor` twice, `replace`
twice) from the grid itself. This is exactly the "derived, not drawn 1:1 from
a single `dsl` call" pattern `fill_cell`/`canvas`/the self-concatenation
actions/`stamp_selected` already established (see `arc_env/actions.py`'s
module docstring) - no new representational mechanism, just two more curated
functions of that same shape.

## Decision

**Add two new derived actions:**

```python
def _canvas_mostcolor(grid, height, width):
    return dsl.canvas(dsl.mostcolor(grid), (height, width))

def _swap_two_least_colors(grid):
    a = dsl.leastcolor(grid)
    g2 = dsl.replace(grid, a, 0)
    b = dsl.leastcolor(g2)
    return dsl.replace(g2, b, a)
```

`canvas_mostcolor(height, width)` keeps `height`/`width` as real, agent-chosen
`DIM_ARG`s - the same two args the existing generic `canvas` action already
exposes - rather than hardcoding `(3, 3)` as a zero-arg action. The *only*
thing `5582e5ca`'s solver actually fixes is the fill color (derived
internally via `dsl.mostcolor`, not agent-chosen); the dimensions are not
task-specific in any deeper sense, so keeping them as general args matches
`canvas`'s own precedent and keeps the action useful beyond this one fixture
(see Alternatives).

`swap_two_least_colors()` is zero-arg, like `hconcat_self` and friends - every
value it needs (which two colors, in which order) is derived from the grid,
never agent-chosen. It swaps a grid's two least-common colors: it clears the
overall least-common color to `0`, then replaces the *new* least-common color
of that intermediate result with the original one - which, because clearing
step 1's color removes it from the color-count entirely, nets out to swapping
the original two least-common colors' labels wherever they occur (assuming
grid dtype/idiom already reserves `0` as backgroundizable, matching every
other curated action's implicit "the DSL already treats `0` this way"
assumption).

**Curated task count: 38 -> 40** (`CURATED_TASK_IDS`, `arc_env/task_loader.py`):

| Task | Sequence |
|---|---|
| `5582e5ca` | `canvas_mostcolor(3, 3)` |
| `aabf363d` | `swap_two_least_colors()` |

Both sequences were verified by direct replay against every train + test pair
of both tasks, via both the bare-function path and the raw-args
`actions.execute` path (`tests/test_dsl_regression.py`'s existing two
parametrized checks, auto-covering any new `CURATED_TASK_IDS` entry) - exact
match on every pair, not just a spot check.

**Shape classification, confirmed computationally rather than assumed:**
both new tasks are same-shape, not variable-shape. This contradicts the
inherited brief's initial guess for `5582e5ca` ("output is always 3x3
regardless of input shape, so almost certainly variable-shape") - checking
the actual task data shows every one of `5582e5ca`'s 4 train+test pairs
already has a 3x3 *input* too, so input shape equals output shape for every
pair; `aabf363d` never crops/resizes at all, so it was already expected to be
same-shape and is. Neither is added to `VARIABLE_SHAPE_TASK_IDS`. Same-shape
count: 17 -> 19; variable-shape count: unchanged at 21; total 38 -> 40.

## Mechanism

No change to `execute`'s signature or control flow. Both actions are ordinary
`kind="transform"` actions (`fn(grid, *decoded_args) -> Grid`), mechanically
identical in shape to every other action in `ZERO_ARG`/`TWO_ARG` - only their
own function bodies differ, same as every prior derived action. `canvas_mostcolor`
is registered in `TWO_ARG` (its arity is 2, matching that group's existing
`replace`/`switch` entries) even though its argument *kinds* are both
`DIM_ARG` rather than `COLOR_ARG` - `Action`/`execute` group membership is
organizational only, arity and `ArgSpec.kind` are what actually drive decoding
and validation, and neither cares which list an action lives in.
`swap_two_least_colors` is registered in `ZERO_ARG`, alongside `hconcat_self`
and friends.

`execute`'s generic per-arg validity loop has no special case for `"dim"`-kind
args beyond `commit`'s own cross-argument bounds check - `canvas_mostcolor`'s
`height`/`width` decode via the existing `_decode_dim` (`raw + 1`, capped at
30 by `RAW_ARG_RANGE`), and the post-execution `new_h`/`new_w` bounds check
`execute` already runs for every ordinary transform catches anything out of
range, exactly as it already does for the generic `canvas` action. No new
validation code needed.

`trainers/ppo/network.py`'s `N_ACTIONS` and `trainers/gp/genome.py`'s arity
bounds both derive from `len(actions.ACTIONS)`/`actions.MAX_ARITY` already -
no code change needed there (reconfirms, same note every prior action-adding
ADR has made). `MAX_ARITY` is unaffected - both new actions' arities (2 and 0)
don't exceed the existing max of 3 (`fill_cell`/`canvas`).

## Alternatives considered

| Option | Why not |
|--------|---------|
| Hardcode `canvas_mostcolor` as a zero-arg action fixed at `(3, 3)`, matching `5582e5ca`'s literal solver call | Rejected: the derived color is the only thing this task's solver actually fixes about `canvas`'s two arguments - hardcoding the dimensions too would be narrower than necessary for a generally useful action, and would leave `canvas_mostcolor` unable to do anything a plain `canvas` call followed by a manual color pick couldn't already do more generally. Keeping `height`/`width` as real `DIM_ARG`s matches the existing `canvas` action's own precedent (which itself takes all three of `value`/`height`/`width` as real args) and costs nothing extra in raw-arg range, since `DIM_ARG`/`_decode_dim` are already shared infrastructure. |
| Give `swap_two_least_colors` an explicit `color` arg (e.g. "swap the least-common color with a chosen one") instead of deriving both colors | Rejected: `aabf363d`'s solver derives *both* colors from the grid with no scalar argument at all - adding an agent-chosen arg where the fixture task doesn't need one would just be surface-area for no benefit, the same restraint ADR-0011/ADR-0016 already applied to their own zero-arg derived actions (`select_largest`/`hconcat_self`) where nothing in the fixture needed a knob. |
| Add `mostcolor`/`leastcolor` as standalone bare-`dsl`-call actions, relying on a later composition step to build `canvas`/`replace` calls around them | Not applicable: `mostcolor`/`leastcolor` are `Grid -> Integer`, not `Grid -> Grid` - they don't fit this module's action signature at all (per the module docstring's core restriction) and can't be a step in a flat "pick one primitive per step" action space on their own. Deriving their result *inside* a `Grid -> Grid` action is the only way either primitive's information reaches the agent, exactly the same reasoning `canvas`/`fill_cell` already established for constants/coordinates that aren't themselves grid transforms. |

## Consequences

- Action count grows 41 -> 43 (`TWO_ARG` gains `canvas_mostcolor`, `ZERO_ARG`
  gains `swap_two_least_colors`). `N_ACTIONS`/genome arity bounds derive from
  `len(actions.ACTIONS)`/`actions.MAX_ARITY` already - no code change needed
  there (reconfirms, same note every prior action-adding ADR has made).
- `CURATED_TASK_IDS` grows 38 -> 40 (19 same-shape + 21 variable-shape - both
  new tasks are same-shape, correcting the inherited brief's initial "almost
  certainly variable-shape" guess for `5582e5ca` after checking the actual
  task data) - compute cost scales linearly with curated task count
  (ADR-0008), an accepted, already-priced-in tradeoff.
- This is the fourth pass (after `fill_cell`/`canvas`, the self-concatenation
  actions, and `stamp_selected`) landing curated tasks via a derived action
  rather than a bare 1:1 primitive wrap - the pattern is now clearly the
  default way this repo lands a "needs exactly one new primitive by name"
  F11 candidate whenever that primitive's own signature doesn't fit
  `Grid [, scalar args] -> Grid` directly.
- Neither new action touches the object-selection mechanism (ADR-0011 and
  on) at all - both are plain `"transform"` actions, so a successful call to
  either invalidates any current selection, same as every other ordinary
  transform.
