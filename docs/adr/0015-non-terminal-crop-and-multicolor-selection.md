# ADR-0015: Non-terminal crop-to-selection, a multicolor `objects()` variant, and why `ofcolor` didn't pan out the same way

- Status: Accepted
- Date: 2026-09-06
- Deciders: repo owner (delegated design, per F11's own next-step framing), via conversation 2026-09-06

## Context

The 2026-09-05 F11 coverage-audit refresh (`scripts/audit_action_coverage.py`,
`docs/QUESTIONS.md` F11) flagged `ofcolor` and `first` as the two
highest-leverage "one new primitive" clusters among the 370 still-uncurated
ARC-AGI-1 training tasks - 5 tasks apiece by the audit's method (does the
solver call exactly one primitive name outside this repo's curated set?).
That method is a name-level check only, and its own documentation says so
("name overlap doesn't guarantee the actual parameterization/composition is
curated" - the same caveat ADR-0013 hit with `select_tallest`/`1c786137`).
This ADR is the manual reachability audit both clusters needed before
committing to either as a real next slice.

**`first`'s cluster, re-examined:** every `first`-flagged solver is
`first(objects(I, <triple>))` → `subgrid` → zero or more further grid
transforms. Checked computationally against every train/test pair, all 5
tasks' `objects(...)` call yields **exactly one object** per grid, so
`first` (an arbitrary-order pick) is trivially equivalent to `argmax`/
`argmin` by any criterion - never actually order-dependent. That reduces the
question to: is `argmax(objects(...), size)` → `subgrid` → `<transform>`
reachable with the existing `select_*` + `commit_selection` mechanism? For 4
of the 5, no - `commit_selection` unconditionally ends the episode at the
`subgrid` step (ADR-0011's design, reconfirmed by ADR-0013), so the
`<transform>` after it (`hconcat_self`, `vmirror`, `upscale`, or
`lefthalf`+`tophalf`) can never actually run in `ArcEnv`'s real step loop -
exactly the same structural wall ADR-0013 hit and left unresolved for
`1c786137` ("trim after subgrid"). This is the same root cause recurring 5
times (1 previously known + 4 new), not 4 independent one-off gaps:

| Task | Triple | Post-crop transform(s) | New triple needed? |
|---|---|---|---|
| `1c786137` (ADR-0013, unreached) | `(T,F,F)` | `trim` | no (`select_tallest` exists) |
| `28bf18c6` | `(T,T,T)` | `hconcat_self` | no (`select_largest` exists) |
| `f25fbde4` | `(T,T,T)` | `upscale(2)` | no (`select_largest` exists) |
| `2013d3e2` | `(F,T,T)` | `lefthalf` then `tophalf` | yes |
| `7468f01a` | `(F,T,T)` | `vmirror` | yes |

The 5th, `a79310a0` (`objects(I,T,F,T)` → `first` → `move(DOWN)` →
`replace(8,2)`), doesn't even hit that wall: `move` (not `subgrid`) is the
very next call, and `move_selected` (ADR-0012, an `act_on_selection` action)
already does exactly `dsl.move`, doesn't end the episode, and its triple
(`T,F,T`) is already `select_largest_no_diag`'s (ADR-0013). This task is
reachable *today*, with **zero new primitives** - the audit's name-level
check missed it because it isn't checking against the deeper
selection-mechanism mapping, only raw primitive names.

**`ofcolor`'s cluster, re-examined:** none of its 5 tasks reduce this
cleanly. Per task:

- `7c008303`: `ofcolor(I,3)` picks a sub-region to `subgrid` into a *second*
  grid, then a *second* `ofcolor` call computes indices *on that sub-grid* -
  two `Indices` values in two different coordinate spaces, combined at the
  end via `fill` onto a *third*, differently-built grid (`compress`+
  `upscale` of a two-color `replace`d copy of `I`). No single selection slot
  can hold this.
- `9ecd008a`: `ofcolor(I,0)` computes indices against `I`, but the final
  `subgrid` call uses them against `vmirror(I)` - a *different* grid than
  the one the selection was computed on. `crop_to_selection` (this ADR)
  doesn't help: it only relaxes *when* the episode ends, not *which grid*
  the selection is checked against, and this repo's design already
  invalidates a selection across any successful ordinary transform
  precisely because a stale selection silently meaning something different
  after the grid changes is worse than requiring re-selection (ADR-0011).
- `a68b268e`: three separate `ofcolor` calls against three different
  quadrant sub-grids, `fill`ed onto a *fourth* quadrant. Needs at least
  three simultaneous selections plus multi-grid bookkeeping - nothing like
  this repo's one-grid, one-selection model.
- `a9f96cdd` / `d364b489`: a single `ofcolor` selection, reused across
  **four** `shift`-then-`fill` "stamp" calls (one per direction, each a
  different color), each stamping a shifted copy of the *original* selected
  cells onto the grid without disturbing it. This one is architecturally
  compatible with persistent selection (proven by this ADR's own
  `crop_to_selection`: an `act_on_selection` action never clears the
  selection, so reusing one selection across several act-on-selection steps
  already works) - but needs a genuinely new primitive, a directional
  "stamp a shifted copy of the selection at a fixed color" action, which
  doesn't exist yet. Curating bare `ofcolor` alone wouldn't unlock either
  task.

None of the 5 is a bare "add one primitive" fix. Recorded here rather than
re-litigated per-task in `docs/QUESTIONS.md` going forward.

## Decision

**Land two changes, generalizing past the specific `1c786137` gap ADR-0013
named:**

1. **`crop_to_selection`** - a new `act_on_selection` action, identical
   function to `commit_selection` (`dsl.subgrid(selected, grid)`), but a
   *different action name*. `ArcEnv.step`'s and `trainers/gp/fitness.py`'s
   episode-termination checks both key off the literal string
   `action_name in ("commit", "commit_selection")` - a differently-named
   action doing the same crop simply isn't in that set, so the episode
   carries on and a further ordinary transform can run on the cropped
   result. This is the generalized version of the fix ADR-0013's
   Alternatives considered sketched as a `1c786137`-specific
   "`commit_selection_trimmed`" - instead of one fused primitive per
   post-crop transform, one non-terminal crop primitive composes with every
   existing zero/one-arg transform already in the curated menu.
2. **`select_largest_multicolor`** - a fourth `objects(...)` connectivity
   variant, `(univalued=False, diagonal=True, without_bg=True)`: the first
   curated variant where an object can span more than one color. `argmax`
   by `size`, same compare function `select_largest` uses (both fixture
   tasks needing this triple have exactly one such object per grid, so the
   compare function never has to break a tie).

**Curated task count: 30 → 36** (`CURATED_TASK_IDS`, `arc_env/task_loader.py`):

| Task | Sequence | New mechanism used |
|---|---|---|
| `a79310a0` | `select_largest_no_diag → move_selected(DOWN) → replace(8,2)` | none (already-curated actions) |
| `1c786137` | `select_tallest → crop_to_selection → trim` | `crop_to_selection` |
| `28bf18c6` | `select_largest → crop_to_selection → hconcat_self` | `crop_to_selection` |
| `f25fbde4` | `select_largest → crop_to_selection → upscale(2)` | `crop_to_selection` |
| `2013d3e2` | `select_largest_multicolor → crop_to_selection → lefthalf → tophalf` | both |
| `7468f01a` | `select_largest_multicolor → crop_to_selection → vmirror` | both |

Every sequence was verified by direct replay against all train + test pairs
(both the bare-function path and the raw-args `actions.execute` path,
`tests/test_dsl_regression.py`'s existing two parametrized checks, which
pick these up automatically via `CURATED_TASK_IDS`) **and** by a dedicated
`ArcEnv`-level test (`tests/test_env.py`) proving `crop_to_selection`
specifically doesn't terminate the episode where `commit_selection` would.

The 5 `ofcolor`-flagged tasks are **not** curated this pass - see Context.

## Mechanism

No change to `execute`'s signature or control flow: `crop_to_selection` is
mechanically identical in shape to `commit_selection` (zero-arg
`act_on_selection`, same underlying `dsl.subgrid` call) and
`select_largest_multicolor` is mechanically identical to `select_largest`
(zero-arg `select`, `argmax` by `size`) - only the fixed `(univalued,
diagonal, without_bg)` triple passed to `dsl.objects` differs, the same
"straightforward additional action" extension ADR-0011's Alternatives
considered anticipated. The only place the *name* `crop_to_selection`
matters is the two termination checks (`arc_env/env.py`'s `step`,
`trainers/gp/fitness.py`'s `run_program`) - both already match by literal
name, not by `Action.kind`, so no code there needed to change; adding a
differently-named action was enough by construction.

`trainers/ppo/network.py`'s `IS_ACT_ON_SELECTION` mask (ADR-0008 amendment)
derives from `Action.kind == "act_on_selection"` generically, so
`crop_to_selection` is automatically included - masked out of the policy's
sampled distribution whenever nothing is selected, same as the other 5
`act_on_selection` actions, with no code change needed there either (only
the comments/tests naming "the 5" needed updating to "6").

## Alternatives considered

| Option | Why not |
|--------|---------|
| A fused `commit_selection_trimmed` action, scoped just to `1c786137` (ADR-0013's own sketch) | Would only land 1 of the 6 tasks this pass reaches. `crop_to_selection` is no more complex to implement and composes with *any* existing transform, not just `trim` - strictly more general for the same implementation cost. |
| Make `commit_selection` itself non-terminal, dropping `commit_selection` from the termination-check tuple entirely | Rejected: `1f85a75f`/`23b5c85d`/`1cf80156`/`be94b721`'s existing curated sequences all end on `commit_selection` specifically *because* it terminates there - repurposing it would require adding an explicit terminal step to all 4, a bigger and riskier change than adding one new action. |
| Chase `ofcolor` as a bare new `select` action anyway, accepting it wouldn't unlock its 5 flagged tasks on its own | Rejected: curating a primitive with no reachable fixture task is exactly the anti-pattern ADR-0013's own Alternatives considered flagged for `1c786137` - "a curated-task entry that can never be solved in practice... silently burns compute." `ofcolor` itself isn't curated as an action this pass (no verified fixture reaches it - see Context); revisit once the actual blocker (multi-selection, or a directional stamp primitive) has its own design. |
| Design and land the directional "stamp" act-on-selection primitive `a9f96cdd`/`d364b489` need, in this same pass | Two more real tasks, but a genuinely new primitive shape (not a variant of an existing one) deserves its own scoped design pass rather than being bundled under this ADR's "non-terminal crop" theme - same "don't conflate separable concerns in one ADR" discipline ADR-0013 applied to its own deferred `commit_selection_trimmed` idea. Flagged in Consequences as the next concrete candidate. |

## Consequences

- Action count grows 38 → 40 (`SELECT` gains `select_largest_multicolor`;
  `ACT_ON_SELECTION` gains `crop_to_selection`). `N_ACTIONS`/`MAX_ARITY` in
  `trainers/ppo/network.py` and `trainers/gp/genome.py` derive from
  `len(actions.ACTIONS)`/`actions.MAX_ARITY` already - no code change needed
  there (reconfirms, same note ADR-0011/0012/0013 each already made).
- `CURATED_TASK_IDS` grows 30 → 36 (15 same-shape + 21 variable-shape,
  `a79310a0` same-shape, the other 5 variable-shape) - compute cost scales
  linearly with curated task count (ADR-0008), an accepted, already-priced-in
  tradeoff.
- `commit_selection` and `crop_to_selection` are now two actions that do the
  *exact same thing* to the grid and differ only in termination behavior -
  a real but narrow duplication, deliberately chosen (see Alternatives) over
  either repurposing `commit_selection` (riskier) or a one-off fused
  primitive per post-crop transform (less general). Any future coverage
  audit should treat "select → crop → transform" and "select → crop → end
  episode" as the same reachability question with two different endings,
  not two different capabilities.
- This ADR's reachability-auditing method - checking the actual
  step-by-step episode model (termination included), not just whether a
  bare-function replay reproduces the expected output - is the same one
  ADR-0013 introduced for `1c786137`; this pass reconfirms it generalizes
  (it correctly predicted 4 more tasks hit the identical wall) and should be
  the default check for any future "one new primitive"-flagged cluster
  before committing to it, rather than trusting `scripts/
  audit_action_coverage.py`'s name-level classification at face value.
- **The `ofcolor` cluster is a genuinely different shape of gap** from
  `first`'s: not one shared, now-fixed mechanism issue, but each task
  needing its own additional capability (multi-selection across coordinate
  spaces, a selection checked against an unrelated grid, or a new
  directional stamp primitive). `docs/QUESTIONS.md` F11 is updated to
  record this as a negative result, not a to-do - re-attempting `ofcolor`
  as a single ADR would be the same mistake this ADR's Context section just
  finished correcting for.
- **Next concrete candidate**, if F11 continues: a directional "stamp a
  shifted copy of the selection onto the grid, with a fixed color, without
  clearing the original" `act_on_selection` primitive - unlocks `a9f96cdd`
  and `d364b489` on its own (2 tasks), is a real, well-defined addition (not
  a speculative one), and deserves its own scoped ADR per Alternatives
  considered above rather than being bundled here.
