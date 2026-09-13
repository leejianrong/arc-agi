# ADR-0023: Bucket C is 3 more derived actions, not a new mechanism

- Status: Accepted
- Date: 2026-09-14
- Deciders: repo owner, via conversation 2026-09-14

## Context

F13 Stage 2's last deferred item ("Bucket C") was 4 tasks - `7c008303`,
`a68b268e` (originally named in F11), `928ad970`, `017c7c7b` (found
independently by F13 Stage 1) - all filed as needing "the pre-crop original
grid held alongside a derived crop," a capability distinct from and heavier
than ADR-0020's region-scoped dual selection, explicitly flagged (in F11,
F13, F14, and ADR-0020's own text) as a decision of the same weight as the
original multi-selection/cross-grid mechanism call.

Per the process that decision itself was reached by, this pass first ran a
broader audit before designing anything.

### The audit, and the reframing it forced

`scripts/audit_action_coverage.py` against the current 57-curated baseline
(343 uncurated) found the same buckets as before, unchanged in kind. The
real work was semantic: for every non-excluded uncurated task (a name-level
AST walk of `arc_env/actions.py` plus a check of `solvers.py` for whether
the raw input-grid parameter is referenced more than once), check whether
that "more than one grid reference" is a genuine cross-step dependency or
just multiple reads inside what could be one atomic function body.

`arc_env/actions.py`'s `"transform"`-kind actions are `fn(grid, *args) ->
Grid` - ordinary, atomic Python functions. Nothing about that signature
limits `fn` to referencing `grid` once; it can compute a crop-derived value
*and* a separately-derived value from the same grid and combine them in one
step, with zero exposed intermediate state. F11/F13/F14's framing implicitly
assumed the *solver's own* multi-step decomposition was the only way to
reach these tasks - the same assumption ADR-0021 already caught and
corrected once for `007bbfb7`/`80af3007`/`8f2ea7aa` (filed as needing "hold
two live grid states," turned out to be a pure function of the input
alone). This pass applies that same correction to Bucket C.

None of the 4 solvers contains an actual agent-level, mid-episode
instance-dependent *choice* requiring cross-step state - each is a fixed
(possibly internally-branching, e.g. `017c7c7b`'s `equality`/`branch`)
deterministic pipeline. An internal Python `if` inside one derived action's
`fn` is not agent-visible state; it's just code.

Broadening the search past the original 4 turned up 2 more tasks sharing
the same previously-miscategorized shape: `eb281b96` and `c9f8e694`.

### Verifying which of the 6 actually generalize

Every prior derived action (`canvas_mostcolor`, `swap_two_least_colors`,
the self-concat family, ADR-0021's `fractal_expand_cellwise`, ...) was
verified against the real train/test pairs only. This pass added a
stronger check, since the whole point of a *reusable* action (vs. a
one-off fixture) is that it also holds up on `re-arc`-generated practice
instances of the same task: for each candidate, a generalized design (agent-
chosen `COLOR_ARG`s in place of the solver's literal color constants,
everything else either structural or derived via existing primitives like
`mostcolor`/`leastcolor`) was brute-force-checked against 30 fresh `re-arc`
instances per task (searching all plausible arg assignments, not just the
one heuristic guess - the design is what's being tested, not any one
color-picking strategy).

Result: a real split, not a uniform win.

| Task | Real pairs | re-arc(30), brute-force over args |
|------|-----------|-------------------------------------|
| `928ad970` | exact | **30/30** |
| `a68b268e` | exact | **30/30** |
| `eb281b96` | exact | **30/30** |
| `7c008303` | exact | 1/30 |
| `c9f8e694` | exact | 6/30 |
| `017c7c7b` | exact | 1/30 |

`7c008303`, `c9f8e694`, and `017c7c7b` all reproduce their own task's real
json exactly but collapse on `re-arc`'s broader instance space: each
solver's literal pipeline (a fixed compress-then-upscale-by-3
reconstruction; a fixed first-column crop; a fixed 2-block repeat) turns
out to fit only the specific dimensions/structure of the one fixed json it
was written against, not the more general pattern `re-arc`'s own generator
explores for that task (a variable embedding offset; a variable repeating
period). Reproducing them properly would need real new detection logic
(e.g. locating a divider's row/col structurally), not a parameterization
swap - the same order of cost as a new primitive, for one task each.
Confirmed with the user: not worth building on spec for 1-task payoffs each.

## Decision

**Three new zero/low-arity `"transform"`-kind actions, no new mechanism, no
new `Action.kind`** - verified bare-`dsl`, then through the real
`arc_env.actions.execute()` → `ArcEnv.step()` path (a temporary env
instance stepped through every train/test pair of all 3 tasks, confirming
`info["exact_match"]` on the single step, not just a bare-function replay):

```python
def _fill_inbox_by_dot_color(grid, dot_color: int) -> Grid:
    dots = dsl.ofcolor(grid, dot_color)
    box = dsl.subgrid(dots, grid)
    fill_color = dsl.leastcolor(dsl.trim(box))
    return dsl.fill(grid, fill_color, dsl.inbox(dots))

def _fill_quadrant_from_colors(grid, color_ul: int, color_ur: int, color_ll: int) -> Grid:
    top, bottom = dsl.tophalf(grid), dsl.bottomhalf(grid)
    ul, ur = dsl.lefthalf(top), dsl.righthalf(top)
    ll, lr = dsl.lefthalf(bottom), dsl.righthalf(bottom)
    result = lr
    for color, quad in ((color_ll, ll), (color_ur, ur), (color_ul, ul)):
        result = dsl.fill(result, color, dsl.ofcolor(quad, color))
    return result

def _repeat_mirror_tile(grid) -> Grid:
    go = dsl.vconcat(grid, dsl.hmirror(grid[:-1]))
    return dsl.vconcat(go, dsl.hmirror(go[:-1]))
```

- `fill_inbox_by_dot_color(dot_color)` - one `COLOR_ARG`. Unlocks `928ad970`
  (dot color is task-fixed at `5` in the real json but `re-arc`-randomized
  across instances; verified 30/30 with an agent-chosen arg instead).
- `fill_quadrant_from_colors(color_ul, color_ur, color_ll)` - three
  `COLOR_ARG`s (arity 3, `MAX_ARITY` unaffected). Unlocks `a68b268e`: reads
  each of 3 quadrants' own color independently and fills the found indices
  onto the 4th (bottom-right) quadrant, sequentially. Verified 30/30.
- `repeat_mirror_tile()` - zero-arg, fully derived (no agent-chosen args at
  all, matching `swap_two_least_colors`'s precedent). Unlocks `eb281b96`:
  `vconcat`s the grid with its own hmirror (minus the shared edge row)
  twice. Verified 30/30, no generalization risk at all - the literal
  solver already holds up structurally.

All 3 are single-step episodes: the action alone produces the exact target
grid (`info["exact_match"] = True` immediately, `ArcEnv.step`'s existing
`exact_match = self._grid == self._target` check, no `commit` needed).

**`7c008303`, `c9f8e694`, `017c7c7b`: no-go**, same reasoning applied
uniformly across all three - narrower than `re-arc`'s own generative concept
of each task, would need new structural-detection capability (not a
mechanism, but a nontrivial one-off algorithm) for a 1-task-each payoff.
Consistent with the `select_tallest`/`1c786137` and `9ecd008a` no-go
precedents: a special case dressed up as a solution isn't one.

## Alternatives considered

| Option | Why not chosen |
|--------|-----------------|
| A third-generation selection mechanism ("hold the pre-crop original alongside a derived crop") as originally framed | The audit found no task in this cluster actually needs it - every genuine win is reachable as an ordinary derived action. Building the heavier mechanism anyway would be unmotivated architecture risk for zero additional task coverage. |
| Invest in structural-detection logic (divider/offset/period finding) to also land `7c008303`/`c9f8e694`/`017c7c7b` | Confirmed with the user as not worth it: real new-primitive-class effort for 1 task each, versus the 3 free/cheap wins already banked. Left as a closed no-go, not a future thread. |
| Bundle the 3 winning tasks' literal solver formulas as-is (hardcoded colors) instead of generalizing to agent-chosen `COLOR_ARG`s | Would still pass the curation gate (real json only) but be far weaker for `re-arc`-augmented PPO/GP training, the same concern that motivated `select_leastcolor`/`canvas_mostcolor`'s "derived, not hardcoded" precedent. The agent-chosen-arg version costs nothing extra and was verified to generalize; there's no reason to ship the weaker one. |

## Consequences

- Unlocks `928ad970`, `a68b268e`, `eb281b96` - 3 tasks, 3 new actions, all
  `"transform"`-kind, `MAX_ARITY` unchanged (stays 4). No changes to
  `arc_env/env.py`'s selection state, `episode_log.py`'s schema,
  `trainers/gp/fitness.py`/`replay.py`, `trainers/ppo/warm_start.py`, or
  `viz/frontend`'s selection overlay - unlike ADR-0020, this pass touches
  only `arc_env/actions.py`.
- Action count: 55 → 58. Curated tasks: 57 → 60 (need per-task same/
  variable-shape recount at implementation time: `928ad970` same-shape,
  `a68b268e` variable-shape - shrinks to one quadrant -, `eb281b96`
  variable-shape - triples height).
- `7c008303`, `c9f8e694`, `017c7c7b` remain uncurated, closed as a reasoned
  no-go (not a silently dropped or still-open thread) - `docs/questions/
  f11-task-coverage.md`, `f13-interactive-editor.md`, and
  `f14-multi-selection-mechanism.md` all update to point here rather than
  carrying this forward as open.
- No open items remain from F13 Stage 2's backlog; F13 as a whole is done.
