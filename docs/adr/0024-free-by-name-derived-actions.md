# ADR-0024: 4 more derived actions from the `free_by_name` re-check bucket

- Status: Accepted
- Date: 2026-09-14
- Deciders: repo owner, via conversation 2026-09-14

## Context

`scripts/audit_action_coverage.py`'s own docstring flags `free_by_name` as
"a re-check candidate, not a guarantee" - every primitive a solver calls
already appears somewhere in `arc_env/actions.py` by name, but that doesn't
mean the exact curated parameterization matches. Past passes have found
both directions: `a79310a0`/`c909285e`/`007bbfb7` turned out to be free
wins nobody had checked carefully; most of `9ecd008a`'s cluster turned out
to need capability this project doesn't have.

Following ADR-0023, re-running the audit against the post-ADR-0023
baseline (60 curated, 340 remaining) surfaced 20 `free_by_name` tasks
worth re-checking - the cheapest possible next tier, since a positive
result needs no new mechanism and sometimes no new action at all.

Read through the 20 by hand; 7 shared one of two "fixed geometric tiling"
shapes, the same "derived, not drawn from a solver 1:1" family
`hconcat_self`/`fractal_expand_cellwise`/etc. already established. Two
more candidates checked and found *not* to generalize (`0520fde7`: 1/30
against fresh `re-arc` instances; `a699fb00`: 5/30) - same disposition as
ADR-0023's no-gos, not pursued further here. The remaining 11
(`11852cab`, `3618c87e`, `67385a82`, `aedd82e4`, `e3497940`, `e8593010`,
`e98196ab`, `47c1f68c`, `c3e719e8`-adjacent color/size-filter object
tasks) need real object-selection logic beyond a bare geometric formula -
left uncurated, candidates for a future pass, not attempted here.

## Decision

**4 new zero-arg `"transform"`-kind actions**, verified bare-`dsl`, then
through the real `arc_env.actions.execute()` → `ArcEnv.step()` path
against every train/test pair, then brute-force-generalization-checked
30/30 against fresh `re-arc` instances for every task (the ADR-0023 bar):

```python
def _quad_rotate_tile(grid: Grid) -> Grid:
    r90, r180, r270 = dsl.rot90(grid), dsl.rot180(grid), dsl.rot270(grid)
    return dsl.vconcat(dsl.hconcat(grid, r90), dsl.hconcat(r270, r180))

def _quad_mirror_tile(grid: Grid) -> Grid:
    top = dsl.hconcat(grid, dsl.vmirror(grid))
    return dsl.vconcat(top, dsl.hmirror(top))

def _stack3_vmirror_tile(grid: Grid) -> Grid:
    left = dsl.hconcat(dsl.vmirror(grid), grid)
    stacked = dsl.vconcat(left, dsl.hmirror(left))
    stacked = dsl.vconcat(stacked, left)
    return dsl.hmirror(stacked)

def _left_third(grid: Grid) -> Grid:
    return dsl.first(dsl.hsplit(grid, 3))
```

- `quad_rotate_tile()` - 2x2 tiling of `grid`, `rot90(grid)`, `rot270(grid)`,
  `rot180(grid)`. Unlocks `46442a0e`, `7fe24cdd` (identical solvers).
- `quad_mirror_tile()` - `hconcat_self_vmirror` (already exists as its own
  action) followed by stacking that block with its own `hmirror`. Unlocks
  `3af2c5a8`, `62c24649`, `67e8384a` (identical solvers).
- `stack3_vmirror_tile()` - a related but distinct tiling (concat order and
  stack count differ enough that it isn't reducible to
  `quad_mirror_tile`, confirmed by direct comparison). Unlocks `8d5021e8`.
- `left_third()` - this is *already* `arc_env/actions.py`'s private
  `_left_third` helper (added by ADR-0022 for the region-scoping menu),
  just never exposed as a standalone transform action in its own right.
  Zero new logic - literally the existing helper, registered as its own
  `Action`. Unlocks `2dee498d`.

All 7 are variable-shape (every dimension changes).

## Alternatives considered

| Option | Why not |
|--------|---------|
| One combined "geometric tile" action taking a tiling-pattern arg | The 3 shapes (`quad_rotate_tile`/`quad_mirror_tile`/`stack3_vmirror_tile`) aren't parameterizable by a single small arg without either losing `MAX_ARITY` headroom or building a mini-DSL of tiling patterns - not worth it for 6 tasks split 2/3/1. Three narrow zero-arg actions match this project's existing convention (the 4 self-concat actions are already this granular). |
| Chase `0520fde7`/`a699fb00` further (brute-force color search, structural re-derivation) | Same 1-task-each economics ADR-0023 already declined for `7c008303`/`c9f8e694`/`017c7c7b` - both fail generalization for structural reasons a color-arg swap can't fix. Left uncurated. |

## Consequences

- Unlocks `46442a0e`, `7fe24cdd`, `3af2c5a8`, `62c24649`, `67e8384a`,
  `8d5021e8`, `2dee498d` - 7 tasks, 4 new actions, all `"transform"`-kind,
  zero-arg. `MAX_ARITY` unaffected (all arity 0). No mechanism, selection,
  episode-log, or visualizer changes.
- Action count: 58 → 62. Curated tasks: 60 → 67 (all 7 new ones
  variable-shape: 36 → 43 variable-shape, 24 same-shape unchanged).
- `0520fde7`, `a699fb00` checked and left uncurated (same no-go reasoning
  as ADR-0023's cluster). The other 11 `free_by_name` tasks are unexamined
  past a first read - candidates for a future pass.
