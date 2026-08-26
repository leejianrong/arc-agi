# ADR-0002: Variable output-grid shape via a fixed-size scratch canvas + commit/crop action

- Status: Accepted
- Date: 2026-08-27
- Deciders: repo owner, via delegated subagent research (see `docs/research/arc-dsl-survey.md`)

## Context

ARC-AGI-1 tasks are not guaranteed to have output grids the same shape as
their input — the agent must sometimes also decide the output's dimensions.
This was originally flagged as a high-risk unknown (`docs/QUESTIONS.md` F2)
likely to swamp the first "does RL learn anything" milestone, with same-shape-only
scoping recommended as a way to dodge it. Research into `arc-dsl` (ADR-0001)
found this is a first-class, well-exercised pattern in that DSL: `canvas`,
`crop`, `subgrid`, `hconcat`/`vconcat`, `upscale`/`downscale`, `trim`, and
`compress` are used throughout the 400 real solver programs to build outputs
whose shape differs from the input.

## Decision

Give the RL/GP agent a fixed-size scratch canvas (30×30, ARC's max grid
dimension) to act on, plus an explicit "commit output" action that crops the
canvas to whatever region the agent has painted — mirroring `arc-dsl`'s own
`canvas` + `crop`/`subgrid` primitives. This keeps the action space fixed-arity
per step while still allowing any final output shape. Slice 1 (`SLICES.md` V1)
still targets a same-shape-only task subset as the first smoke test, with the
canvas/commit mechanism added in a later slice (V3) once the base env/trainer
loop is proven — not because shape-changing is a hard unknown anymore, but to
keep the riskiest-mechanism-first slice minimal.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Restrict the whole milestone to same-shape tasks | Was the original recommendation before research; no longer justified once `canvas`/`crop` showed up as a solved pattern in the DSL we're adopting anyway — needlessly narrows the task pool for no remaining technical reason. |
| Let the agent directly choose output dimensions as a separate discrete action before painting | Works but couples "decide size" and "decide content" as two different action types the policy must learn to sequence correctly; the scratch-canvas approach lets size just fall out of where the agent chose to paint, which is closer to how the DSL's own solvers work. |

## Consequences

- Slice 3 (`SLICES.md` V3) must add the canvas/commit action and a
  same-shape-vs-variable-shape task split to the env; this is new scope
  versus the original same-shape-only plan, but bounded and well-specified.
- The action space grows by exactly one action type (`commit`) plus the
  `canvas`/`crop` primitives already counted in ADR-0001's ~150-primitive
  catalog — no open-ended growth.
- The visualizer's replay renderer must handle a canvas larger than the
  committed output (showing the scratch area, then the cropped final grid)
  — a UI detail, not an architectural one.
