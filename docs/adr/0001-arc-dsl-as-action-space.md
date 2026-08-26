# ADR-0001: Adopt Hodel's `arc-dsl` as the action-space DSL; discard the `arc-ngps` scaffold

- Status: Accepted
- Date: 2026-08-27
- Deciders: repo owner, via delegated subagent research (see `docs/research/arc-dsl-survey.md`)

## Context

The revamp needs a discrete, typed action space an RL policy can choose from
one step at a time, and that a genetic-programming search can compose into
whole programs (see ADR-0003). Two candidates existed: extend the project's
own half-built scaffold at `research/arc-ngps/src/arc_ngps/{dsl,executor}`
(6 AST node types, 3 executor primitives, no shape-changing capability, an
explicitly unfinished `Compose` node), or adopt Michael Hodel's `arc-dsl`
(160 pure-function primitives, one solver program per ARC-AGI-1 training
task, MIT licensed) and his companion `re-arc` (400 procedural task-instance
generators, also MIT licensed).

## Decision

Vendor `arc-dsl`'s `dsl.py`, `arc_types.py`, and `constants.py` under
`third_party/arc-dsl/`, and `re-arc` under `third_party/re-arc/`. Use
`arc-dsl`'s primitives directly as the RL/GP action space and executor — no
separate AST/executor layer of our own. Discard `research/arc-ngps`'s
`dsl/` and `executor/` modules entirely rather than adapting them. Exclude
higher-order primitives (`compose`, `chain`, `fork`, `rbind`, `lbind`,
`power`) from the initial action space, since they build closures rather
than directly transforming a grid.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Extend `arc-ngps`'s own DSL/executor | It is an early, ~1-week-scale sketch of the same idea Hodel spent months on — 3 executor primitives vs. ~30 object/geometry/canvas primitives covering the same ground and more, and no shape-changing primitives at all. Extending it means re-deriving work that already exists, tested, for free. |
| Hybrid (arc-ngps's typed AST wrapping Hodel's primitives) | Adds a layer of indirection with no payoff — Hodel's functions are already pure and directly callable; an AST wrapper only pays off if we needed program serialization beyond what a flat list of `(primitive, args)` steps already gives a trajectory log. |
| Write our own DSL from scratch | Would take real time to reach even a fraction of the coverage `arc-dsl` already has validated against all 400 ARC-1 training tasks, for no benefit over vendoring MIT-licensed, working code. |

## Consequences

- The RL action space is exactly "pick one of ~150 primitives (excluding
  higher-order ones) + typed arguments," directly steppable and directly
  the terminal set for GP program ASTs (ADR-0003) — one executor serves
  both trainers.
- `arc-dsl`'s 400 solver programs become a free regression-test fixture:
  running them through our env's executor must reproduce the expected
  outputs exactly (see PLAN.md Testing approach).
- We inherit `arc-dsl`'s frozen-upstream status (no longer accepting PRs);
  a community fork (`arc-dsl-2`) exists for ARC-AGI-2 work but isn't needed
  since this project's scope is ARC-AGI-1 only.
- `research/arc-ngps`'s DSL/executor code becomes dead weight in the repo.
  It is not deleted by this ADR — see PLAN.md Scope for its disposition —
  but it is no longer on the path to the shipped agent.
