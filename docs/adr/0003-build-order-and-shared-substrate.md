# ADR-0003: RL first, genetic programming as the evolutionary fast-follow, one shared substrate

- Status: Accepted
- Date: 2026-08-27
- Deciders: repo owner (build order), via delegated subagent research (evolutionary-track choice; see `docs/research/rl-evolutionary-survey.md`)

## Context

The user asked for "RL or evolutionary algorithms," and confirmed building RL
first with evolutionary as a fast-follow (`docs/QUESTIONS.md` F3). Research
into evolutionary options for ARC found genetic programming over a DSL
(population, crossover, mutation of program ASTs) is essentially unbenchmarked
in the literature — a real gap, not a dead end — while neuroevolution
(CMA-ES/ES over a policy network's weights) has no ARC-specific results at
all and is more expensive per fitness evaluation (a full episode of forward
passes vs. one program execution) on this project's CPU-only hardware
(16 cores, no GPU).

## Decision

Build the RL trainer (PPO, ADR-0004) first. Build genetic programming over
the same `arc-dsl` action space (ADR-0001) as the evolutionary fast-follow,
not neuroevolution. Both trainers share one Gymnasium-style environment, one
DSL/executor, and one trajectory log format (ADR-0006), so that:
- a GP-found solving program's execution trace can be logged and replayed
  identically to an RL episode in the visualizer, and
- GP-found programs are a documented future option as behavior-cloning
  demonstrations to warm-start the RL policy (out of scope for this
  milestone's slices, but the shared trajectory format is what makes it
  possible later without rework).

## Alternatives considered

| Option | Why not |
|--------|---------|
| Neuroevolution (CMA-ES) instead of genetic programming | CPU-viable but costs a full episode per fitness evaluation vs. GP's microsecond-scale program execution; no ARC-specific prior art to calibrate against; less direct a fit to the DSL/action-space design already adopted. |
| Build RL and evolutionary as fully separate tracks with no shared substrate | Loses the BC-warm-start interoperation option for free and risks two divergent action-space representations; sharing the env/DSL/log-format costs nothing extra since both need those anyway. |
| Evolutionary first, RL as fast-follow | User's explicit call was RL first (clearer dense-reward story, and PPO is a well-trodden path even though ARC-specific results (ARCLE) show it's hard); no research finding contradicted this. |

## Consequences

- One environment and one DSL executor must be built to serve both trainers
  from day one, even though GP's slice (`SLICES.md` V4) comes later — this is
  a small amount of extra care in the env's interface (e.g., don't bake in
  PPO-only assumptions like a fixed episode length if GP wants to run a whole
  program in one shot) rather than extra code.
- No existing benchmark number exists for GP-over-DSL on ARC, so V4's success
  criterion has to be self-referential (finds programs that solve more of the
  task subset than a random baseline) rather than compared to a published
  number.
- Neuroevolution is deliberately not built this milestone; if GP
  underperforms badly, re-opening the neuroevolution option is a future ADR,
  not a silent scope change.
