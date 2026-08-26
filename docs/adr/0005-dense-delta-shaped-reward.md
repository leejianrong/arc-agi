# ADR-0005: Dense, delta-based, non-background-normalized reward shaping

- Status: Accepted
- Date: 2026-08-27
- Deciders: repo owner, via delegated subagent research (see `docs/research/rl-evolutionary-survey.md`)

## Context

Reward design was originally an assumed default, not a fork, but the
RL/evolutionary research surfaced it as genuinely load-bearing: ARCLE (the
one existing purpose-built ARC RL paper) found sparse terminal-only reward
starves PPO of signal on its own — they needed human-demonstration behavior
cloning just to make training tractable. Two further pitfalls are documented
in the general RL literature and apply directly to grid-editing: reward
hacking via "vibrating in place" around a partially-correct state, and
absolute-pixel-match reward being biased toward a mostly-background grid,
which can trap a policy at a "paint nothing" local optimum.

## Decision

Reward at each step is:

```
reward_t = (similarity(grid_t, target) - similarity(grid_{t-1}, target))
           - step_cost
           + (terminal_bonus if exact_match else 0)
```

where `similarity` is measured over cells that actually need to change
between input and target (not raw whole-grid pixel match, to avoid the
background-bias trap), and `step_cost` is a small constant penalty per action
to discourage no-op/oscillating behavior.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Sparse terminal-only reward (exact match only) | ARCLE's own finding: this stalls PPO on ARC without a human-demonstration dataset to compensate, which we don't have. |
| Raw whole-grid pixel-match similarity (not normalized to changed cells) | Documented background-bias pitfall — a mostly-empty grid scores well for doing nothing, trapping the policy at a "paint nothing" optimum. |
| Absolute similarity reward with no step-cost/no-op penalty | Documented "vibrating in place" pitfall — the agent can find a locally-good state and oscillate without net progress, since absolute similarity doesn't distinguish progress from stagnation. |

## Consequences

- Reward computation needs to know which cells differ between the task's
  input and target ahead of time (cheap to precompute per episode from the
  training pair).
- This does not solve ARC's fundamental difficulty — ARCLE still needed
  demonstration data even with reasonable shaping — so PPO may still show
  limited or no learning on harder tasks in the scoped subset; this is a
  known, documented risk (see PLAN.md Open risks), not a shaping bug to chase.
- The same reward function is reused by genetic programming's fitness
  evaluation (ADR-0003) where useful, keeping one definition of "how close is
  this grid to correct" across both trainers.
