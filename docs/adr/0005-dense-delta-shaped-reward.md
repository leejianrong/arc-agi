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

## Amendment (2026-08-29): shape-distance gradient for variable-shape pairs

`similarity`'s variable-shape fallback (`diff_mask is None`, ADR-0002) scored
a flat `0.0` for any wrong-shape grid until it happened to land on the
target's exact shape - a genuine dead zone, not a tuning problem: neither
trainer could reliably solve a task needing `commit`'s exact 4-argument
(row, col, height, width) combination, because there was no reward signal at
all to distinguish a close-but-wrong crop from a wildly wrong one. Both
trainers share `compute_reward`/`similarity`, so this blocked both equally.

`similarity` now blends in `SHAPE_MATCH_CREDIT` (0.3) worth of score for how
close the current grid's shape is to the target's (normalized Manhattan
distance between shapes), continuous with the content-match score exactly at
the shape boundary, before falling back to the original per-cell match once
shape is achieved. This preserves the ADR's dense/delta-based/non-background-
normalized design - it only fills in what `similarity` returns during the
"wrong shape" case that was previously undefined-in-practice (always 0.0).
The same-shape (`diff_mask` given) path is untouched: a shape mismatch there
means an already-same-shape task's episode played a shape-changing action
unnecessarily, which isn't the scenario this fixes.

Verified against the `d10ecb37` fixture task (ADR-0002's `commit`-only
solver): see `tests/test_reward.py`'s shape-gradient tests for the
delta/monotonicity/continuity properties, and `tests/test_train_gp.py` for
end-to-end confirmation that a trainer solves `d10ecb37` within a bounded
budget.

## Amendment (2026-09-05): invalid-action penalty no longer stacks with step cost

`compute_reward` used to apply `step_cost` unconditionally and then subtract
`INVALID_ACTION_PENALTY` on top of it for an invalid action, so an invalid
action actually cost `step_cost + INVALID_ACTION_PENALTY` (0.03) — strictly
more than a safe, valid no-op like `identity`, which costs just `step_cost`
(0.01). That extra stacking wasn't an intentional part of the design: the
ADR's decision only ever specified one constant per-step penalty plus a
separate invalid-action penalty, not that the two should compound. It
mattered specifically for PPO, which scores every single step's credit
assignment (unlike GP, which only scores a whole finished program's outcome)
— the stacked cost pushed PPO to treat invalid actions as needlessly worse
than they already are from having no similarity-delta upside, on top of the
step cost every other action also pays.

`compute_reward` now charges exactly one of the two constants per step:
`STEP_COST` for a valid action (unchanged), or `INVALID_ACTION_PENALTY` in
its place (not in addition) for an invalid one. A valid action that makes no
progress still costs exactly `STEP_COST`, as before; an invalid action now
costs exactly `INVALID_ACTION_PENALTY` (0.02) instead of 0.03. This only
changes the constant-penalty term — the similarity delta and
`TERMINAL_BONUS` are untouched, and both trainers still share the same
`compute_reward` function, so the fix applies uniformly to PPO and GP.

Checked whether `exact_match` can ever coincide with `valid_action=False`
(`arc_env/env.py`'s `step()`): an invalid action always leaves the grid
unchanged (`arc_env/actions.py`'s `execute()`), so `exact_match=True` on an
invalid-action step is only reachable if the grid already matched the target
*before* that step — which itself would have already terminated the episode
on a prior step, except for the degenerate case of a train pair whose input
already equals its output. In that edge case the penalty and
`TERMINAL_BONUS` simply compose additively (`-INVALID_ACTION_PENALTY +
TERMINAL_BONUS`), same as they always have — no special-casing needed; see
`tests/test_reward.py`'s coverage of this composition.
