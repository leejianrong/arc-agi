# ADR-0004: Gymnasium-style env + hand-rolled PPO, not Stable-Baselines3

- Status: Accepted
- Date: 2026-08-27
- Deciders: repo owner

## Context

The RL trainer needs an algorithm and a library boundary. `arc-dsl`'s action
space (ADR-0001) is "pick one of ~150 typed primitives + typed arguments,"
which is not a plain fixed-size `Discrete` or `Box` space that libraries like
Stable-Baselines3 (SB3) expect out of the box. The project also plans a
genetic-programming trainer (ADR-0003) that needs to reuse the same rollout
buffer/logging/env-stepping code, which off-the-shelf RL libraries aren't
designed to share with a non-gradient trainer.

## Decision

Expose the environment through a Gymnasium-style `step`/`reset`/`observation_space`/
`action_space` interface (for familiarity and compatibility with standard
tooling), but implement PPO as a small, hand-rolled training loop (rollout
collection, GAE advantage estimation, clipped surrogate objective) rather
than depending on Stable-Baselines3.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Stable-Baselines3 | Fastest to get a baseline running, but its policy/action-space abstractions assume `Discrete`/`Box`/`MultiDiscrete`; representing "primitive + typed args" cleanly would mean fighting the library's assumptions more than writing PPO directly. Also pulls in a heavier dependency tree for a CPU-only box. |
| RLlib / other full RL frameworks | Same fundamental mismatch as SB3, at even higher setup/dependency cost. |
| Skip Gymnasium's interface entirely, write a fully custom env API | Loses compatibility with the wider ecosystem's mental model (and any future desire to try an existing library on a simplified version of the env) for no real benefit over just implementing `step`/`reset`. |

## Consequences

- This ADR fixes the library/algorithm boundary only. What the network
  looks like, and whether training is per-task or shared across tasks, is a
  separate decision — see ADR-0008.
- ~150-300 lines of PPO plumbing (rollout buffer, GAE, clipped loss) is our
  code to maintain and test, not a dependency to trust.
- The action space can be represented naturally (e.g., a factored
  categorical-plus-arguments head) without contorting it into SB3's box.
- Adding genetic programming (ADR-0003) later can reuse the same env-stepping
  and logging code paths directly, since nothing about them is PPO-specific.
- We do not get SB3's built-in tensorboard integration, vectorized-env
  helpers, or battle-tested implementation details for free — a smoke test
  (PLAN.md Testing approach) that PPO can solve a trivial single task is the
  guard against a subtly broken hand-rolled implementation.
