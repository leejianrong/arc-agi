"""GAE advantage estimation (ADR-0004).

Standard bootstrapped-TD-lambda GAE, with one deliberate simplification in
the interface: rather than take `terminated`/`truncated` and reach into the
env itself to decide how to bootstrap a truncated episode, the caller
(`trainers/ppo/rollout.py`) precomputes `next_values[t]` for every step -
0.0 at a true termination, `V(final_obs)` at a time-limit truncation,
`values[t + 1]` for an ongoing episode, and an external bootstrap value for
the last step in the buffer if it's still ongoing. That keeps this function
pure array math, easy to hand-verify (see `tests/test_gae.py`).

`dones[t]` marks an episode boundary (terminated OR truncated) - it's only
used to stop advantage propagating backward across episodes; the reward
signal itself is already fully captured via `next_values`.
"""

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple:
    """Returns `(advantages, returns)`, both shape `(T,)`."""

    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        cont = 0.0 if dones[t] else 1.0
        gae = delta + gamma * lam * cont * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns
