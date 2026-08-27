"""Unit test (SLICES.md V2): GAE advantage computation matches a
hand-computed value for a small fixed trajectory."""

import numpy as np
import pytest

from trainers.ppo.gae import compute_gae


def test_gae_matches_hand_computation_for_a_three_step_trajectory():
    # T=3, terminates at the last step (next_values[2] = 0.0).
    # Hand-computed (gamma=0.9, lam=0.8):
    #   delta_2 = 2.0 + 0.9*0.0 - 0.4 = 1.6            -> A_2 = 1.6
    #   delta_1 = 0.0 + 0.9*0.4 - 0.6 = -0.24           -> A_1 = -0.24 + 0.9*0.8*1.6 = 0.912
    #   delta_0 = 1.0 + 0.9*0.6 - 0.5 = 1.04            -> A_0 = 1.04 + 0.9*0.8*0.912 = 1.69664
    rewards = np.array([1.0, 0.0, 2.0])
    values = np.array([0.5, 0.6, 0.4])
    next_values = np.array([0.6, 0.4, 0.0])  # values[1], values[2], then 0.0 at termination
    dones = np.array([False, False, True])

    advantages, returns = compute_gae(rewards, values, next_values, dones, gamma=0.9, lam=0.8)

    np.testing.assert_allclose(advantages, [1.69664, 0.912, 1.6], rtol=1e-9)
    np.testing.assert_allclose(returns, advantages + values, rtol=1e-9)


def test_gae_does_not_propagate_advantage_across_an_episode_boundary():
    # Two 1-step episodes back to back: done at every step means each
    # step's advantage is just its own TD error, independent of the other.
    rewards = np.array([1.0, 5.0])
    values = np.array([0.0, 0.0])
    next_values = np.array([0.0, 0.0])
    dones = np.array([True, True])

    advantages, _ = compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95)

    np.testing.assert_allclose(advantages, [1.0, 5.0])


def test_gae_reduces_to_plain_td_error_when_lambda_is_zero():
    rewards = np.array([1.0, 2.0, 3.0])
    values = np.array([0.1, 0.2, 0.3])
    next_values = np.array([0.2, 0.3, 0.0])
    dones = np.array([False, False, True])

    advantages, _ = compute_gae(rewards, values, next_values, dones, gamma=0.9, lam=0.0)

    expected = rewards + 0.9 * next_values - values
    np.testing.assert_allclose(advantages, expected)
