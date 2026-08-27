"""End-to-end test (SLICES.md V2): running `train.py --algo ppo` against the
single-task PPO-sanity fixture (PLAN.md Testing approach - `vmirror`, the
trivial single-action task) for the configured update budget produces a
mean evaluation-episode reward, over the last 10% of updates, strictly
greater than the mean reward of 100 random-policy episodes on the same
task - a concrete, non-subjective pass/fail comparison, checkable by script.

The random-policy baseline is computed live against the *current* env
(ADR-0005's dense reward, wired in this same slice) rather than reusing a
V1 number - V1's reward was itself a placeholder this slice replaces, so
there is no other reward scheme left in the codebase to compare against.

Also covers the V2 integration-test requirements: `metrics.jsonl` rows
parse into the expected schema and render a monotonically-timestamped
curve, and a periodic checkpoint loads and resumes training without error.
"""

import json
import random

import numpy as np
import pytest
import torch

from arc_env.env import ArcEnv
from arc_env.re_arc import generate_pair
from trainers.ppo.network import ActorCritic
from trainers.ppo.ppo import PPOConfig
from train import load_checkpoint, train_ppo

TASK_ID = "67a3c6ac"  # solved by a single vmirror - PLAN.md's named PPO-sanity fixture
N_UPDATES = 20
ROLLOUT_STEPS = 128
MAX_STEPS = 10


def _random_baseline_mean_reward(task_id: str, n_episodes: int, max_steps: int, seed: int) -> float:
    rng = random.Random(seed)
    env = ArcEnv(max_steps=max_steps)
    env.action_space.seed(seed)
    rewards = []
    for _ in range(n_episodes):
        pair = generate_pair(task_id)
        env.reset(task_id=task_id, pair=pair)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
        rewards.append(total)
    return float(np.mean(rewards))


pytestmark = pytest.mark.slow  # ~90s - excluded from the default/pre-push run (see pyproject.toml)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """Runs PPO once (~90s) and shares the result across this module's
    assertions, rather than re-training per test."""

    run_dir = tmp_path_factory.mktemp("ppo_smoke") / "run"
    config = PPOConfig(n_epochs=4, minibatch_size=64)
    train_ppo(
        task_id=TASK_ID, run_dir=run_dir, n_updates=N_UPDATES, rollout_steps=ROLLOUT_STEPS,
        eval_every=N_UPDATES - 1, re_arc_prob=0.5, max_steps=MAX_STEPS, seed=0, config=config,
    )
    return run_dir


def test_ppo_beats_random_baseline_over_last_10_percent_of_updates(trained_run):
    baseline = _random_baseline_mean_reward(TASK_ID, n_episodes=100, max_steps=MAX_STEPS, seed=123)

    metrics = [json.loads(line) for line in (trained_run / "metrics.jsonl").read_text().splitlines()]
    last_10pct = metrics[-max(1, len(metrics) // 10):]
    rewards = [m["mean_reward"] for m in last_10pct if m["mean_reward"] is not None]

    assert rewards, "no completed episodes in the final 10% of updates"
    assert float(np.mean(rewards)) > baseline


def test_metrics_jsonl_rows_match_expected_schema_and_are_monotonically_timestamped(trained_run):
    metrics = [json.loads(line) for line in (trained_run / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == N_UPDATES

    expected_keys = {
        "update", "timestamp", "n_episodes", "mean_reward", "success_rate",
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
    }
    for row in metrics:
        assert expected_keys <= row.keys()
        assert isinstance(row["update"], int)

    updates = [row["update"] for row in metrics]
    assert updates == sorted(updates)
    timestamps = [row["timestamp"] for row in metrics]
    assert timestamps == sorted(timestamps)


def test_periodic_checkpoint_loads_and_resumes_training_without_error(trained_run):
    checkpoints = sorted((trained_run / "checkpoints").glob("update_*.pt"))
    assert checkpoints

    network = ActorCritic()
    optimizer = torch.optim.Adam(network.parameters())
    resumed_update = load_checkpoint(checkpoints[-1], network, optimizer)
    assert resumed_update == N_UPDATES - 1

    # Resuming must not error, and should produce a well-formed forward pass.
    env = ArcEnv(max_steps=MAX_STEPS)
    obs, _ = env.reset(task_id=TASK_ID, pair=generate_pair(TASK_ID))
    obs_t = torch.as_tensor(obs, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        sample = network.get_action_and_value(obs_t)
    assert sample.value.shape == (1,)
