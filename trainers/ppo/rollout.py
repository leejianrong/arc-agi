"""Rollout collection (ADR-0004): step the env with the current stochastic
policy for a fixed number of steps, auto-resetting into a new episode
whenever one ends, and build a GAE-ready buffer.

Episodes span PPO updates: `RolloutCollector` carries the current episode's
state across calls to `collect()`, so a call that ends mid-episode picks up
exactly where it left off next time - `collect()` always returns exactly
`n_steps` transitions.

Which task pair to reset into is entirely the caller's decision, injected
via `next_pair_fn` - keeps this module ignorant of `arc_env.task_loader`/
`arc_env.re_arc` specifics (SLICES.md V2 build plan step 3's train-pairs +
re-arc-generated-instances mix lives in `train.py`, not here) and makes
rollout collection unit-testable with a fixed fixture pair.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from arc_env.env import ArcEnv
from arc_env.task_loader import Pair
from trainers.ppo.network import MAX_ARITY, ActorCritic


@dataclass
class RolloutBuffer:
    obs: torch.Tensor  # (T, 30, 30) long
    primitive: torch.Tensor  # (T,) long
    args: torch.Tensor  # (T, MAX_ARITY) long
    log_prob: torch.Tensor  # (T,) float
    value: torch.Tensor  # (T,) float
    reward: np.ndarray  # (T,) float64
    terminated: np.ndarray  # (T,) bool
    truncated: np.ndarray  # (T,) bool
    next_value: np.ndarray  # (T,) float64 - precomputed GAE bootstrap (see trainers/ppo/gae.py)
    episode_returns: list = field(default_factory=list)  # total reward of each episode fully completed in this rollout
    episode_successes: list = field(default_factory=list)  # bool (terminated, i.e. exact match) per completed episode

    @property
    def n_steps(self) -> int:
        return self.reward.shape[0]

    def action_dict(self) -> dict:
        """`primitive`/`args` reshaped into the `{"primitive", "arg1", ...}`
        shape `ActorCritic.get_action_and_value(action=...)` and `ArcEnv.step`
        both expect."""

        d = {"primitive": self.primitive}
        for i in range(MAX_ARITY):
            d[f"arg{i + 1}"] = self.args[:, i]
        return d


class RolloutCollector:
    def __init__(self, env: ArcEnv, network: ActorCritic, next_pair_fn):
        self.env = env
        self.network = network
        self.next_pair_fn = next_pair_fn
        self._obs = None
        self._episode_reward = 0.0
        self._reset_episode()

    def _reset_episode(self) -> None:
        task_id, pair = self.next_pair_fn()
        obs, _ = self.env.reset(task_id=task_id, pair=pair)
        self._obs = obs
        self._episode_reward = 0.0

    def _value_of(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.network.get_value(torch.as_tensor(obs, dtype=torch.long).unsqueeze(0)).item())

    def collect(self, n_steps: int) -> RolloutBuffer:
        obs_buf, primitive_buf, args_buf = [], [], []
        log_prob_buf, value_buf, reward_buf = [], [], []
        terminated_buf, truncated_buf = [], []
        next_value_buf: list = []
        episode_returns, episode_successes = [], []

        for _ in range(n_steps):
            obs_t = torch.as_tensor(self._obs, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                sample = self.network.get_action_and_value(obs_t)

            primitive = int(sample.primitive.item())
            args = tuple(int(a) for a in sample.args[0].tolist())
            action = {"primitive": primitive, **{f"arg{i + 1}": args[i] for i in range(MAX_ARITY)}}

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            obs_buf.append(self._obs)
            primitive_buf.append(primitive)
            args_buf.append(args)
            log_prob_buf.append(float(sample.log_prob.item()))
            value_buf.append(float(sample.value.item()))
            reward_buf.append(reward)
            terminated_buf.append(terminated)
            truncated_buf.append(truncated)
            self._episode_reward += reward

            if terminated or truncated:
                next_value_buf.append(0.0 if terminated else self._value_of(next_obs))
                episode_returns.append(self._episode_reward)
                episode_successes.append(info["exact_match"])  # not `terminated` - V3's commit can end without matching
                self._reset_episode()
            else:
                next_value_buf.append(None)  # patched below, once value_buf[t + 1] exists
                self._obs = next_obs

        for t in range(n_steps - 1):
            if next_value_buf[t] is None:
                next_value_buf[t] = value_buf[t + 1]
        if next_value_buf and next_value_buf[-1] is None:
            next_value_buf[-1] = self._value_of(self._obs)  # bootstrap: episode continues past this buffer

        return RolloutBuffer(
            obs=torch.as_tensor(np.stack(obs_buf), dtype=torch.long),
            primitive=torch.tensor(primitive_buf, dtype=torch.long),
            args=torch.tensor(args_buf, dtype=torch.long),
            log_prob=torch.tensor(log_prob_buf, dtype=torch.float32),
            value=torch.tensor(value_buf, dtype=torch.float32),
            reward=np.array(reward_buf, dtype=np.float64),
            terminated=np.array(terminated_buf, dtype=bool),
            truncated=np.array(truncated_buf, dtype=bool),
            next_value=np.array(next_value_buf, dtype=np.float64),
            episode_returns=episode_returns,
            episode_successes=episode_successes,
        )


def evaluate_episode(env: ArcEnv, network: ActorCritic, task_id: str, pair: Pair) -> dict:
    """Runs one episode with the current *greedy* (argmax) policy - used for
    periodic eval-episode logging (SLICES.md V2 build plan step 4), not
    training. Returns the full step-by-step trace for `EpisodeWriter`."""

    obs, _ = env.reset(task_id=task_id, pair=pair)
    steps = []
    terminated = truncated = False
    exact_match = False
    total_reward = 0.0

    while not (terminated or truncated):
        grid_before = env.get_grid()
        obs_t = torch.as_tensor(obs, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            primitive, args = network.get_greedy_action(obs_t)
        action = {"primitive": int(primitive.item()), **{
            f"arg{i + 1}": int(args[0, i].item()) for i in range(MAX_ARITY)
        }}
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        exact_match = info["exact_match"]
        steps.append({
            "grid_before": grid_before,
            "action_name": info["action_name"],
            "action_args": info["action_args"],
            "grid_after": env.get_grid(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "valid_action": info["valid_action"],
            "exact_match": exact_match,
            "selected": info["selected"],
        })

    return {"steps": steps, "success": exact_match, "total_reward": total_reward}
