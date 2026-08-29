"""The PPO update step (ADR-0004): clipped surrogate objective, over
`n_epochs` passes of shuffled minibatches from one rollout buffer."""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from trainers.ppo.gae import compute_gae
from trainers.ppo.network import ActorCritic
from trainers.ppo.rollout import RolloutBuffer


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    n_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


def ppo_update(network: ActorCritic, optimizer: torch.optim.Optimizer, buffer: RolloutBuffer, config: PPOConfig) -> dict:
    dones = buffer.terminated | buffer.truncated
    advantages, returns = compute_gae(
        buffer.reward, buffer.value.detach().numpy(), buffer.next_value, dones, config.gamma, config.lam
    )
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
    returns_t = torch.as_tensor(returns, dtype=torch.float32)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    old_log_prob = buffer.log_prob
    action = buffer.action_dict()

    n = buffer.n_steps
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}

    for _ in range(config.n_epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, config.minibatch_size):
            mb_idx = torch.as_tensor(perm[start:start + config.minibatch_size], dtype=torch.long)

            mb_obs = buffer.obs[mb_idx]
            mb_action = {k: v[mb_idx] for k, v in action.items()}
            sample = network.get_action_and_value(mb_obs, action=mb_action)

            ratio = torch.exp(sample.log_prob - old_log_prob[mb_idx])
            mb_adv = advantages_t[mb_idx]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (sample.value - returns_t[mb_idx]).pow(2).mean()
            entropy = sample.entropy.mean()

            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_prob[mb_idx] - sample.log_prob).mean().item()
                clip_frac = ((ratio - 1.0).abs() > config.clip_eps).float().mean().item()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_frac"].append(clip_frac)

    return {k: float(np.mean(v)) for k, v in stats.items()}
