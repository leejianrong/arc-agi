"""GP-to-PPO behavior-cloning warm-start (ADR-0009): loads a same-task GP
run's best-found-program trajectory and runs a one-time supervised
pretraining phase against the policy's factored action heads, before the
normal PPO rollout/update loop starts unmodified. The value head is left
untouched - a demonstration trace has no logged return/advantage to
regress a critic against.

Reuses `viz.backend.server.read_episode` to parse the GP run's
`episodes/best-program.jsonl` (ADR-0006's schema), per ADR-0009's "no new
parsing logic beyond what viz/backend/server.py already has".
"""

from pathlib import Path

import numpy as np
import torch

from arc_env import actions
from arc_env.env import PAD_VALUE
from trainers.ppo.network import MAX_ARITY, ActorCritic
from viz.backend import server as backend

# Inverts `arc_env.actions`'s per-`ArgSpec.kind` `decode` function, to
# recover a raw `Discrete(RAW_ARG_RANGE)` value from a logged *decoded*
# episode value. `color`/`factor` decode (`raw % 10` / `2 + raw % 3`) is
# many-to-one, so this picks the canonical smallest raw value in that
# decode's preimage - any raw value in the same preimage decodes
# identically, so the policy loses no reachable behavior, only ends up
# imitating one arbitrary representative among equivalent raw encodings.
_DECODE_INVERSE = {
    "color": lambda value: value,
    "factor": lambda value: value - 2,
    "coord": lambda value: value,
    "dim": lambda value: value - 1,
}


def _pad_grid(grid: list) -> np.ndarray:
    obs = np.full((actions.MAX_GRID_DIM, actions.MAX_GRID_DIM), PAD_VALUE, dtype=np.int8)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            obs[i, j] = v
    return obs


def _encode_action(action_name: str, action_args: dict) -> tuple:
    primitive_index = actions.ACTION_BY_NAME[action_name]
    action = actions.ACTIONS[primitive_index]
    raw_args = [0] * MAX_ARITY  # slots beyond this primitive's arity are masked out in the loss
    for i, spec in enumerate(action.args):
        raw_args[i] = _DECODE_INVERSE[spec.kind](action_args[spec.name])
    return primitive_index, tuple(raw_args)


def load_demonstration(gp_run_dir: Path, episode_id: str = "best-program") -> list:
    """One dict per logged step of `gp_run_dir`'s best-program episode:
    `{"grid": (30, 30) int8 array, "primitive": int, "arg1": int, ...,
    "arg{MAX_ARITY}": int}` - ready to batch into tensors for
    `pretrain_from_demonstration`. Every logged step is included (even ones
    with `valid_action=False`), matching ADR-0009's "clone the demonstrated
    action" with no fitness-quality or validity gate."""

    episode = backend.read_episode(Path(gp_run_dir).parent, Path(gp_run_dir).name, episode_id)
    demonstration = []
    for step in episode["steps"]:
        primitive_index, raw_args = _encode_action(step["action"]["name"], step["action"]["args"])
        demonstration.append({
            "grid": _pad_grid(step["grid_before"]),
            "primitive": primitive_index,
            **{f"arg{i + 1}": raw_args[i] for i in range(MAX_ARITY)},
        })
    return demonstration


def pretrain_from_demonstration(
    network: ActorCritic,
    demonstration: list,
    n_epochs: int,
    batch_size: int,
    lr: float,
) -> list:
    """Minimizes cross-entropy - equivalently, negative log-likelihood under
    the same factored `Categorical` distributions PPO's own log-prob
    recomputation (`ActorCritic.get_action_and_value(action=...)`) already
    produces - between the policy's action distribution and each
    demonstrated `(grid, action)` pair, for `n_epochs` passes. Returns the
    mean loss per epoch (empty if `demonstration` is empty)."""

    if not demonstration:
        return []

    obs = torch.as_tensor(np.stack([d["grid"] for d in demonstration]), dtype=torch.long)
    action = {
        "primitive": torch.tensor([d["primitive"] for d in demonstration], dtype=torch.long),
        **{
            f"arg{i + 1}": torch.tensor([d[f"arg{i + 1}"] for d in demonstration], dtype=torch.long)
            for i in range(MAX_ARITY)
        },
    }

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    n = obs.shape[0]
    losses = []

    for _ in range(n_epochs):
        perm = np.random.permutation(n)
        epoch_losses = []
        for start in range(0, n, batch_size):
            mb_idx = torch.as_tensor(perm[start:start + batch_size], dtype=torch.long)
            mb_obs = obs[mb_idx]
            mb_action = {k: v[mb_idx] for k, v in action.items()}

            sample = network.get_action_and_value(mb_obs, action=mb_action)
            loss = -sample.log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        losses.append(float(np.mean(epoch_losses)))

    return losses
