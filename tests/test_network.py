"""Tests for `trainers.ppo.network.ActorCritic`, focused on the ADR-0008
amendment: primitive-level action masking for the 5 `kind="act_on_selection"`
actions (`commit_selection`, `delete_selected`, `recolor_selected`,
`move_selected`, `paint_selected_at`) whenever the observation's selection
channel (channel 1) is empty. `arc_env.actions.execute` already makes every
one of these an unconditional no-op when nothing is selected - this masking
stops the policy from even being able to *sample* one of them in that case,
rather than only being penalized for it after the fact."""

import torch

from arc_env import actions
from arc_env.env import PAD_VALUE
from trainers.ppo.network import IS_ACT_ON_SELECTION, N_ACTIONS, ActorCritic

GRID_DIM = actions.MAX_GRID_DIM

ACT_ON_SELECTION_INDICES = [actions.ACTION_BY_NAME[name] for name in (
    "commit_selection", "delete_selected", "recolor_selected", "move_selected", "paint_selected_at",
)]


def _obs(selected: bool, batch: int = 1) -> torch.Tensor:
    """A minimal (B, 2, 30, 30) long observation: a small non-PAD patch in
    channel 0 (so `_encode`'s active-region mask isn't degenerate), and
    channel 1 either all-zero (`selected=False`) or with one selected cell
    per row (`selected=True`)."""

    grid = torch.full((batch, GRID_DIM, GRID_DIM), PAD_VALUE, dtype=torch.long)
    grid[:, 0:3, 0:3] = 2
    sel = torch.zeros((batch, GRID_DIM, GRID_DIM), dtype=torch.long)
    if selected:
        sel[:, 0, 0] = 1
    return torch.stack([grid, sel], dim=1)


def test_is_act_on_selection_matches_the_5_curated_actions():
    assert IS_ACT_ON_SELECTION.sum().item() == 5
    got = {actions.ACTIONS[i].name for i in torch.nonzero(IS_ACT_ON_SELECTION).flatten().tolist()}
    assert got == {"commit_selection", "delete_selected", "recolor_selected", "move_selected", "paint_selected_at"}


def test_empty_selection_masks_out_all_5_act_on_selection_primitives():
    torch.manual_seed(0)
    network = ActorCritic()
    obs = _obs(selected=False, batch=4)

    pooled = network._encode(obs)
    primitive_logits = network.primitive_head(pooled)
    from trainers.ppo.network import _mask_act_on_selection_logits
    masked_logits = _mask_act_on_selection_logits(primitive_logits, obs)

    for idx in ACT_ON_SELECTION_INDICES:
        assert torch.isinf(masked_logits[:, idx]).all()
        assert (masked_logits[:, idx] < 0).all()

    dist = torch.distributions.Categorical(logits=masked_logits)
    for idx in ACT_ON_SELECTION_INDICES:
        assert (dist.probs[:, idx] == 0).all()

    # Sampling many times must never yield a masked primitive.
    samples = dist.sample((5000,))
    sampled_indices = set(samples.flatten().tolist())
    assert sampled_indices.isdisjoint(set(ACT_ON_SELECTION_INDICES))

    # Entropy/log-prob must stay finite (no NaN leaking from -inf logits) for
    # non-masked entries.
    assert torch.isfinite(dist.entropy()).all()
    non_masked_idx = actions.ACTION_BY_NAME["identity"]
    lp = dist.log_prob(torch.full((4,), non_masked_idx, dtype=torch.long))
    assert torch.isfinite(lp).all()


def test_nonempty_selection_leaves_all_primitives_sampleable():
    torch.manual_seed(0)
    network = ActorCritic()
    obs = _obs(selected=True, batch=1)

    with torch.no_grad():
        pooled = network._encode(obs)
        primitive_logits = network.primitive_head(pooled)
        from trainers.ppo.network import _mask_act_on_selection_logits
        masked_logits = _mask_act_on_selection_logits(primitive_logits, obs)

    assert torch.equal(masked_logits, primitive_logits)  # untouched when something is selected
    dist = torch.distributions.Categorical(logits=masked_logits)
    assert (dist.probs > 0).all()
    assert N_ACTIONS == dist.probs.shape[-1]
    for idx in ACT_ON_SELECTION_INDICES:
        assert dist.probs[0, idx] > 0


def test_mixed_batch_masks_each_row_independently():
    torch.manual_seed(0)
    network = ActorCritic()
    selected_obs = _obs(selected=True, batch=1)
    empty_obs = _obs(selected=False, batch=1)
    obs = torch.cat([selected_obs, empty_obs], dim=0)  # row 0 selected, row 1 empty

    with torch.no_grad():
        pooled = network._encode(obs)
        primitive_logits = network.primitive_head(pooled)
        from trainers.ppo.network import _mask_act_on_selection_logits
        masked_logits = _mask_act_on_selection_logits(primitive_logits, obs)

    for idx in ACT_ON_SELECTION_INDICES:
        assert torch.isfinite(masked_logits[0, idx])  # row 0: selected, untouched
        assert torch.isinf(masked_logits[1, idx])  # row 1: empty, masked

    dist = torch.distributions.Categorical(logits=masked_logits)
    for idx in ACT_ON_SELECTION_INDICES:
        assert dist.probs[0, idx] > 0
        assert dist.probs[1, idx] == 0


def test_get_action_and_value_never_samples_act_on_selection_with_empty_selection():
    torch.manual_seed(0)
    network = ActorCritic()
    obs = _obs(selected=False, batch=8)

    sampled_primitives = set()
    with torch.no_grad():
        for _ in range(200):
            sample = network.get_action_and_value(obs)
            sampled_primitives.update(sample.primitive.tolist())

    assert sampled_primitives.isdisjoint(set(ACT_ON_SELECTION_INDICES))
    assert torch.isfinite(sample.log_prob).all()
    assert torch.isfinite(sample.entropy).all()


def test_get_greedy_action_never_picks_a_masked_primitive_even_if_its_raw_logit_is_highest():
    """Construct a network whose primitive head is hand-set so that a
    masked-when-empty primitive (`commit_selection`) has the single highest
    raw logit in the batch - argmax must still avoid it once masked."""

    network = ActorCritic()
    commit_idx = actions.ACTION_BY_NAME["commit_selection"]
    identity_idx = actions.ACTION_BY_NAME["identity"]

    with torch.no_grad():
        network.primitive_head.weight.zero_()
        network.primitive_head.bias.zero_()
        network.primitive_head.bias[commit_idx] = 100.0  # would win argmax unmasked
        network.primitive_head.bias[identity_idx] = 1.0

    obs = _obs(selected=False, batch=3)
    with torch.no_grad():
        primitive, _ = network.get_greedy_action(obs)

    assert (primitive != commit_idx).all()
    assert (primitive == identity_idx).all()

    # With something selected, the same hand-set head now must pick commit_selection.
    obs_selected = _obs(selected=True, batch=3)
    with torch.no_grad():
        primitive_selected, _ = network.get_greedy_action(obs_selected)
    assert (primitive_selected == commit_idx).all()


def test_ppo_update_recomputation_uses_the_same_mask_as_rollout_sampling():
    """`trainers.ppo.ppo.ppo_update` recomputes log-prob/entropy for
    already-taken rollout actions via `get_action_and_value(mb_obs,
    action=mb_action)`. Since masking is a pure function of `mb_obs` (no
    external/cached selection flag), the mask at recompute time must exactly
    match the mask in effect when the action was originally sampled - proven
    here by checking the two calls (sample, then recompute against the
    sampled action) produce identical log-probs for the primitive term."""

    torch.manual_seed(0)
    network = ActorCritic()
    obs = _obs(selected=False, batch=4)

    with torch.no_grad():
        rollout_sample = network.get_action_and_value(obs)
        action = {"primitive": rollout_sample.primitive, **{
            f"arg{i + 1}": rollout_sample.args[:, i] for i in range(rollout_sample.args.shape[-1])
        }}
        recomputed = network.get_action_and_value(obs, action=action)

    assert torch.allclose(rollout_sample.log_prob, recomputed.log_prob)
    assert torch.allclose(rollout_sample.entropy, recomputed.entropy)
    assert torch.isfinite(recomputed.log_prob).all()
