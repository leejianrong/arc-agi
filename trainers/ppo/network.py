"""The policy/value network, per ADR-0008 point 2.

    color-embed each cell over the 30x30 scratch canvas, with an active-
    region mask channel -> a few conv/residual blocks + self-attention ->
    a factored action head (primitive first, then its typed arguments) ->
    a linear value head off the same pooled features.

One scoped-down deviation from ADR-0008's literal wording, worth flagging:
"coordinates via a pointer/attention lookup over the spatial feature map"
is implemented here as a small MLP head over the pooled+primitive-embedding
context, not a dedicated spatial pointer network. `fill_cell` and `commit`
(V3, ADR-0002) are the curated actions with coordinate args - `commit`'s
*are* exercised by a fixture task (`d10ecb37`: `commit(0, 0, 2, 2)`) - but
ADR-0008 also fixes per-task, solve-time training (point 1): one policy is
trained per task, never shared across tasks, so a coordinate head only ever
needs to represent *one task's* fixed answer (here, always "(0, 0)"), not
generalize across many objects/positions the way a real pointer network
would justify. A small conditioned MLP can already fit that; a full spatial
pointer mechanism would be added complexity with no accuracy payoff at this
training regime - upgrading it later (e.g. for multi-task or cross-task
training) wouldn't touch `arc_env`/the action space at all. The
self-attention layer(s) over the spatial feature map (the part of ADR-0008
that isn't task-specific) are implemented as specified.

The network doesn't know which *kind* of argument (color/factor/coord/dim)
each of the 4 generic arg slots means for a given primitive - `arc_env.
actions`'s decode functions already handle that from a plain
`Discrete(RAW_ARG_RANGE)` value, so every arg head is just a
`RAW_ARG_RANGE`-way categorical conditioned on the chosen primitive,
matching the env's own action `Dict` space exactly.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from arc_env import actions
from arc_env.env import PAD_VALUE

N_COLORS_PLUS_PAD = PAD_VALUE + 1  # 0-9 real colors, 10 = padding
N_ACTIONS = len(actions.ACTIONS)
GRID_DIM = actions.MAX_GRID_DIM
RAW_ARG_RANGE = actions.RAW_ARG_RANGE
MAX_ARITY = actions.MAX_ARITY

EMBED_DIM = 24
CONV_CHANNELS = 64
N_CONV_BLOCKS = 2
N_ATTN_LAYERS = 1
N_ATTN_HEADS = 4
PRIMITIVE_EMBED_DIM = 16

# Per-primitive arity, as a lookup tensor - which of the 3 generic arg slots
# are actually "live" (used by `arc_env.actions.execute`) for a given
# sampled primitive, needed to mask arg log-probs/entropy in a batched
# update (different rows of a batch can have different arities).
ARITY_BY_PRIMITIVE = torch.tensor([a.arity for a in actions.ACTIONS], dtype=torch.long)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


@dataclass
class ActionSample:
    primitive: torch.Tensor  # (B,)
    args: torch.Tensor  # (B, MAX_ARITY)
    log_prob: torch.Tensor  # (B,) - primitive log-prob + live args' log-probs
    entropy: torch.Tensor  # (B,) - primitive entropy + live args' entropy
    value: torch.Tensor  # (B,)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.color_embed = nn.Embedding(N_COLORS_PLUS_PAD, EMBED_DIM)
        self.input_proj = nn.Conv2d(EMBED_DIM + 1, CONV_CHANNELS, kernel_size=1)
        self.conv_blocks = nn.ModuleList(ResidualBlock(CONV_CHANNELS) for _ in range(N_CONV_BLOCKS))

        self.attn_layers = nn.ModuleList(
            nn.MultiheadAttention(CONV_CHANNELS, N_ATTN_HEADS, batch_first=True)
            for _ in range(N_ATTN_LAYERS)
        )
        self.attn_norms = nn.ModuleList(nn.LayerNorm(CONV_CHANNELS) for _ in range(N_ATTN_LAYERS))

        self.value_head = nn.Linear(CONV_CHANNELS, 1)
        self.primitive_head = nn.Linear(CONV_CHANNELS, N_ACTIONS)

        self.primitive_embed = nn.Embedding(N_ACTIONS, PRIMITIVE_EMBED_DIM)
        arg_context_dim = CONV_CHANNELS + PRIMITIVE_EMBED_DIM
        self.arg_heads = nn.ModuleList(nn.Linear(arg_context_dim, RAW_ARG_RANGE) for _ in range(MAX_ARITY))

    def _encode(self, grid: torch.Tensor) -> torch.Tensor:
        """`grid`: (B, 30, 30) long, values 0-10. Returns pooled features (B, CONV_CHANNELS)."""

        mask = (grid != PAD_VALUE).float()  # (B, 30, 30)
        x = self.color_embed(grid)  # (B, 30, 30, EMBED_DIM)
        x = torch.cat([x, mask.unsqueeze(-1)], dim=-1)  # (B, 30, 30, EMBED_DIM + 1)
        x = x.permute(0, 3, 1, 2)  # (B, EMBED_DIM + 1, 30, 30)
        x = self.input_proj(x)
        for block in self.conv_blocks:
            x = block(x)

        b, c, h, w = x.shape
        seq = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        for attn, norm in zip(self.attn_layers, self.attn_norms):
            attn_out, _ = attn(seq, seq, seq, need_weights=False)
            seq = norm(seq + attn_out)

        flat_mask = mask.flatten(1).unsqueeze(-1)  # (B, H*W, 1)
        pooled = (seq * flat_mask).sum(dim=1) / flat_mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def get_value(self, grid: torch.Tensor) -> torch.Tensor:
        pooled = self._encode(grid)
        return self.value_head(pooled).squeeze(-1)

    def get_action_and_value(self, grid: torch.Tensor, action: dict = None) -> ActionSample:
        """`action`, if given, is `{"primitive": (B,), "arg1": (B,), ...}` -
        recomputes log-prob/entropy for an already-taken action (PPO update)
        instead of sampling a new one (rollout collection)."""

        pooled = self._encode(grid)
        value = self.value_head(pooled).squeeze(-1)

        primitive_logits = self.primitive_head(pooled)
        primitive_dist = Categorical(logits=primitive_logits)

        if action is None:
            primitive = primitive_dist.sample()
        else:
            primitive = action["primitive"]

        log_prob = primitive_dist.log_prob(primitive)
        entropy = primitive_dist.entropy()

        arity = ARITY_BY_PRIMITIVE.to(primitive.device)[primitive]  # (B,)
        primitive_ctx = self.primitive_embed(primitive)
        arg_context = torch.cat([pooled, primitive_ctx], dim=-1)

        arg_values = []
        for i, head in enumerate(self.arg_heads):
            arg_logits = head(arg_context)
            arg_dist = Categorical(logits=arg_logits)
            arg_value = arg_dist.sample() if action is None else action[f"arg{i + 1}"]
            arg_values.append(arg_value)

            live = (arity > i).float()
            log_prob = log_prob + live * arg_dist.log_prob(arg_value)
            entropy = entropy + live * arg_dist.entropy()

        args = torch.stack(arg_values, dim=-1)  # (B, MAX_ARITY)
        return ActionSample(primitive=primitive, args=args, log_prob=log_prob, entropy=entropy, value=value)

    def get_greedy_action(self, grid: torch.Tensor) -> tuple:
        """Argmax over the primitive head, then argmax args conditioned on
        it - the deterministic-policy counterpart of `get_action_and_value`'s
        sampling, for eval-episode replay (not training)."""

        pooled = self._encode(grid)
        primitive = torch.argmax(self.primitive_head(pooled), dim=-1)
        primitive_ctx = self.primitive_embed(primitive)
        arg_context = torch.cat([pooled, primitive_ctx], dim=-1)
        args = torch.stack([torch.argmax(head(arg_context), dim=-1) for head in self.arg_heads], dim=-1)
        return primitive, args
