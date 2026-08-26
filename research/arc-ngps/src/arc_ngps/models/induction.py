from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, TransformerBlock


class PairInduction(nn.Module):
    """Within-pair alignment: bidirectional cross-attn between input/output latents -> edit intent."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.xattn_in_to_out = MultiHeadAttention(d_model, n_heads, dropout)
        self.xattn_out_to_in = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln_in = nn.LayerNorm(d_model)
        self.ln_out = nn.LayerNorm(d_model)

        self.combine = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

        self.intent_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, z_in: torch.Tensor, z_out: torch.Tensor) -> torch.Tensor:
        # z_in, z_out: [B,K,D]
        a = z_in + self.xattn_in_to_out(self.ln_in(z_in), self.ln_out(z_out), kv_mask=None)
        b = z_out + self.xattn_out_to_in(self.ln_out(z_out), self.ln_in(z_in), kv_mask=None)
        z = self.combine(torch.cat([a, b, a - b, a * b], dim=-1))  # [B,K,D]
        for blk in self.blocks:
            z = blk(z)
        intent = self.intent_pool(z).mean(dim=1)  # [B,D]
        return intent


class DeepSetsAggregator(nn.Module):
    """Across-pair permutation invariant aggregation."""

    def __init__(self, d_model: int = 256, d_hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.phi = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, intents: torch.Tensor, pair_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        # intents: [B,P,D]
        x = self.phi(intents)
        if pair_mask is not None:
            m = pair_mask.to(x.dtype)[..., None]
            x = x * m
            denom = pair_mask.sum(dim=1).clamp(min=1).to(x.dtype)  # [B]
            pooled = x.sum(dim=1) / denom[:, None]
        else:
            pooled = x.mean(dim=1)
        return self.rho(pooled)  # [B,D]
