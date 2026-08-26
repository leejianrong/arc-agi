from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, TransformerBlock


class GridTokenEmbed(nn.Module):
    """Embeds (color, y, x) into token vectors."""

    def __init__(self, d_model: int, max_h: int = 30, max_w: int = 30, n_colors: int = 10):
        super().__init__()
        self.color = nn.Embedding(n_colors, d_model)
        self.pos_y = nn.Embedding(max_h, d_model)
        self.pos_x = nn.Embedding(max_w, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.LongTensor, pos: torch.LongTensor) -> torch.Tensor:
        # tokens: [B,N], pos: [B,N,2] with (y,x)
        y = pos[..., 0].clamp(min=0)
        x = pos[..., 1].clamp(min=0)
        emb = self.color(tokens) + self.pos_y(y) + self.pos_x(x)
        return self.ln(emb)


class PerceiverEncoder(nn.Module):
    """Perceiver-style encoder: tokens -> K latent slots via cross-attention bottleneck."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_latents: int = 64,
        n_self_layers: int = 4,
        dropout: float = 0.0,
        max_h: int = 30,
        max_w: int = 30,
    ):
        super().__init__()
        self.embed = GridTokenEmbed(d_model, max_h=max_h, max_w=max_w)
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_ln_q = nn.LayerNorm(d_model)
        self.cross_ln_kv = nn.LayerNorm(d_model)
        self.self_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_self_layers)
        ])
        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.LongTensor,        # [B,N]
        pos: torch.LongTensor,           # [B,N,2]
        mask: Optional[torch.BoolTensor] = None,  # [B,N] True valid
    ) -> torch.Tensor:
        x = self.embed(tokens, pos)  # [B,N,D]
        B = x.shape[0]
        lat = self.latents.expand(B, -1, -1)  # [B,K,D]
        # Cross-attend: latents query tokens
        lat = lat + self.cross_attn(self.cross_ln_q(lat), self.cross_ln_kv(x), kv_mask=mask)
        for blk in self.self_blocks:
            lat = blk(lat, mask=None)  # latents are fixed length; no padding
        return self.out_ln(lat)  # [B,K,D]
