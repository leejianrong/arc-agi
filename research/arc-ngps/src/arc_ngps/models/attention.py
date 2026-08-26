from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Minimal MHA with optional key padding mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,              # [B, Nq, D]
        x_kv: torch.Tensor,             # [B, Nk, D]
        kv_mask: Optional[torch.Tensor] = None,  # [B, Nk] True for valid
    ) -> torch.Tensor:
        B, Nq, D = x_q.shape
        _, Nk, _ = x_kv.shape

        q = self.qkv(x_q)[..., :D]
        kv = self.qkv(x_kv)
        k = kv[..., D:2*D]
        v = kv[..., 2*D:]

        q = q.view(B, Nq, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,Nq,dh]
        k = k.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,Nk,dh]
        v = v.view(B, Nk, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B,H,Nq,Nk]
        if kv_mask is not None:
            # kv_mask: True valid, so mask invalid with -inf
            invalid = ~kv_mask[:, None, None, :]  # [B,1,1,Nk]
            attn = attn.masked_fill(invalid, float("-inf"))

        w = torch.softmax(attn, dim=-1)
        w = self.dropout(w)
        y = w @ v  # [B,H,Nq,dh]
        y = y.transpose(1, 2).contiguous().view(B, Nq, D)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # self-attn: q=kv=x
        x = x + self.dropout(self.attn(self.ln1(x), self.ln1(x), kv_mask=mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x
