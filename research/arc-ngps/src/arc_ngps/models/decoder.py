from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecoderConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_layers: int = 6
    max_len: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal[None, None, :, :], float("-inf"))
        w = torch.softmax(att, dim=-1)
        w = self.drop(w)
        y = w @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D], mem: [B,M,D]
        B, T, D = x.shape
        M = mem.shape[1]
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(mem).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(mem).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        w = torch.softmax(att, dim=-1)
        w = self.drop(w)
        y = w @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.ln1(x)))
        x = x + self.drop(self.cross_attn(self.ln2(x), mem))
        x = x + self.drop(self.ff(self.ln3(x)))
        return x


class DSLDecoder(nn.Module):
    """Transformer decoder producing DSL token sequences.

    NOTE: This is an unconstrained decoder scaffold.
    Add grammar-constrained decoding in `arc_ngps/search/beam.py`.
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)
        ])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ys: torch.LongTensor, mem: torch.Tensor) -> torch.Tensor:
        # ys: [B,T] input tokens (teacher-forcing); mem: [B,D] or [B,M,D]
        B, T = ys.shape
        if mem.ndim == 2:
            mem = mem[:, None, :]  # [B,1,D]
        pos = torch.arange(T, device=ys.device)
        x = self.tok_emb(ys) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x, mem)
        x = self.ln(x)
        return self.head(x)  # [B,T,V]

    @torch.no_grad()
    def next_logits(self, ys: torch.LongTensor, mem: torch.Tensor) -> torch.Tensor:
        logits = self.forward(ys, mem)
        return logits[:, -1, :]  # [B,V]
