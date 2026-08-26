from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn

from arc_ngps.models.perceiver import PerceiverEncoder
from arc_ngps.models.induction import PairInduction, DeepSetsAggregator
from arc_ngps.models.decoder import DSLDecoder, DecoderConfig


@dataclass
class NGPSConfig:
    d_model: int = 256
    n_latents: int = 64
    enc_heads: int = 8
    enc_layers: int = 4
    enc_ff: int = 1024
    pair_layers: int = 2
    pair_ff: int = 1024
    dec: DecoderConfig = field(default_factory=DecoderConfig)


class NGPSModel(nn.Module):
    """End-to-end model: (I/O pairs) -> hypothesis embedding -> DSL tokens."""

    def __init__(self, cfg: NGPSConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = PerceiverEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.enc_heads,
            d_ff=cfg.enc_ff,
            n_latents=cfg.n_latents,
            n_self_layers=cfg.enc_layers,
        )
        self.pair = PairInduction(
            d_model=cfg.d_model,
            n_heads=cfg.enc_heads,
            d_ff=cfg.pair_ff,
            n_layers=cfg.pair_layers,
        )
        self.agg = DeepSetsAggregator(d_model=cfg.d_model)
        self.decoder = DSLDecoder(cfg.dec)

    def encode_grid(self, tokens: torch.LongTensor, pos: torch.LongTensor, mask: torch.BoolTensor) -> torch.Tensor:
        return self.encoder(tokens=tokens, pos=pos, mask=mask)  # [B,K,D]

    def task_hypothesis(
        self,
        in_tokens: torch.LongTensor, in_pos: torch.LongTensor, in_mask: torch.BoolTensor,   # [B,P,N..]
        out_tokens: torch.LongTensor, out_pos: torch.LongTensor, out_mask: torch.BoolTensor,
        pair_mask: Optional[torch.BoolTensor] = None,  # [B,P]
    ) -> torch.Tensor:
        B, P, N = in_tokens.shape
        intents = []
        for p in range(P):
            z_in = self.encode_grid(in_tokens[:, p], in_pos[:, p], in_mask[:, p])
            z_out = self.encode_grid(out_tokens[:, p], out_pos[:, p], out_mask[:, p])
            intents.append(self.pair(z_in, z_out))
        intents = torch.stack(intents, dim=1)  # [B,P,D]
        hyp = self.agg(intents, pair_mask=pair_mask)  # [B,D]
        return hyp

    def forward(
        self,
        in_tokens: torch.LongTensor, in_pos: torch.LongTensor, in_mask: torch.BoolTensor,
        out_tokens: torch.LongTensor, out_pos: torch.LongTensor, out_mask: torch.BoolTensor,
        ys: torch.LongTensor,
        pair_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        hyp = self.task_hypothesis(in_tokens, in_pos, in_mask, out_tokens, out_pos, out_mask, pair_mask)
        return self.decoder(ys, hyp)
