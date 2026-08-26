from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

from arc_ngps.models.ngps_model import NGPSModel, NGPSConfig
from arc_ngps.models.decoder import DecoderConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    cfg = NGPSConfig(
        d_model=128,
        n_latents=32,
        enc_heads=4,
        enc_layers=2,
        enc_ff=256,
        pair_layers=1,
        pair_ff=256,
        dec=DecoderConfig(vocab_size=128, d_model=128, n_heads=4, d_ff=256, n_layers=2, max_len=64),
    )
    model = NGPSModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Fake batch: B=2, P=3 pairs, N=25 tokens (5x5)
    B, P, N = 2, 3, 25
    in_tokens = torch.randint(0, 10, (B, P, N), device=device)
    out_tokens = torch.randint(0, 10, (B, P, N), device=device)
    pos = torch.stack(torch.meshgrid(torch.arange(5), torch.arange(5), indexing="ij"), dim=-1).reshape(-1, 2)
    in_pos = pos[None, None, :, :].expand(B, P, N, 2).to(device)
    out_pos = in_pos.clone()
    in_mask = torch.ones((B, P, N), dtype=torch.bool, device=device)
    out_mask = torch.ones((B, P, N), dtype=torch.bool, device=device)

    bos, eos = 1, 2
    ys_in = torch.full((B, 8), bos, dtype=torch.long, device=device)
    ys_tgt = torch.randint(0, cfg.dec.vocab_size, (B, 8), device=device)

    for step in range(args.steps):
        logits = model(in_tokens, in_pos, in_mask, out_tokens, out_pos, out_mask, ys_in)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ys_tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step:03d} loss {loss.item():.4f}")

    print("Smoke train done.")


if __name__ == "__main__":
    main()
