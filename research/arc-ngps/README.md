# ARC Neural-Guided Program Synthesis (NGPS) — Scaffold

This repo is an initial scaffold for an ARC-AGI solver based on:
- Perceiver-style Vision Encoder (pixel tokens -> K latent slots)
- Cross-attention within-pair induction + permutation-invariant across-pair aggregation
- Transformer symbolic decoder to a typed DSL (AST, JSON/S-expr serializable)
- Decode -> Execute -> Verify loop with neural-guided beam search

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m arc_ngps.train.smoke_train --help
```

## What’s implemented (initial)
- ARC JSON loader + padding/masking tokenization
- Perceiver-style encoder (cross-attention bottleneck)
- Pair induction module (bi-cross-attn + intent pooling)
- DeepSets aggregator
- DSL AST scaffolding + simple executor (minimal ops)
- Beam search + verification loop skeleton
- Smoke training script (synthetic stub)

This is a starting point: expand the DSL + executor, add constrained decoding,
and plug in synthetic DSL program generation for pretraining.
