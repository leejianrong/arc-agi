# TTT spike (web) - ARC-AGI-2 on Kaggle L4x4
P = primary source (team/paper/official page). S = secondary/third-party (treat as hearsay).

## 1. 2025 results
- Tech report arXiv 2601.10904 (P): 1,455 teams; top private 24%; "TTT is responsible for top scores in 2024 (ARChitects) and 2025 (NVARC)". Private: NVARC 24.03, ARChitects 16.53, MindsAI 12.64. Report gives NVARC cost $0.20/task. https://arxiv.org/html/2601.10904v1 ; https://arcprize.org/blog/arc-prize-2025-results-analysis
- NVARC (Sorokin, Puget; 4B). NVIDIA blog (P): synthetic data + TTT, NeMo Skills for SDG, ~$0.20/task, 27.64% public. https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/
  Trelis interview (S, summary of team talk): Qwen 4B alone in final; tokenizer cut to 16 tokens; Re-ARC + competition data x256 augmentations, synthetic x24-32 (8 geometric x colour perms); up to 30 pairs/puzzle in training; ~260k LLM-generated puzzles filtered to ~100k (GPT-OSS; code kept if 8/20 impls agree); TRM component; ensembling w/ TTT only slight gain; post-comp extended TTT pass@1 10->18%, pass@128 ~30%; 12h on 4 small GPUs. https://trelis.substack.com/p/nvarc-2025-arc-prize-winners
  Third-party port (S): checkpoint `sorokin/qwen3_4b_grids15_sft139` public; per-task LoRA r=256 rslora; batched DFS decode + NLL voting; datasets NVARC training 47,337 puzzles / full 55,886 (Kaggle dataset sorokin/nvarc-artifacts-puzzles); no unsloth needed. LR/epochs NOT found. https://github.com/pyaroslav/arc-agi-2
  Official code: Kaggle notebook linked from ARC Prize page (unsloth-flash-lora-batch8-queue-trm2 title suggests unsloth + batch8) (P link, content not readable). Writeup https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc (page body not fetchable by me).
- ARChitects 2025 (P: https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/ ): LLaDA-8B-Base masked diffusion w/ 2D RoPE, 32x32 padded; pretrain data Re-ARC, ARC-GEN-100K, ARC1/2 train+eval, ARC-Heavy, ConceptARC; pretrain rank-512 LoRA 175k steps bs8, 39h on 8xH100; per-task TTT: rank-32 LoRA, 128 steps, bs1, distinct random augmentation each step, run on Kaggle L4s; inference 102 recursive refinement steps, pick most-visited, top-2. ~30.5% eval (shape known), 21.67 public. Code https://github.com/LambdaLabsML/ARC2025_Solution_by_the_ARChitects (no licence file seen; report CC BY-SA 4.0).
- ARChitects 2024 (P: https://github.com/da-fr/arc-prize-2024): Mistral-NeMo-Minitron-8B, LoRA via unsloth, trained on H100, 24GB cards need bs2; 53.5 on ARC-1 private; Apache-2.0; HF bnb-4bit model published.
- MindsAI (P: https://github.com/jcole75/arc_2025_mindsai): MIT; CodeT5-660M-based configs, pretrain on HF `mindware/arc-agi-mega`; per-task TTFT then augmented inference (AIRV); TPU scripts. 12.64 private.
- Barbadillo 5th (6.53) (P: https://ironbar.github.io/arc25/05_Solution_Summary/): Architects single-task TTT on 4 GPUs 11.94 in April; search-and-learn w/ BARC 8B: +5pts pass@n on ARC-1 eval, 0 ARC-2 private; RL collapsed; small models cannot self-refine. No licence stated.
- TRM 7M: 8% ARC-2, CompressARC 76K params ~4% ARC-2 (20% ARC-1), ~20 min/puzzle on RTX4070 (P: tech report; https://arxiv.org/abs/2512.06104).

## 2. Papers
- Akyurek et al. 2411.07279 (P): Llama-3 8B FT 18.3% -> 47.1% with TTT on public eval (ARC-1); BARC model+TTT 53.0; ensemble 61.9. 1B ablation 5% -> 29% (80 tasks). Recipe: per-task LoRA r128 alpha16, lr 5e-5 (1B/3B) / 1e-4 (8B), 2 epochs, bs 1-2, up to 250 examples/task (leave-one-out ICL tasks, invertible geometric augs, colour/order shuffles), loss on demo+test outputs; shared adapter hurt; QLoRA 1B lost ~3/80. Pre-FT: Re-ARC, lr 2.5e-5, 2 ep. ~12h on one A100 per 100 tasks incl. TTT+inference (8B). Hierarchical aug voting. https://arxiv.org/html/2411.07279
- 2026: Multi-Perspective Transformers (2605.01154, P abstract): TinyLM, TTT+PoE, 96.1% train split / 21.7% ARC-2 eval split (https://arxiv.org/abs/2605.01154). ARC-AGI-2 Technical Report 2603.06590 and 2610.08680 only seen via search snippets.

## 3. Repos/Reddit
- Could not find Reddit/LocalLLaMA threads with useful L4 tips (search returned none). Community 2026 repos (S, 0-1 stars): lucyellu/arc-agi2-v2 (Qwen3-4B+TTT on NVARC; 28.19 public LB May 2026; selection bottleneck: correct answer ranked 7th), pyaroslav/arc-agi-2 (reproduced 22.8% on 30 eval tasks, aug_n=8, oracle 37% vs achieved 29%; projects ~5.4h/240 tasks on L4x4 [S, unmeasured on L4]; transformers 5.9 tokenizer bug). davidlueng986 repo mentions public notebooks at LB 33.89 (details unreadable).
- Kaggle 2026 board (P-ish snippet): top 88.06 (Tufa Labs), ~50% of test; method unpublished; gold line rank 16. Unexplained; do not use as target.
- ARC Prize 2026 rules page (P): offline, open-source required for prizes, hardware to be announced; TTT not mentioned (one article claiming TTT disallowed is unverified; ARC-2025 winners used it). https://arcprize.org/competitions/2026/arc-agi-2

## 4/5. Judgement (mine, not sourced unless noted)
Viable: Qwen3-4B (Apache-2.0, NVARC lineage), Qwen3-1.7B/0.6B, Llama-3.2-1B/3B (Llama licence, not permissive-clean), Minitron-8B (NVIDIA licence, not Apache). 4B bf16 + LoRA fits 24GB L4 (L4 supports bf16, no flash-attn needed per port).
Recipes:
1. Reuse NVARC 4B checkpoint + per-task LoRA TTT + DFS + aug vote (pyaroslav/lucyellu-style). Evidence: 22.8% (30 tasks, S), 24.03 private (P), ~27 public. Needs checkpoint+tokenizer offline (public HF; licence NOT verified - check Qwen3 Apache-2.0 + NVARC artifacts licence). Risks: licence of third-party repos (none stated), 11h budget, selection. Expected 15-25%. Best.
2. ARChitects-2024-style Minitron-8B or LLaDA recipe: needs pretrain checkpoint (HF bnb-4bit exists; LLaDA licence MIT but 8B diffusion decode slow). Expected 10-20%; higher risk on time.
3. Own small model (Qwen3-0.6B-1.7B) pretrained on Re-ARC/ARC-GEN + per-task TTT: needs days of GPU pretrain; unrealistic by Nov 2 beyond ~5-10%.
Unrealistic: own synthetic generation at NVARC scale, RL, anything pretraining 8B.

## Could not verify
NVARC LR/epochs/steps/batch, kaggle writeup text, licence of NVARC checkpoint/dataset/ARChitects code, real L4 throughput numbers, Reddit tips, Tufa 88 method, 2603.06590 results, whether TTT is allowed by 2026 rules text.
