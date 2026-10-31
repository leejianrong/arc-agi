# TTT spike: public Kaggle ARC-AGI-2 2026 notebooks (read only, nothing executed)

Sources pulled to this dir: kaiserm/nvarc-winning-solution-2025-4xl4s (stock NVARC, read in full), koushikrudra/failed-in-aimo (original of the "33.89" baseline), mikelou1/...perfpatch (diff vs stock: only GPU-side logit math), luxluxshan/arc2-nvarc-v1, evgendvorkin/...boyko-nvarc, junaid512, yiheng (2-line diff of stock), sorokin/pip-install-unsloth-flash-patch.

## Headline
Every public TTT notebook is the SAME pipeline: NVARC 2025 (ARC Prize 2025 winner) re-run on 4xL4. Public LB 27-36. The top of the LB is 80-88% (Tufa Labs 88.06, #2 87.22, #3 83.89, ...) and per forum is Qwen3.8-Flash-Next / 27B agent harnesses, not TTT. Anil Thomas (forum 750546): "many of the leading teams are using Qwen3.8-Flash-Next along with the Duck harness"; his NVARC-style TTT was ~35 and ensembling with a Qwen3.8-27B run gave 45. TTT is a ~30-35 floor, not a route to a win.

## 1. Pipeline (stock NVARC notebook, cells 2-7)
- Base: `sorokin/qwen3_4b_grids15_sft139` v1 Transformers/bfloat16, Qwen3-4B (36 layers, 7.27 GB), author "ivan" (Ivan Sorokin, NVARC). License on Kaggle: Apache 2.0. Pre-fine-tuned by NVARC (SFT on their large synthetic ARC data) with a reduced 16-token grid vocabulary (digits 0-9 -> ids 0-9, newline=10, <|im_end|>=15; ARC_VOCAB, USER=11, ASSISTANT=12, PAD=13). Training data/recipe for sft139 NOT in the notebooks. A copy: konstantinboyko/qwen3-4b-bfloat16-v02-01-01 (Apache 2.0).
- Prompt: multi-turn chat. Each train pair `<|im_start|>user\n{rows of digits, no spaces}<|im_end|><|im_start|>assistant\n{grid}<|im_end|>`, then the test input and the answer. One char per cell, newline per row. max_seq_length 8192; examples dropped from the front (cut_to_len) when too long.
- TTT: per task, sequentially, one task at a time per GPU (4 independent workers, one per L4, shared queue; no DDP). Per task: reset LoRA to saved initial weights, train, infer. LoRA r=256, rsLoRA, alpha=32, dropout 0, targets q,k,v,o,gate,up,down + embed_tokens + lm_head; lr 5e-5 cosine, warmup_ratio 0.1, 1 epoch, per-device batch 1, grad-accum 1, adamw_torch, bf16, no gradient checkpointing, completion-only loss (only assistant replies after the first example). Training set per task = demonstration pairs + test input...  precisely: the task's train pairs (test outputs unknown) augmented 8 geometries (identity, transpose, rot90 x3 and transposes) x 16 random colour permutations (permute_rnd_all_ over all 10 colours incl. 0) = 128 sequences, with example order shuffled (shuffle_ex). Leave-last-out: fmt_train(last_is_challenge=True) makes the last demo pair the target.
- Inference: custom "turbo_dfs": probability-threshold DFS (max_score = -ln 0.2, i.e. keep branches with cumulative prob > 0.2), batched KV cache over views, tokens restricted to the 12 ARC ids. 16 views per test input (augment n=2 -> 8 geometries x 2 colour perms), decoded in 4 batches of 4. Per-task wall cap 1200 s, DFS call cap 540 s.
- Scoring/selection: each distinct candidate is re-scored (teacher-forced NLL) under 8 fresh augmentations (calc_scores). `score_kgmon` = (#views that produced the grid) - mean(augmented NLL); top 2 distinct grids -> attempt_1/attempt_2. Fallback [[0]].

## 2. Time / memory
- Stock notebook: global_end_time = now + 12h - 10min; tasks processed in sorted key order, so late tasks are silently skipped if time runs out. Hidden rerun = 240 tasks per luxluxshan; stock 128-seq config uses the whole 12 h.
- Hardware: 4xL4 (24 GB each) with CUDA_VISIBLE_DEVICES per worker; 4B bf16 + r256 LoRA fits (forum 749756: workers peak ~17 GB per L4). Notebook metadata machine_shape NvidiaL4. The 2025 version was T4x2 (nihilisticneuralnet, albartrose notebooks, not read; "best 19.17").
- Throughput numbers: luxluxshan cites half-cost (64 seq, 8 views) runs ~6 h, scoring 35.97 (5h44) and 33.89 (3h50) on LB runtime board (unverified claims). Forum: 2 epochs/bigger beams -> timeouts, score fell 30.14 -> 24.17 (749415).
- mikelou1 perfpatch (33.89 label): only moves log-softmax to GPU/ restricts to 12 tokens, no use_cache in scoring.
- Same-code reruns vary 26.9-32.2 (luxluxshan's five reruns: 29.7/31.8/26.9/32.2/31.4); evgendvorkin: 28.5-30.6 single pass, 31.8-32.6 two-pass.
- Platform risk: forum 753315 says some L4x4 servers have ~10x slower inter-GPU/memory speed; 753865 report of 56.94 -> 26.67 unexplained; scoring queues 18-36 h and "Kaggle Error" failures (747515, 753723, 753659).

## 3. Offline dependencies
- `sorokin/pip-install-unsloth-flash-patch` is a UTILITY SCRIPT kernel (attached as kernel_source; luxluxshan checks sys.path for /kaggle/usr/lib). Its cells (run with internet): `pip install --target=/kaggle/working unsloth==2025.9.7 unsloth_zoo==2025.9.9 numpy==2.2.6 matplotlib==3.10.6 scikit-learn==1.7.2`; then flash_attn 2.8.2+cu128torch2.8-cp311 wheel from github.com/mjun0812/flash-attention-prebuild-wheels v0.3.14; then a `patch` of unsloth/models/qwen3.py (fast decode path -> flash_attn_func, fixes temp_O shape). Python 3.11, torch 2.8 docker image (pinned image digest in metadata).
- Boyko variant: python 3.12, offline wheelhouse as kernel `konstantinboyko/unsloth-2025-9-7-torch-2-8-0-py3-12-patched` + dataset `konstantinboyko/flash-attn-2-8-2` (cp312 wheel), installs via `uv pip install --no-index`; also a commented unsloth 2026.7.2/torch 2.10 stack.
- Pitfalls seen in code: `pip uninstall -y tensorflow`; staggered worker start (rank>0 waits for /kaggle/workerN, "Fix Unsloth patching issue"); custom UnslothFixedTrainer for DDP/view-tensor bug (unsloth issue 2435); TRITON_PTXAS_PATH; model path is /kaggle/input/models/sorokin/... and competition data at /kaggle/input/competitions/arc-prize-2026-arc-agi-2/ (stock cell uses old path forms in some versions).
- Debug mode (non-rerun) runs only 4 eval tasks (0934a4d8, 36a08778, 981571dc, aa4ec2a5); hidden rerun uses arc-agi_test_challenges.json.

## 4. Forum (ids): 749415 (4xL4 ceiling ~30, "higher than you think"), 749756 (4B TTT stuck 28, 27B FP8 load error), 748082 (Qwen3.8-27B 15-26.8 LB for early agents, NVARC/Qwen correlation 0.43), 750546 (Flash-Next + harness, 55+ scores, 900 tok/s engineering), 752709 (Qwen Community License 1.0 prize eligibility - host reply NOT seen; 1 comment), 748741 (LLM-generated synthetic data ToS: unresolved, Kaggle declines to advise), 748895 (fine-tuning NVARC further on 342K new synthetic examples: no LB gain, base 30.83 vs 27.9-29.2, noise +-2-4), 751292 (grand prize 85% = private LB per site), 753315/753865/751344 (infra instability).
No thread found stating NVARC dataset licence.

## 5. Scores
TTT public notebooks: 27-36 (best cited 35.97 for half-cost; typical 30-32). Public Qwen3.8-27B agent notebooks 15-26. Top private/public teams 72-88 (top 20 all >=72.2).

## 6. Feasibility for us (24 h)
Technically feasible, ~1 day: attach `sorokin/qwen3_4b_grids15_sft139` + utility script `sorokin/pip-install-unsloth-flash-patch`, copy arc_loader/solver/decoder/starter (single notebook, 9 cells), use the 4xL4 shape, budget 12h-10min. Expected result 28-33 (vs our weak run), with run-to-run noise +-3; an expanded knob (64-seq/8-view, cheap-first ordering, two passes) is the best-documented tweak. Risks: (a) licence: notebook code is by NVARC/ARChitects/Boyko/Koushik/mikelou1; we saw no licence header; Kaggle code default is Apache-2.0 but competition rules require our submitted solution under CC0/MIT-0 - third-party code would need clean-room reimplementation or permissive-licence verification; model is Apache-2.0 (fine as pretrained third-party model under rule 2.6, with the exemption clause quoted in 752709); (b) 1 submission/day and 18-36 h scoring queues + deadline 2026-11-02 23:59: only ~1-2 attempts can still finish, so no debug cycle; (c) slow-server variance, 12 h wall; (d) it will not reach the 70-88 band; ceiling ~35 while the top uses agent harnesses with Qwen3.8-Flash-Next; (e) the base "smaller model" belief is correct for TTT (4B, r256 LoRA, 17 GB/L4) but 27B+ TTT is not what the leaders do.
Recommendation: if the goal is a non-trivial score before deadline, a faithful NVARC fork is the lowest-risk path, but resolve licensing by writing our own loader/DFS (small, ~600 lines) and keep pip-install utility script only as environment; do not expect to beat the LLM-harness teams.

## Could not verify
- sft139 training data, recipe, licence of that data; NVARC repo licence; notebook code licences.
- Actual wall-clock per task and peak memory (all notebook outputs stripped; 17 GB/L4 and 6 h/3h50 are forum/markdown claims).
- Tokens/sec of TTT/DFS.
- Whether the hidden test set is 240 tasks (luxluxshan claim).
- Host answer on Qwen3.8 licence eligibility, and the "Duck harness" details.
- T4x2 NVARC notebooks, ARChitects 2025 notebooks, junaid512 details (enable_gpu false in metadata), kaggle leaderboard beyond the top 20.
- Whether the sorokin utility script still builds offline (needs internet to run; used as attached output).
