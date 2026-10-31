# F18 - Test-time training spike (2026-10-31)

Question: the user approved exploring test-time training (TTT) for ARC-AGI-2 and expects it needs small models. How do
strong teams do it, and is it the right bet? Two read-only research passes; raw reports in
`docs/research/arc2-ttt-spike/` (`kaggle-side.md`: public notebooks and forum; `web-side.md`: papers, GitHub, blogs).
Neither could run anything; items marked *hearsay* come from forum posts or secondary sources.

## What both reports agree on

- Every public TTT notebook this year is the same pipeline, the ARC Prize 2025 winner (NVARC) re-run on 4x L4:
  `sorokin/qwen3_4b_grids15_sft139` (a Qwen3-4B pre-fine-tuned by NVARC on synthetic ARC data, 16-token grid
  vocabulary, Apache-2.0 per Kaggle), then **per-task LoRA** (r=256 rsLoRA, lr 5e-5, 1 epoch, batch 1, completion-only
  loss) on the task's own demonstrations augmented 8 geometries x 16 colour permutations = 128 sequences, then a
  probability-threshold DFS decode over 16 augmented views and re-scoring by teacher-forced NLL under 8 fresh
  augmentations; the top two distinct grids are the two attempts.
- A 4B model with r=256 LoRA peaks near 17 GB per L4, one worker per GPU, tasks in sorted order, 12 h minus margin.
  Half-cost configs (64 sequences, 8 views) take about 6 h and score about the same (33.9-36.0 on the public board;
  rerun noise 26.9-32.2). Two epochs or larger beams caused timeouts and lower scores.
- Offline dependencies come from an attached utility kernel (`sorokin/pip-install-unsloth-flash-patch`: unsloth
  2025.9.7, flash-attn 2.8.2 cu128 torch2.8, Python 3.11) or a Python 3.12 variant (`konstantinboyko/...`, uv
  `--no-index`). One community port reports it needs neither unsloth nor flash-attn.
- Small models are indeed what TTT uses (Qwen3-4B; ARChitects 2024 used an 8B at batch 2 on 24 GB; Akyurek et al.
  report Llama-3 8B 18.3% -> 47.1% on ARC-AGI-1 with TTT, a 1B model 5% -> 29%).

## Where they differ, and how to read it

- Web side: "TTT on a small model is the proven path", expected 15-25% (2025 private numbers: NVARC 24.0%, ARChitects
  16.5%). Kaggle side: public TTT notebooks score about 27-36 on this year's **public** board, which is a 30-35 floor,
  and **the top of the board (72-88) does not use TTT**. These are different benchmarks (2025 private vs 2026 public),
  so both can be true: a TTT fork would land near 30 publicly, and probably lower privately.
- *Hearsay, forum topic 750546:* the leading teams reportedly use **Qwen3.8-Flash-Next with an agent harness
  ("Duck harness")**, reaching 55+ and up to 88; one competitor reports that NVARC-style TTT (about 35) plus a
  Qwen3.8-27B run gave 45. Public Qwen3.8-27B agent notebooks score 15-26. Nobody has published the top method.

## Reading for our decision

1. A faithful NVARC fork is the lowest-risk path to roughly 28-33 public (about a day of work). It is not a path to the
   top: ceiling about 35.
2. The biggest unexplained lever is **which model and harness the leaders use**, not TTT. Flash-Next is a different
   model (Kaggle mirrors exist: `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`,
   `sethmoudry/qwen38-flash-next-fp8-int4-experts`, drafter `dfranzen/albucino-qwen3-8-flash-next-drafter`). Its
   licence is the Qwen Community License 1.0, which the hosts declined to rule on for prize eligibility (topics 750575,
   752709). Throughput per token is probably what matters, since our 27B dense model manages only about 15-25
   tokens/sec per sequence.
3. Licensing of reuse: no licence header was found on any notebook code; the NVARC checkpoint and artifact licences are
   unverified (Qwen3 itself is Apache-2.0). Safe route: reimplement the loader, augmentation and DFS ourselves
   (about 600 lines, MIT-0) and use the public checkpoint and utility kernel only as inputs.
4. Operational risk is real: scoring queues of 18-36 h, same-code score swings (one team 56.9 -> 26.7), slow L4x4
   interconnect on some servers.

## Open questions (what we could not verify)

- What Flash-Next is (size, active parameters, speed on L4x4, vLLM support in our pinned 0.19.0), what the "Duck
  harness" does, and what the leaders' per-task token use looks like.
- NVARC's own training hyperparameters and data licences; real wall-clock and memory per phase (notebook outputs are
  stripped).
- Whether TTT is permitted under the 2026 rules (the official page does not mention it; one article says no).

## Next

A second, narrower spike on Flash-Next plus the leading harness design, then decide between (a) a clean-room NVARC-style
TTT build, (b) moving our synthesis/agent harness to Flash-Next, (c) the combination. Board: ARC-131 (this spike),
new cards for the Flash-Next spike and the TTT build.
