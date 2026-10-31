# Flash-Next spike report (2026-10-31)

Short answer: no leader notebook is public. The leaders' recipe is only partly visible. Visible pieces: Franzen's SGLang stack, the Duck harness, and forum remarks.

## 1. Qwen3.8-Flash-Next
- HF card (Qwen/Qwen3.8-Flash-Next): ~125B language-model params, ~6B active per token. Plus 51B n-gram embedding params and a 4B MTP layer; the safetensors listing shows 180B.
- Architecture: 48 layers, 3 Gated DeltaNet blocks to 1 sparse-attention block, each followed by a 512-expert MoE (10 routed + 1 shared). 262k native context. License qwen-community-1.0.
- Kaggle variants (uncompressed bytes):
  - dfranzen/intel-qwen3.8-flash-next-w4a16-autoround, 181 GB (66 votes). BF16 PLE table included.
  - dfranzen/albucino-...-w4a16-fp8ple, 125 GB.
  - nvidia/qwen3-8-flash-next-nvfp4, 133 GB.
  - dineshkumar0705/...-fp8, 186 GB.
  - sethmoudry/...-fp8-int4-experts, 8 parts of ~19-21 GB.
  - Drafters: dfranzen/albucino-...-drafter (4.2 GB, MTP head), pramodgupta15/...-dflash2-drafter (1.2 GB, Apache-2.0 DFlash2).
- Fit on 4x24GB (96 GB VRAM): the byte sizes above exceed it. I infer the huge n-gram/PLE table must sit in host RAM or be hot-cached, but could not verify. Franzen's dataset ships `hot_tokens_64k.pt`, which fits that reading. The FP8PLE variant exists "reducing PLE memory usage".
- Runtime: dfranzen/pennyroyal-v253 (6.6 GB, **65,850 downloads**, 4 votes, 2026-09-29) is a wheelhouse of **SGLang 0.5.19 (custom build)**, torch 2.13+cu130, flashinfer 0.6.17, flash-attn-4, transformers 5.12.1, humming-kernels, tokenspeed-triton, sglang-kernel cu130. It contains no vllm wheel. The top-downloaded stack is therefore SGLang, not vLLM. Download count is only a proxy for use.
- Our wheelhouse (vllm 0.19.0 / torch 2.10 / transformers 4.57 or 5.8 overlay) is for Qwen3.8-27B. No evidence it runs Flash-Next. Keithtyser's `qwen38-flash-next-vllm-nvfp4-runtime-v1` (7.9 GB) is a vLLM runtime for NVFP4, which is not native on L4. I did not open it.
- Throughput: Darren Amadeus Martin (topic 750546) says his team reached "a top speed of 900 tok/s", peak 980 tok/s "decode speed under harness" (output tokens / wall-clock, including tool calls and prefill). With much prefill it drops to ~250 tok/s. Average on 2 passes over the public eval in ~6 h is truncated, so unknown. Aurora reports 250-400 tps. Anil agrees: "this is the way". Adarsh found 27B and FN "almost breakeven in speed and accuracy on L4x4" in his own tests, and was contradicted by others. Our 15-25 tok/s per sequence is single-stream; the leaders count aggregate over concurrent sequences.

## 2. Top public notebooks
There is no public Flash-Next notebook. Searches for flash-next, duck, harness, sglang, and pennyroyal found none. The public best are Qwen3.8-27B-FP8:
- madarshbb/arc-qwen-3-8-27b-agent (51 votes). Adarsh reports ~15, later 21.5 (topic 748082). Sibling madarshbb/arc-agent-sub-q38-direct-b98k-c16-792ef9 (22 votes) scored 20-24. Hiểu Vy reports 26.81 for plain 27B and 8.61 for Flash (no details).
- Their design, from the downloaded notebook:
  - vLLM TP=4, wheelhouse mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3 plus the transformers 5.8 overlay from keithtyser/taaf-duck-qwen38-serving-v1. That is the same stack we pin.
  - Flags: `--kv-cache-dtype fp8`, `--speculative-config {"method":"qwen3_5_mtp","num_speculative_tokens":2}`, `--reasoning-config` (think start/end), `--max-num-seqs 16`, max_model_len 81920, prefix caching, qwen3_coder and qwen3 parsers. A ladder of fallback option sets is tried if MTP fails.
  - Published config is "mode: direct": the model writes the grids itself, 2 samples per task (= the two attempts), `reasoning_effort: xhigh`, temp 1.0, top_p 0.95, top_k 20, thinking_token_budget 40960, max_tokens 49152, concurrency 10, and an 8192-token no-think "rescue" turn for cut replies.
  - Deadline = start + 12h - 35 min, and a valid fallback submission is written first.
  - The notebook also has a REPL agent mode (`repl_agent.py`): tools `python(code)` and `submit(code)`. Submit runs `transform` on all demos and returns a cell diff. All demos passing means verified and the attempt ends. Other pieces: a "submit early" nudge every 6 python calls, a thinking-cap hint, context trimming with hysteresis for prefix-cache hits, and per-attempt token cap = throughput x time / tasks. Answers are voted over verified programs, with the gap filled by best unverified.
  - It credits "dfranzen's ARC-AGI-3 fork of the duck harness". So Franzen, a leading competitor and the model/runtime packager, forked Duck.
- Other public notebooks are NVARC-lineage Qwen3-4B TTT, scoring ~28-34. cosmorin's "Why #1 is #1" says the 88 is "most likely a training-side number". It is explicitly inference and contradicts Anil and the others.

## 3. Duck harness
- Tufa Labs' github.com/Tufalabs/duck-harness ("ARC3-Inference"), a harness for ARC-AGI-3. Our local doc `duck-control-v1.md` pins revision 7652836. Its license file is absent (claims MIT in the pyproject classifier).
- Design: one ephemeral sandboxed `python` tool (30 s timeout, output capped at 1,024 tokens), segmentation helpers, an `action()` callback, thinking on, temp 0.6, 32k context, qwen3_coder parser, prefix caching.
- The README is ARC-AGI-3 only. Its default config is Qwen3.6-27B-FP8 on B200 via Slurm. The Kaggle bundle `keithtyser/taaf-duck-qwen38-serving-v1` contains the full source (tool_agent.py, prompts.py, kaggle.py). No ARC-AGI-2 variant is public.
- Anil (topic 750546): "many of the leading teams are using Qwen3.8-Flash-Next alongwith the Duck harness". Tufa Labs is also #1 on ARC-AGI-2. So on ARC-AGI-2 it is the same family adapted (python tool and submit/verify), not a distinct harness. How it was adapted is unverified.

## 4. Licensing and instability
- Topics 752709 (12 votes), 750575, 746307: the host (María Cruz) declined to interpret third-party licenses. A teammate-style reply (Mark Barney) reports that the ARC Prize team confirmed on Discord that Flash-Next is allowed for competition use, with Greg Kamradt confirming. This is hearsay, and no official written ruling is on the forum.
- The OSI-open-source question for prize winners and the Paper Track is unresolved. 752709's analysis cites rules 2.5.a, 2.8, and 1.5, which require open-sourcing the solution. The model is used unmodified, so linking to the weights is the arguable path.
- Instability:
  - 753315 "Unstable GPU server": inter-GPU speed up to 10x slower on some L4x4 hosts, decode 3x slower, and "newer and faster submissions scored lower".
  - 747515, 753659, 753674, 753723, 751344: queue delays of 19+ hours or more, Kaggle Error, 0.00 scores, a proposal for 24h+ scoring, and requests for rerun or extension.
  - 753865: a significant score drop not reproduced locally.
  - CV-to-LB correlation is poor. Darren: CV above 75% but the LB does not follow. mikelou1 reports 66% LB (~60-70 CV) with a private Qwen harness.

## 5. Recommendation
1. Cheap and likely worthwhile: copy Adarsh's configuration ideas into our vLLM launcher: MTP spec decoding (n=2), fp8 KV, reasoning-config with a thinking-token budget (~40k), a rescue turn, and concurrency 10-16 with TP=4. The aim is to get aggregate tok/s far above 15-25. Then try a REPL/submit loop with diff feedback (we already have verified synthesis).
2. Moving to Flash-Next means swapping the whole runtime. Do not try it inside vllm 0.19.0. Either attach Franzen's pennyroyal-v253 SGLang wheelhouse plus the W4A16 model and the drafter, or find a vLLM version that supports it. We have not verified that either works on L4.
3. Since Duck's source is public via the keithtyser bundle but unlicensed, reimplement from behaviour, as our ARC-AGI-3 notes already do.
4. Biggest risks:
   - The Flash-Next stack is untested on our side and the leaders' exact recipe is unknown.
   - The model is huge, so host RAM and disk, load time, and CPU offload are open questions.
   - Kaggle queue delays and slow GPU hosts.
   - Qwen license eligibility and the Paper Track (hearsay only).
   - Public 27B agents top out around 15-27, so only the Flash-Next stack plus heavy engineering has shown 70+.

## Could not verify
- Any leader's actual harness or code, or whether Flash-Next is the only difference.
- vLLM version or flags for Flash-Next; SGLang flags used. Only the wheelhouse contents are known.
- How the model fits on 4x24 GB. Throughput numbers are self-reported and unsourced.
- Contents of keithtyser's NVFP4 vLLM runtime; whether our wheelhouse supports Flash-Next.
- Official licensing or Paper Track ruling. The full text of the forum comments is truncated by the CLI.
- Public scores of Flash-Next notebooks. I found none.
