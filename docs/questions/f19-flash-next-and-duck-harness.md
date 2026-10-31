# F19 - Flash-Next and the Duck harness (2026-10-31)

Follows `f18-test-time-training-spike.md`. Raw reports: `docs/research/arc2-flash-next-spike/` (Kaggle side, web side).
The ARC-AGI-3 session (`~/projects/arc-agi/arc-agi-3/solve-arc-agi-3`) also answered a direct question about Duck.

## Findings

**The model we already run is much stronger than we have been using it.** ARC Prize's results page for Qwen3.8-27B
(https://arcprize.org/results/alibaba-qwen3-8-27b): ARC-AGI-2 semi-private **42.4% at xhigh effort ($0.447/task)**, low
22.8%, medium 13.2%, no thinking 1.5%. We ran medium (or the unset default) with 24-40K token caps. The public 27B agent
notebooks that do use xhigh (temperature 1.0, top_p 0.95, top_k 20, `thinking_token_budget` 40960, `max_tokens` 49152)
score only 15-27 on Kaggle, so the gap to 42 is tokens per task and throughput, not the model.

**Duck and the leaders.** Duck (github.com/Tufalabs/duck-harness, `ARC3-Inference/`, no LICENSE file) is an ARC-AGI-3
harness: one sandboxed `python(code)` tool over a persistent REPL, observations as variables plus an image and a
letter-coded ASCII grid, 4-connected segmentation, oldest messages evicted. Tufa Labs is #1 on the ARC-AGI-2 board
(88.06); forum claims say the leaders use Flash-Next plus Duck. Unverified for ARC-AGI-2, and no leader notebook is
public. A public 27B notebook (`madarshbb/arc-agent-sub-q38-direct-b98k-c16-792ef9`, credits dfranzen's Duck fork)
already contains a REPL agent mode: `python(code)` and `submit(code)` tools, `submit` runs `transform` on every demo and
returns a cell diff, verified programs end the attempt, a nudge every 6 python calls without a submit, context trimming
that keeps prefix-cache hits, per-attempt token cap = throughput x time / tasks, voting over verified programs, an
8192-token no-think rescue turn.

**Flash-Next is a runtime change, not a drop-in.** MoE, 125B total, 6B active (+51B n-gram table, 4B MTP layer), licence
qwen-community-1.0 (hosts declined to rule on prize eligibility; hearsay says ARC Prize allowed it). vLLM needs >= 0.30
(we pin 0.19.0); the most downloaded Flash-Next stack on Kaggle is an SGLang wheelhouse
(`dfranzen/pennyroyal-v253`), models are 125-186 GB, the n-gram table lives in host RAM (about 50-110 GB), a 4x3090 recipe
reports 108-117 tok/s single stream but KV for only 3-6 concurrent 40K sequences. Nothing verified on L4/SM89. No public
Flash-Next notebook exists, and one user reports 8.61 for Flash-Next vs 26.81 for plain 27B.

**From the ARC-AGI-3 session (measured by them):** their loop is a clean-room single-step tool-calling agent, not Duck's
REPL; thinking on did not help *there*; failure modes that cost them time: prose-only replies (70% fallbacks at first),
silently inapplicable tunables, tool-call `index` fields; reusable pieces: `sandbox.py` (restricted builtins, AST filter,
module proxies, rlimits, escape regression tests), `world_model.py`/`planner.py` (replay-check a program against
transitions), `step_trace.py` + replay viewer, `observation.py` (`segment`, `describe_objects`, `describe_change`),
launchers, and the semula + Haiku loop (`docs/SEMULA-HAIKU-LOOP.md`). Qwen3.6 on an RTX PRO 6000: 39 tok/s single,
270 aggregate at 8 streams, 780 at 32.

## Decision

The user prefers the Duck-style method over TTT. Plan:

1. **Build a Duck-style REPL agent mode for ARC-AGI-2** on Qwen3.8-27B (reimplemented, MIT-0; do not copy Duck code or
   prompts): persistent sandboxed `python` tool with the task preloaded as variables and helper functions, `submit`
   verified against the demos with a cell diff, nudges, context trimming, token budget from measured throughput.
   Reuse our verified-program loop, scheduler and observability. Verify mechanics with semula + Haiku (tool calls).
2. **Run at xhigh with a large thinking budget** and measure tokens per task in the single Kaggle/RunPod integration
   runs; size concurrency and per-task budget from measured throughput.
3. **Hold Flash-Next** (runtime unverified, 125+ GB models, licence ruling absent) and **hold TTT** (ceiling about 35
   public, about a day to build, licence of reused code unclear) as fallbacks; revisit after step 1 has numbers.
