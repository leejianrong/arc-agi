# ARC-AGI-2 on Kaggle: RunPod, vLLM and iteration notes

What we learned getting an offline Qwen3.8-27B-FP8 solver running (2026-10-30/31). Facts below were observed
unless marked *unverified*. Code lives in `kaggle/arc2_baseline/` (stdlib baseline, first submission, public
score 0.00) and `kaggle/arc2_synth/` (direct-answer solver, program-synthesis solver, RunPod and Kaggle drivers).

## Competition constraints

- Kaggle code competition, internet off, no hosted-API models. Deadline 2026-11-02 23:59 UTC; Paper Track 2026-11-09.
- One submission per day, plus a pending limit: a new one cannot start while the last is still scoring.
  Queue waits of 15-24 h and "Kaggle Error" were reported on the L4 pool. A submission re-runs the notebook, so a
  long notebook means a long wait; submit the real run early.
- Hardware per hosts: L4x4 (96 GB GPU memory), 30 h/week interactive quota, L4x4 bills at 2x. Runtime limit
  not confirmed (public notebooks use about 11.5 h); we budget 9-10 h.
- Prize and Paper rules: open-source code before private scores; code we author CC0/MIT-0, third-party code needs a
  permissive licence. Qwen3.8-27B-FP8 is Apache-2.0 (Hugging Face), unlike Flash-Next (Qwen Community License).
- Top public notebooks (31-34%) use per-task LoRA test-time training on a pre-fine-tuned small Qwen. Deferred (ADR-0030).

## Kaggle runtime facts (from our probes, also in the ARC-AGI-3 repo's checklist)

- Default Python is 3.13, but the vLLM wheelhouse (`mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`, Apache-2.0)
  is cp312. `pip install --no-index` on 3.13 fails with "No matching distribution found for aiohttp==3.13.5".
  Fix: `uv venv /tmp/vllm-venv --python /usr/bin/python3.12`, then `uv pip install --no-index --find-links <wheelhouse>
  -r requirements.lock`. Use `/tmp`; `/kaggle/working` is only 20 GB.
- Mounts: `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/`, `/kaggle/input/datasets/<owner>/<slug>`,
  `/kaggle/input/models/<owner>/<model>/pytorch/<variant>/<n>`. The notebook finds dirs by content, not path.
- `kernel-metadata.json` `machine_shape` values seen: `NvidiaRtxPro6000`, `NvidiaL4`, `NvidiaTeslaT4`. **`NvidiaL4X4` is
  accepted by the CLI but silently ignored: the kernel got 2x Tesla T4** (16 GB, no FP8/bf16), so the 27B model
  OOMed. The public ARC-AGI-2 notebooks all use `machine_shape: NvidiaL4`, which is the competition's L4 pool.
  Always check the `gpus:` and device name in the kernel log.
- Cold-mount cost: `uv pip install` of the wheelhouse took 13.6 min on a cold mount (the ARC-AGI-3 notes saw 22 min
  once); budget it inside the run.
- Wheelhouse pins: vllm 0.19.0, torch 2.10.0 (cu128), transformers 4.57.6, triton 3.6.0, flashinfer 0.6.6.
  Third-party datasets can vanish (one was re-created after deletion); keep private mirrors if time allows.
- Kaggle credentials here are OAuth tokens (`~/.kaggle/credentials.json`). Never copy them to a rented pod.

## vLLM serving Qwen3.8-27B-FP8 on SM89 (RTX 4090 and L4 are both Ada)

- Flags that worked: `--tensor-parallel-size 4 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN
  --no-async-scheduling --reasoning-parser qwen3 --max-model-len 65536 --gpu-memory-utilization 0.94`, with env
  `VLLM_FORCE_ATTN_BACKEND=TRITON_ATTN` (the MTP drafter does not inherit the CLI flag) and
  `PYTORCH_ALLOC_CONF=expandable_segments:True`. FP8 KV needs Triton on SM89; FlashInfer FP8 attention is SM90+.
- **`/health` is not readiness.** MTP with 6 speculative tokens passed `/health`, then returned HTTP 500 on the first
  real requests and shut down. MTP=2 passed an 8-request generation smoke and ran with mean acceptance length about 2.8.
  Always smoke-test with real generation before trusting a config (`kaggle/arc2_synth/smoke.py`).
- Without MTP, 16 concurrent sequences gave about 410 tok/s aggregate (about 25 tok/s each); with MTP=2 and 32
  sequences about 890 tok/s (about 21-28 each). KV cache was only 45% full at 32 sequences, so 48+ is possible.
- Startup on a fresh pod: install vLLM 0.5-2.5 min, weights load about 1.5 min, then torch.compile and CUDA graph
  capture 5-17 min depending on the host (cold cache). Plan 9-25 min before the first request. Kaggle's RTX PRO 6000
  runs reached model-ready in 9-11 min.
- **Reasoning effort dominates the score.** ARC Prize's own results page for Qwen3.8-27B (run on Baseten) lists ARC-AGI-2:
  xhigh 42.4% ($0.447/task), low 22.8%, medium 13.2%, no thinking 1.5%. Our runs used medium (the public notebook's
  `reasoning_effort: medium` default) or the unset default, so we were operating at a fraction of the model's ability.
  The price is tokens: xhigh thinks far longer, so throughput per token budget (concurrency, MTP, hardware) is the lever.
  Source: https://arcprize.org/results/alibaba-qwen3-8-27b. The page says Qwen's chat template adds effort-specific
  instructions for low and xhigh but not medium, which may explain medium scoring below low.
- Reasoning length is the central problem. Default effort, 32K cap: 22 of 24 tasks never finished thinking; the two
  that did (24.7K and 30.5K tokens) were solved. Medium effort, 24K cap: 31 of 32 hit the cap. Plan for 40K+ tokens
  or a different method.
- `chat_template_kwargs: {"enable_thinking": false}` does not make Qwen3.8 answer briefly; it reasons in the visible
  reply and was cut off at 4096 tokens with no JSON. `reasoning_effort` (low/medium/high) is the supported knob.
- Budget forcing works through `/v1/completions` with raw ChatML: user turn, `<think>`, the partial reasoning, a "time
  is up" line, `</think>`, then the opening of the JSON or code block (`llm_solve.force_answer`, `synth.force_code`).
  *Unverified on real vLLM at time of writing*; the 40K run (below) is meant to confirm it.
- The reasoning trace comes back in `message.reasoning` (or `reasoning_content` on other versions); the code reads both.

## RunPod facts

- Drive pods with the `runpod-jobs` skill's `rp` wrapper. The API key lives in `~/.runpod/config.toml` as an
  unquoted `apikey = ...`; load it by reference (`awk`), never print it.
- 4x RTX 4090 secure is about $3.56/hr (community about $1.36). Stock changes by the hour; when another pod holds
  the cards, `rp up` fails with "no instances currently available". 2x L40S ($2.18/hr, TP=2) worked as a stand-in.
  RunPod has no L4 and no 4x A100 in reach on this account's budget.
- Image used: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, 120 GB container disk (Kaggle-style downloads keep
  zip plus extracted files; Hugging Face downloads need about 30 GB). Image pull took 1 min on one host and 6 min on
  another. vLLM and the model come from PyPI and Hugging Face (ungated) so no Kaggle secret is needed on the pod.
- The vLLM docker image was not used: its entrypoint is `vllm serve`, and `rp` sets `dockerStartCmd` without an
  entrypoint override.
- **`rp up` reuses a pod with the same name, even one in its 30 s shutdown grace period, and silently ignores the
  new job command.** Wait for `runpodctl pod list` to be empty, and give concurrent pods distinct `RP_POD_NAME` and
  `RP_STATE_FILE`.
- The pod has no public IP: `rp logs` shows container stdout only (tail), so the job must echo what you need
  (vLLM progress is in a file; the job greps and prints it). Artifacts come back with `runpodctl send` and
  `rp relay-get` using the `RP_ARTIFACT_CODE` line; fetch within the grace window after `RP_ARTIFACT_CODE` appears.
- Files reach the pod by embedding a base64 tarball in `RP_JOB_CMD` (about 13 KB for the solver bundle).
- **Cleanup bug that cost a run (about $2.30):** after a vLLM config fails, its worker processes outlive the API
  server and keep GPU memory (13.6 of 23.5 GiB free), so the next config fails with "Free memory ... less than
  desired GPU memory utilization". `pod_job.sh` now `kill -9`s everything vllm and waits for GPU memory to drain.
  Also persist crash traces: the artifact keeps only the last lines, which are dominated by 500s.
- Shared account: another Claude session (ARC-AGI-3) also starts pods. Check `runpodctl pod list` names; never delete
  pods that are not yours. Balance and spend rate: `runpodctl me`.
- Hosts that failed us: 2x L40S with TP=2 hung twice at NCCL init ("vLLM is using nccl", then silence for 15 min);
  a single L40S (TP=1, no NCCL) served the 27B FP8 model fine. One 4x4090 host needed 30+ min of torch.compile with no
  readiness and was torn down. On a slow host, abort early instead of waiting out the timeout.
- Cost so far this effort: about $15 for four direct-answer runs and one synthesis run. Typical run: boot 10-25 min
  (billed) plus 30-60 min of solving.

## Iteration strategy: spend RunPod only on questions that are worth $4

RunPod runs are slow (25-90 min) and billed, so every run must answer a question written down beforehand, with the
cheap checks already done. Tiers, cheapest first:

1. **Unit tests with a fake server** (seconds, free): `tests/test_kaggle_*.py`. Catches parsing, sandbox, selection,
   deadline and merge bugs.
2. **Semula plus a Haiku answerer on real ARC-AGI-2 tasks** (minutes, Haiku tokens only): the real harness, real
   prompts and real feedback rounds against a real LLM. Finds prompt, format, feedback and parsing problems and
   measures prompt sizes. It cannot tell us Qwen's speed, thinking length or solve rate. See "Semula workflow".
3. **RunPod** for what only Qwen can answer: thinking length versus budget, throughput, verified-program yield,
   the forced-answer path, MTP stability. Prepare the next candidate with tier 2 while a pod runs.
4. **Kaggle** (queue plus quota) for what only Kaggle can answer: offline install, driver, real L4 speed.

Rules: one written hypothesis per pod run; pre-build and syntax-check the bundle; always use the smoke test; cap
lifetime to the question; fetch crash traces; prefer replaying recordings over re-asking.

## Semula workflow (fake OpenAI server answered by Haiku)

- Our instance: `cd ~/projects/abang-ai/semula && SEMULA_HOST_PORT=18081 SEMULA_MODE=record docker compose -p
  semula-arc2 up -d --wait semula` (container `semula-arc2-semula-1`, `http://127.0.0.1:18081/v1`). The ARC-AGI-3
  project runs its own on port 8080; do not share containers, so recordings do not mix.
- Point `synth.py` or `llm_solve.py` at it with `--base-url http://127.0.0.1:18081/v1 --model haiku --effort ""
  --request-timeout 900`. Unknown prompts park until answered; answers are stored and replayed, so a rerun with an
  unchanged prompt is free and deterministic (and replays old wrong answers too: use a fresh project to start clean).
- Answerer: a Haiku sub-agent loops `docker exec semula-arc2-semula-1 semula pending --wait 60 --json`,
  `semula show <id> --json`, `docker exec -i ... semula answer <id> --stdin` with `{"message": {"content": "..."}}`.
  Tell it to use only the request contents. Cap it (about 40-50 answers, stop after 3 empty waits); its context grows
  with every call. Repo policy: at most 2 sub-agents at once, none may spawn sub-agents.
- `semula pending` shows only the first 200 characters of the last message, so put a task tag at the start of every
  user message to let two answerers split tasks without racing.
- Semula returns no `usage`, so token counters read 0 in these runs.
- First result: 4 easy ARC-AGI-1 tasks, 4 of 4 verified and solved, but all on round 1, so the refine loop was
  not exercised. Use ARC-AGI-2 tasks, where Haiku will fail some and trigger feedback rounds.

## Observability: know within minutes if a pod is wasting money

A rented pod has no public IP and `rp logs` shows only container stdout, so we build visibility in.

- **Pod side (`pod_job.sh`):** every 60 s during startup and every 120 s while solving, an `HB` line reports Hugging
  Face cache size, per-GPU memory and utilisation, how old the vLLM log is, and how long nothing has changed.
  `STALL?` appears after 5 min of no change; after `STALL_SECS` (default 720) the config is aborted and the next one
  is tried. A config is accepted only after a burst of real generation requests (`smoke.py`). Crash traces from every
  failed config are saved in the artifact (`out/crash_*.log`, `out/vllm_*.log.tail`).
- **Driver side (`watch_pod.py`):** run one per pod under the Monitor tool
  (`python3 -u kaggle/arc2_synth/watch_pod.py <env file> <label> --interval 60 --expect-min <N>`). It prints one line
  per event: `ALERT` (traceback, VLLM_DIED, SMOKE_FAILED, 500s, OOM, STALL?, connection refused, no space left),
  `FEED` (log stopped advancing for 5 min), `GONE` (pod disappeared: fetch the artifact), task and phase lines,
  throttled `HB`/`VLLM` digests, and a cost estimate every 10 min. It also alerts if the pod outlives `--expect-min`.
- **What healthy looks like (RTX 4090 x4, 27B FP8):** install done within 3 min; `TRYING config`; GPUs show 100%
  utilisation at about 660 MiB while NCCL initialises (normal); HF cache grows to about 28 GB; then compile and
  CUDA-graph capture; `SMOKE [True x8]`; `PHASE=vllm_up=1`; task lines. No vLLM log growth for 5+ minutes outside
  the NCCL phase is not normal.
- **Rules:** never launch without a watcher armed; treat any `ALERT`/`FEED`/`GONE` as a stop-and-look event; read the
  crash artifact before relaunching; when a pod stalls past its expected phase time, tear it down rather than wait.
- **Gap:** no remote shell (sshd is not started because `rp` overrides the container start command), so we cannot
  inspect a live pod beyond what the job prints. Anything we might need to see must be printed by the job.

## Kaggle integration run that worked (2026-10-31, kernel v3, `machine_shape: NvidiaL4`)

First end-to-end run of the generated notebook on the real competition hardware (`kaggle/arc2_synth/launcher.py.tmpl`,
synth mode, 4 tasks, 8192-token calls, MTP=2):

| Phase | Time into run |
|---|---|
| baseline submission written | +0.0 min |
| `uv venv` (py3.12) | +0.0 min |
| `uv pip install` of the wheelhouse (cold mount) | **+22.9 min** (the 4 GPUs were visible only after this) |
| vLLM boot, MTP=2 (no retry needed), 8-request smoke passed | +31.2 min |
| 4 tasks solved and verified | +35.0 min (225 s of solving, 23,355 tokens) |
| submission validated: 240 tasks, 0 repaired | +35.0 min |

- 4 GPUs of about 21 GB used each; **about 41 tokens/sec per sequence at 4 concurrent requests** (measured from the
  calls, with MTP=2), aggregate about 104 tok/s at that concurrency. Better than the 4090 RunPod pods at 16-32 sequences.
- Startup eats about 31 min of the 12 h budget; most of it is the cold wheelhouse install. Budget for it.
- The competition's `arc-agi_test_challenges.json` mounted in the kernel contains 240 tasks whose first sorted ids are
  ARC-AGI-1-style tasks (for example 007bbfb7), so "verified" there says nothing about ARC-AGI-2 difficulty.
- Not yet exercised on Kaggle: the forced-code path, the MTP=0 retry, higher concurrency, long thinking.

## Real-Qwen synthesis on RunPod (1x L40S, medium effort, 32K cap), 2026-10-31

`synth.py` against Qwen3.8-27B-FP8 on 16 public-eval tasks (first 16 by id), TP=1, MTP off, 16 concurrent calls, 3000 s
deadline, no scheduler floor: **1 of 16 verified (`1818057f`, one round, 20K tokens) and that program solved its test
input (1 of 28 test inputs)**. 414K completion tokens in 3000 s, about 138 tok/s aggregate (about 8-15 tok/s per
sequence; the KV cache hit 99.6% at 16 sequences, which causes preemption, so one 48 GB card cannot hold 16 long
sequences).

- 11 tasks used a whole 32-35K-token first call; each was cut off, the **forced-code path produced a program anyway
  (confirmed working on real vLLM via `/v1/completions` with raw ChatML)**, and every forced program failed the demos.
  The second round found no time left. 3 tasks never reached the model.
- Reading: medium effort with a 32K cap leaves almost every ARC-AGI-2 task unsolved, in line with ARC Prize's
  13.2% (medium) vs 42.4% (xhigh). More tokens per task (xhigh, bigger caps) and faster serving are the levers, which
  is what the REPL-agent measurement (ARC-136) is for.

## Pre-flight gate

Run `make arc2-preflight` (or `uv run python scripts/arc2_preflight.py [--fast] [--modes direct,synth,agent,bench]`) before pushing any notebook to Kaggle. It runs ruff, the Kaggle tests, builds each mode and checks `kernel-metadata.json` (L4, no internet, private, one competition source, wheelhouse and Qwen sources) and `main.py` (compiles, no `@@` placeholders, under 1 MB, bundle holds the expected modules), then dry-runs each built notebook against a fake OpenAI server and requires a validated 3-task `submission.json`. It stops at the first failure and ends with `PREFLIGHT OK` or `PREFLIGHT FAILED: <step>`; `--fast` runs only the launcher, agent and scripted-player tests. It cannot see real vLLM, the real Kaggle mounts or the wheelhouse install.

## Dev run vs scored rerun

The notebook (`launcher.py.tmpl`) checks `KAGGLE_IS_COMPETITION_RERUN`, which Kaggle sets only when it
re-executes a submitted notebook on the hidden test set. Pushing the kernel or running it interactively
does not set it.

- Scored rerun: solves every test task into `submission.json` with the baked budget (log line `RERUN: full run over N test tasks`).
- Anything else (dev run): solves the first `--dev-tasks` (default 4, env `ARC_DEV_TASKS`) public-eval tasks
  by sorted id within `--dev-budget-min` (default 20, env `ARC_DEV_BUDGET_MIN`) into `dev_submission.json`,
  then logs `DEV score: solved A of B test inputs (pass@2) ...` (or `DEV: unscored` without the solutions file).
  `submission.json` stays the valid baseline for the test tasks. Agent mode still writes `trace.jsonl`.
- Both modes write the baseline for the test file first and validate/repair `submission.json` at the end.
- `ARC_FORCE_RERUN=1`/`0` overrides detection (tests, dry runs). Bench mode ignores all of this.

## Pre-submission checklist (a zero score is almost always setup, not model quality)

1. `make arc2-preflight` passes (builds every mode, dry-runs the launcher, checks bundle size).
2. Kernel metadata: `machine_shape` is `NvidiaL4`, internet off, model + wheelhouse sources attached, kernel version pinned.
3. The launcher writes the baseline `submission.json` first, then overwrites it only with a validated full answer set.
4. Run kind: the scored rerun is detected by `KAGGLE_IS_COMPETITION_RERUN` **or** by the mounted test file not matching the public
   placeholder's id fingerprint (`PLACEHOLDER_SHA`). The log line `run kind: ...` states which. If the placeholder is ever
   replaced upstream, set `ARC_PLACEHOLDER_SHA` (or re-fingerprint) before submitting.
5. Total budget at most ~11h (`deadline_min = total - elapsed - 15`); the session limit is ~12h.
6. Real-vLLM dev run of the exact build: a normal Save & Run (not a rerun) solves the first `--dev-tasks` public-eval tasks,
   scores them against the public solutions, and leaves `submission.json` as the baseline. Check it scores at least what direct
   did on the same tasks (6/16 on the first 16) with no crash, then submit that same kernel version, not a rebuild.
7. One submission per day plus queue delays: submit early enough that a failed run can still be retried before the deadline.

## Hybrid mode (safe floor for a scored run)

`build_kernel.py --mode hybrid [--hybrid-direct-frac 0.45]` (env `ARC_HYBRID_DIRECT_FRAC`): phase 1 runs the direct solver over
all tasks for that share of the budget (`deadline_min = total - elapsed - 15`), writing `submission.json` as it goes; phase 2 runs
the agent on the rest (agent flags such as `--max-turns`, `--agent-args` apply here), least-sure direct answers first (no parsed
answer, then a forced answer), skipped when under 1 minute remains. Both phases stop at the same absolute deadline. Per test input
the merge (`synth.select_hybrid`) puts a verified, non-plausibility-flagged agent output in attempt_1 with the direct answer as
attempt_2; otherwise the direct answer stays attempt_1 and attempt_2 is the agent's best other output. Dev runs use the same flow on the eval slice.
