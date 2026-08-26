# Decision register — ARC-AGI RL/Evolutionary revamp

Mode: **Fresh** (no prior planning artifacts existed; idea captured from conversation + repo audit).

Status values: `FORK` (escalated to user), `ASSUMED` (default applied), `DEFERRED` (not needed for this milestone).

## Repo-audit facts used below
- No GPU: `torch` not even installed at root env; no `nvidia-smi`; 16 CPU cores available. Training must be CPU-tractable.
- `./ARC-AGI` is the official ARC-AGI-1 clone: `data/training`, `data/evaluation` JSON tasks, plus the human-facing `apps/testing_interface.html`. No private/Kaggle test set present or obtainable.
- `./arc-ngps` is a half-built *supervised program-synthesis* scaffold (Perceiver encoder, pair-induction, DSL AST + executor with `grid_ops.py`, beam search) — a different paradigm from RL/evolutionary, but its DSL/executor primitives are a plausible action-space library.
- Root `baseline.py`/`evaluate.py`/`arc_io.py` are a simple non-learned geometric+color-bijection baseline with matplotlib visualization — superseded by this revamp's goals but not necessarily deleted (could stay as a sanity-check baseline).

## 1. Primary user and actors — ASSUMED
Solo user (the repo owner) building/running this for personal research. No multi-tenant, no other humans. The only "actor" besides the human is the training process itself (RL/evolutionary trainer) and, later, a trained policy being replayed.

## 2. Scope boundary — FORK (task-shape scope) + ASSUMED (rest)
ASSUMED in-scope: an ARC-1 Gym-style environment with a step-by-step grid-editing action space; an RL trainer; an evolutionary trainer (fast-follow, see F3); a local visualizer for (a) training metrics and (b) step-by-step replay of an episode "playing" a task.
ASSUMED out-of-scope: ARC-AGI-2, Kaggle/private-test submission tooling, distributed/multi-GPU training, cloud deployment, multiplayer or human-in-the-loop play.
FORK: whether milestone 1 restricts to same-shape (input.shape == output.shape) tasks — see F2.

## 3. Core data model and identity — ASSUMED
- ARC tasks keep their existing `task_id` (filename stem from `ARC-AGI/data/{training,evaluation}`).
- Training runs get a `run_id` (timestamp-based directory name).
- Each episode logs a JSONL trajectory: one line per step `{step, grid_before, action, args, grid_after, reward, done}`.
- Aggregate training metrics (reward, success rate, loss) logged as JSONL/CSV, one row per episode or update.
- A `run_meta.json` per run records config, algorithm, `schema_version`.
All human-readable/diffable text formats, no binary trajectory blobs.

## 4. State and storage — ASSUMED
Everything lives under `runs/<run_id>/` on local disk: `meta.json`, `metrics.jsonl`, `episodes/*.jsonl`, `checkpoints/*.pt` (or evolutionary population snapshots). No database. Fully inspectable/diffable by the user directly.

## 5. Concurrency and conflict — ASSUMED
Single human, no simultaneous writers to the same run. Multiple independent training runs may run concurrently in separate `run_id` directories with no shared mutable state — no locking needed.

## 6. Interfaces and contracts — FORK (x3: F4, F5, F6) + ASSUMED (CLI)
ASSUMED: a CLI entrypoint to launch a training run (`python -m arc_rl.train --algo ppo --config ...`) that writes to `runs/<run_id>/` per above.
FORK: RL framework/library choice (F4).
FORK: whether the visualizer needs live streaming from an in-flight trainer, or reads logged files (F5).
FORK: visualizer tech stack (F6).

## 7. Failure behaviour — ASSUMED
Invalid/out-of-bounds actions from the policy are clipped to a no-op with a small negative reward and logged, rather than crashing the episode. Episodes hard-terminate at a max-step budget. Training crashes are recoverable via the last checkpoint in `runs/<run_id>/checkpoints/`.

## 8. External dependencies — FORK (x3: F1, F3, F4, F6 — stack choices) + ASSUMED (rest)
FORK: how much of `arc-ngps` to reuse (F1).
FORK: RL vs evolutionary build order (F3).
FORK: RL framework (F4).
FORK: visualizer stack (F6).
ASSUMED: `numpy` for grid math, `torch` for any learned policy (CPU build), no other heavy deps (no SB3 unless F4 answer says otherwise), all offline/local, all OSS/permissive licenses (numpy BSD, torch BSD-style, ARC-AGI data is the official public training+evaluation set under its own license already vendored in `./ARC-AGI`).

## 9. Runtime and deployment — ASSUMED
Single local machine (this WSL2 box), CPU-only (confirmed: no CUDA, 16 cores). Training and visualizer both run as local processes; visualizer is a local web server (`localhost`), not deployed anywhere.

## 10. Measurable success — ASSUMED
Primary metric: task-exact-match accuracy (all test grids correct) on `ARC-AGI/data/training` (learning set) and `ARC-AGI/data/evaluation` (held-out check), tracked per run in `metrics.jsonl`, same definition as the existing `evaluate.py`'s `task_acc`. Milestone-1 success is a working, visualizable, non-zero-improving-over-training pipeline on the scoped task subset (F2) — not SOTA accuracy.

## 11. Security and secrets — ASSUMED
Nothing sensitive. No credentials, no PII, no network calls beyond localhost. Nothing to log-scrub.

## 12. Versioning and migration — ASSUMED
`run_meta.json` carries a `schema_version` for the trajectory/metrics log format so the visualizer can degrade gracefully if the format changes later. Low stakes for a solo research tool.

## Forks escalated to user (see chat for the compact grill)
- F1. `arc-ngps` reuse scope (action-space DSL/executor vs. discard vs. also reuse encoder)
- F2. Milestone-1 task scope: same-shape-only vs. full variable-size ARC-1
- F3. Build order: RL first, evolutionary first, or both in parallel
- F4. RL framework: Gymnasium + Stable-Baselines3 vs. Gymnasium + custom training loop
- F5. Visualizer live-streaming vs. log-and-replay
- F6. Visualizer stack: custom web app (Canvas/JS) vs. Streamlit/Gradio
