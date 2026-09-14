# Questions — ARC-AGI RL/Evolutionary revamp

Statuses: `DECIDED` (user or delegated-research answered) · `ASSUMED` (default
taken, correct it if wrong) · `FORK` (waiting on the user) · `DEFERRED` (not
needed this milestone).

Most rows below carry their full answer inline. A few questions accumulate a
long running history instead of a single answer, an object-selection mechanism
that grew over ten ADRs, an LLM-in-the-loop experiment, a staged multi-part
build. Those live in their own file under `docs/questions/`, linked from the
row below, so this register stays an index you can scan rather than a pile of
narrative. New long-running threads should start life as their own file
rather than accreting in this table.

## Open forks

None — all forks from the 2026-08-27 round are resolved (see Register).

## Register

| ID | Question | Status | Answer or default | Landed |
|----|----------|--------|--------------------|--------|
| F1 | DSL / action-space foundation | DECIDED (delegated research: `docs/research/arc-dsl-survey.md`) | Adopt Hodel's `arc-dsl` (160 primitives, MIT) as the action space/executor; discard `research/arc-ngps`'s DSL/executor; vendor `re-arc` for training-data generation | ADR-0001 |
| F2 | Variable output-grid shape handling / milestone-1 task scope | DECIDED (delegated research, reverses original same-shape-only recommendation) | Fixed 30×30 scratch canvas + explicit commit/crop action, mirroring `arc-dsl`'s own `canvas`/`crop` primitives; Slice 1 still starts same-shape-only as a smoke test, canvas/commit added in Slice 3 | ADR-0002, SLICES.md V1 & V3 |
| F3 | Build order: RL vs. evolutionary vs. both | DECIDED (user) | RL (PPO) first; genetic programming over the same DSL as the evolutionary fast-follow; both share one env/DSL/trajectory-log format | ADR-0003 |
| F4 | RL framework: Stable-Baselines3 vs. custom loop | DECIDED (user) | Gymnasium-style env + hand-rolled PPO (no SB3) | ADR-0004 |
| F5 | Visualizer live-streaming vs. log-and-replay | DECIDED (user) | Log-and-replay via JSONL/CSV on local disk, no trainer↔visualizer IPC | ADR-0006 |
| F6 | Visualizer tech stack | DECIDED (user) | Custom local web app; TypeScript (not JavaScript) + Canvas frontend | ADR-0007 |
| F7 | Reward shaping design | DECIDED (delegated research: `docs/research/rl-evolutionary-survey.md`) | Dense, delta-based, non-background-normalized similarity reward + no-op step penalty + terminal exact-match bonus | ADR-0005 |
| F8 | Evolutionary method: genetic programming vs. neuroevolution | DECIDED (delegated research) | Genetic programming over the DSL (CPU-cheap, zero prior benchmark to beat); neuroevolution documented as a future option only | ADR-0003 |
| F9 | PPO training lifecycle (per-task vs. shared/cross-task) and whether a pretrained autoencoder/VAE embedding is needed | DECIDED (user, via conversation 2026-08-27) | Per-task, solve-time training — one fresh PPO policy per `task_id`, trained only on that task's own train pairs + `re-arc` variations, never shared across tasks. No representation-pretraining this milestone: the grid encoder (color-embedding + conv/attention) is learned end-to-end from the PPO reward signal; a reconstruction-based AE/VAE would target a different problem (general ARC perception) and is a weak proxy for the relational structure ARC needs. Cross-task generalization + shared pretraining documented as an explicit future direction, not this milestone | ADR-0008 |
| F10 | GP-to-PPO behavior-cloning warm-start: always-on pipeline vs. opt-in flag vs. continuous auxiliary loss | DECIDED (user, via conversation 2026-08-29) | Opt-in only: `train.py --algo ppo --warm_start_from <gp_run_dir>` loads a same-task GP run's best-program trajectory and runs a one-time supervised pretrain phase before normal PPO proceeds unmodified. Design decision only — implementation not yet built | ADR-0009 |
| F11 | Task-coverage scaling past the curated 16: broaden primitives vs. add object selection vs. hold at 16 vs. deepen via re-arc | DECIDED (user, 2026-08-29) - landed incrementally, 67/400 curated as of ADR-0024 | Both, sequenced: broaden primitives first, then add object selection. A 2026-09-06 follow-up concluded 400/400 is out of reach under this architecture. The 2 originally-hardest remaining tasks (`7c008303`, `a68b268e`) turned out, on a 2026-09-14 re-audit, not to need a new mechanism after all - see F14/ADR-0023. A same-day follow-up (ADR-0024) re-checked the audit's `free_by_name` bucket and landed 7 more tasks via 4 cheap geometric-tiling derived actions. Full history: [`docs/questions/f11-task-coverage.md`](questions/f11-task-coverage.md) | ADR-0010, ADR-0011, ADR-0012, ADR-0013, ADR-0015, ADR-0016, ADR-0019, ADR-0023, ADR-0024 |
| F12 | Whether/how to pursue LLM-driven approaches (program synthesis, hypothesis-testing, on-the-fly reward/primitive design) as a track toward higher ARC-AGI-1 coverage and, eventually, ARC-AGI-2/3 | DEFERRED (not needed this milestone; revisits Q2's "no LLM-in-the-loop" scope boundary for a later one) | A narrower "LLM-seeded search" refinement (an LLM proposes a curated-DSL action sequence as a second demonstration source for GP/PPO, never arbitrary code) was tried on 2026-09-06 and helped in every comparison run. Full history: [`docs/questions/f12-llm-driven-approaches.md`](questions/f12-llm-driven-approaches.md) | ADR-0018, `docs/research/llm-seeded-search-experiment.md` |
| F13 | Whether to build an interactive "Photoshop-like" editor - a human solves ARC-AGI-1 tasks by hand with a constrained toolset mirroring the curated action space, logged as a demonstration trajectory for warm-start/imitation learning | DECIDED (user, 2026-09-06) - staged and complete: Stage 0 done 2026-09-06, Stage 1 done 2026-09-13, Stage 2 done 2026-09-14 | Stage 0 built the write path into the visualizer. Stage 1 played 8 excluded tasks by hand to observe mechanism gaps. Stage 2's Buckets A and B (9 tasks, 8 new actions, no new mechanism state) landed 2026-09-13; Bucket C (2026-09-14) closed the "hold the pre-crop original grid" cluster - it turned out to need no new mechanism either (2 tasks landed as plain derived actions, 3 others a reasoned no-go). Full history: [`docs/questions/f13-interactive-editor.md`](questions/f13-interactive-editor.md) | ADR-0017, `docs/research/f13-stage1-play-audit.md`, ADR-0021, ADR-0022, ADR-0023 |
| F14 | The multi-selection/cross-grid mechanism F11 deferred: design and scope it before building | DECIDED (user, 2026-09-13) - design verified and extended, implementation landed; the originally-deferred cluster resolved 2026-09-14 without needing this mechanism | A broader audit found the real cluster needing this is bigger than F11's original 2 tasks (8 tasks sharing a "combine indices from 2 halves via a set op" shape), but a different cluster than those 2 - user picked a 2-slot, region-scoped design (`select_by_color_in_region`/`combine_slots`/`fill_new_canvas`/`fill_onto_region`) verified against all 8 target tasks by direct replay. A 2026-09-13 follow-up (ADR-0022) widened the region menu to 3-way splits and added `fill_slot_onto_region`/`replace_region_and_fill`, landing 2 more near-miss tasks. `7c008303`/`a68b268e` (the original motivating tasks) explicitly stayed out of scope at the time - a 2026-09-14 re-audit (ADR-0023) found that framing wrong: neither needed this mechanism (or any new mechanism) after all, and landed via ordinary derived actions instead. Full history: [`docs/questions/f14-multi-selection-mechanism.md`](questions/f14-multi-selection-mechanism.md) | ADR-0020, ADR-0022, ADR-0023 |
| Q1 | Primary user and actors | ASSUMED | Solo user (repo owner), personal research; no multi-actor conflicts | PLAN.md Users and actors |
| Q2 | Scope boundary | ASSUMED + F2 | ARC-AGI-1 only; no Kaggle/private test; no distributed training; no LLM-in-the-loop; see full in/out list | PLAN.md Scope |
| Q3 | Core data model and identity | ASSUMED | `task_id` (ARC filename stem) + timestamped `run_id`; JSONL trajectory schema per step | PLAN.md Implementation decisions, ADR-0006 |
| Q4 | State and storage | ASSUMED | Local disk only, `runs/<run_id>/`, no database | ADR-0006 |
| Q5 | Concurrency and conflict | ASSUMED | No locking needed; independent `run_id` directories, no shared mutable state | PLAN.md Assumed defaults |
| Q6 | Interfaces and contracts (CLI) | ASSUMED | `train.py --algo ppo\|gp --task_id <id> [hyperparameter flags]` (flat per-hyperparameter flags, no single `--config`) | PLAN.md Affordances |
| Q7 | Failure behaviour | ASSUMED | Invalid actions → no-op + small penalty; hard max-step termination; checkpoint-based recovery | PLAN.md Assumed defaults |
| Q8 | External dependencies (license, offline) | ASSUMED + F1 | `arc-dsl`/`re-arc` MIT; `gymnasium` MIT; numpy/torch (BSD-style), CPU-only wheel installed explicitly (avoids repeating `arc-ngps`'s 6.9GB CUDA-wheel mistake); all offline/local | ADR-0001, PLAN.md Implementation decisions |
| Q9 | Runtime and deployment | ASSUMED | Single local machine, CPU-only (confirmed: no CUDA, 16 cores), `localhost`-only visualizer | PLAN.md Assumed defaults |
| Q10 | Measurable success | ASSUMED + research-calibrated | Task-exact-match accuracy on scoped subset; calibrated against ~20% brute-force-search floor (icecuber) as context, not a target | PLAN.md Assumed defaults, `docs/research/rl-evolutionary-survey.md` |
| Q11 | Security and secrets | ASSUMED | None applicable — no credentials/PII, public dataset only | PLAN.md Assumed defaults |
| Q12 | Versioning and migration | ASSUMED | `schema_version` field in `run_meta.json` | PLAN.md Assumed defaults |

## Coverage

| Category | Covered by |
|----------|-----------|
| Primary user and actors | Q1 |
| Scope boundary | Q2, F2 |
| Data model and identity | Q3 |
| State and storage | Q4 |
| Concurrency and conflict | Q5 |
| Interfaces and contracts | Q6, F5, F6 |
| Failure behaviour | Q7 |
| External dependencies | Q8, F1, F8 |
| Runtime and deployment | Q9 |
| Measurable success | Q10 |
| Security and secrets | Q11 |
| Versioning and migration | Q12 |
| Reward design (domain-specific) | F7 |
| Build order / paradigm sequencing (domain-specific) | F3, F4 |
| Policy architecture / training lifecycle (domain-specific) | F9 |
| RL/GP interoperation (domain-specific) | F10 |
| Task-coverage scope past the curated 16 (domain-specific) | F11 |
| Cross-benchmark generalization strategy past the curated-DSL approach (domain-specific) | F12 |
| Human-in-the-loop demonstration collection (domain-specific) | F13 |
| Multi-selection/cross-grid mechanism design (domain-specific) | F14 |

## Repo-audit facts used throughout

- No GPU: `torch` not installed at root env at time of audit; no `nvidia-smi`;
  16 CPU cores available. Confirmed directly, not assumed.
- `third_party/ARC-AGI/` is the official ARC-AGI-1 clone (de-gitted, plain
  tracked files): `data/training`, `data/evaluation`, plus the human-facing
  `apps/testing_interface.html`. No private/Kaggle test set present or
  obtainable.
- `research/arc-ngps/` was a half-built supervised program-synthesis scaffold;
  superseded per ADR-0001, not deleted, disposition deferred.
- `legacy/` holds the original non-learned geometric-transform baseline;
  kept as a reference, not deleted.
- The two solver-corpus audits behind F11's task-coverage numbers (the
  original 2026-08-29 pass and the 2026-09-05 refresh using
  `scripts/audit_action_coverage.py`) live in
  [`docs/questions/f11-task-coverage.md`](questions/f11-task-coverage.md)
  now, not here, since they only ever supported that one decision.
