# Questions — ARC-AGI RL/Evolutionary revamp

Statuses: `DECIDED` (user or delegated-research answered) · `ASSUMED` (default
taken, correct it if wrong) · `FORK` (waiting on the user) · `DEFERRED` (not
needed this milestone).

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
