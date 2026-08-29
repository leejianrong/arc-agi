# ARC-AGI RL/Evolutionary Agent: Plan

Status: MVP shipped (V1-V4 all landed - see `SLICES.md`; PRs #1-#6)

## Problem

The current repo only has a non-learned geometric-transform baseline
(`legacy/`) and a half-built, never-finished supervised program-synthesis
scaffold (`research/arc-ngps/`) — neither learns from experience, and neither
lets you *watch* an agent attempt an ARC-AGI-1 task. There's no way to see a
policy actually improve over training, or to watch it "play" a puzzle
step-by-step the way a human would in ARC-AGI's own testing interface.

The goal is a from-scratch RL (and, as a fast-follow, evolutionary) agent
that edits an ARC grid one action at a time, plus a local tool to watch both
the training run's progress and any individual episode's playthrough.

## Solution

Run `train.py --algo ppo --task_id <id>` and watch a run directory fill up
under `runs/<run_id>/` with metrics and periodic evaluation episodes. Open
the local visualizer, see a reward/success-rate dashboard update as training
progresses, pick any logged episode, and watch the agent's grid change one
DSL action at a time with play/pause/step/speed controls — the same
grid-and-palette look as ARC-AGI's own human testing interface, but showing
the agent's moves instead of a person's clicks.

## Users and actors

Solo user (repo owner), for personal research. Non-human actors: the
training process (PPO or genetic-programming trainer) and, later, a trained
policy being replayed. No conflicting-actor scenarios — single user, single
decision-maker.

## Scope

**In this milestone.**

- A Gymnasium-style ARC environment exposing a curated subset of `arc-dsl`
  primitives (ADR-0001) as a discrete action space.
- A same-shape-only task subset for the first slice, extended to
  variable-output-shape via a scratch canvas + commit/crop action (ADR-0002)
  by the third slice.
- A hand-rolled PPO trainer (ADR-0004) with dense delta-shaped reward
  (ADR-0005).
- A genetic-programming trainer over the same DSL/env (ADR-0003), as the
  evolutionary fast-follow.
- JSONL/CSV trajectory and metrics logging to local `runs/<run_id>/`
  directories (ADR-0006).
- A local TypeScript + Canvas web visualizer: step-by-step episode replay,
  and a training-metrics dashboard (ADR-0007).
- `re-arc`-generated task variations as an expanded training curriculum.

**Out.**

- ARC-AGI-2 (this project targets ARC-AGI-1 only, per the vendored dataset).
- Kaggle/private-test-set submission tooling — the private test set isn't
  available to us; success is measured against the public
  training/evaluation splits.
- Distributed or multi-GPU training — this machine is CPU-only (16 cores, no
  CUDA), and nothing here needs a cluster.
- Any LLM-in-the-loop approach — deliberately out of scope by the project's
  own premise (RL/evolutionary, not language-model-based).
- Neuroevolution (CMA-ES/ES) — documented as a future option (ADR-0003) if
  genetic programming underperforms, not built this milestone.
- Behavior-cloning warm-start of RL from GP-found programs — the shared
  trajectory format (ADR-0006) is built to make this possible later, but it
  is not a slice in this milestone.
- Deployment anywhere beyond `localhost` on this machine.

## Requirements

| ID | Requirement | Status |
| --- | --- | --- |
| R0 | Demonstrate measurable, non-zero learning improvement on a bounded ARC-AGI-1 task subset via an RL and/or evolutionary agent editing the grid step-by-step, visualizable end to end | Core goal |
| R1 | Gymnasium-style ARC env exposing `arc-dsl` primitives as a discrete action space, with scratch-canvas + commit/crop for variable output shape | Must-have |
| R2 | Hand-rolled PPO trainer with dense delta-shaped reward, training one dedicated policy per task at solve-time (ADR-0008), logging to `runs/<run_id>/` | Must-have |
| R3 | Genetic-programming trainer over the same DSL/env, sharing the trajectory log format | Must-have |
| R4 | Local TypeScript web visualizer: step-by-step episode replay (play/pause/step/speed) | Must-have |
| R5 | Training-metrics dashboard (reward curve, success rate) tailing `runs/<run_id>/metrics.jsonl` | Must-have |
| R6 | Task-exact-match accuracy computed identically regardless of which split (`data/training` or `data/evaluation`) a `task_id` comes from — a metric capability exercised whenever a run is pointed at evaluation-split task IDs, not a dedicated slice | Nice-to-have |
| R7 | `re-arc`-generated task variants usable as an expanded, difficulty-tunable curriculum for PPO training | Nice-to-have |
| R8 | GP-found solving programs are loggable in a format reusable as future RL behavior-cloning demonstrations | Nice-to-have |

## Shape

| Part | Mechanism | ADR |
| --- | --- | --- |
| S1 | ARC Gymnasium-style env wrapping vendored `arc-dsl` primitives as actions, 30×30 scratch canvas + commit/crop, task loader over `third_party/ARC-AGI` + `third_party/re-arc` | ADR-0001, ADR-0002 |
| S2 | JSONL trajectory logs + JSONL/CSV metrics + `run_meta.json` (with `schema_version`) under `runs/<run_id>/` | ADR-0006 |
| S3 | Hand-rolled PPO trainer: rollout buffer, GAE, clipped surrogate objective, dense delta-shaped reward, one policy trained per task at solve-time, color-embedding + conv/attention encoder with a factored action head | ADR-0004, ADR-0005, ADR-0008 |
| S4 | Genetic-programming trainer: population of DSL-program ASTs, crossover/mutation, fitness = train-pair match / pixel distance, same env/executor as S1 | ADR-0003 |
| S5 | Local visualizer: backend serving `runs/` as JSON, TypeScript + Canvas frontend for replay + metrics dashboard | ADR-0007 |

## Affordances

**UI.**

| Affordance | Place | Wires to |
| --- | --- | --- |
| Run picker | Visualizer home | Scans `runs/` on disk |
| Training dashboard (reward curve, success-rate curve, task breakdown) | Visualizer, per-run view | Polls `runs/<run_id>/metrics.jsonl` |
| Episode replay (grid canvas, play/pause/step/speed) | Visualizer, per-episode view | Loads `runs/<run_id>/episodes/<episode_id>.jsonl` |

**Non-UI.**

| Affordance | Kind | Wires to |
| --- | --- | --- |
| `train.py --algo ppo\|gp --task_id <id> [hyperparameter flags]` | CLI command | S1 env, S3/S4 trainer, S2 logging |
| ARC env module | Library / Gym interface | `third_party/arc-dsl` executor, `third_party/ARC-AGI` + `third_party/re-arc` data |
| Visualizer backend | Local HTTP server | `runs/` directory tree |

## Implementation decisions

- `arc_env/` package: the Gymnasium-style env, the DSL action wrapper around
  vendored `third_party/arc-dsl` primitives (ADR-0001), the scratch-canvas/
  commit mechanism (ADR-0002), and the task loader (reads
  `third_party/ARC-AGI/data/{training,evaluation}` and, once R7 lands,
  `third_party/re-arc`-generated variants).
- `trainers/ppo/`: rollout collection, GAE, clipped-objective update step
  (ADR-0004), consuming the shared reward function (ADR-0005). Trains one
  dedicated policy per `task_id` at solve-time — never a policy shared
  across tasks — using the color-embedding + conv/attention encoder and
  factored action head described in ADR-0008. No representation-pretraining
  stage (e.g. an autoencoder/VAE) precedes this; the encoder is learned
  end-to-end from the PPO reward signal (ADR-0008).
- `trainers/gp/`: population management, crossover/mutation over DSL-program
  ASTs, fitness evaluation reusing the same reward function and env/executor
  (ADR-0003).
- `runs/<run_id>/`: `run_meta.json` (config, algorithm, `schema_version`),
  `metrics.jsonl`, `episodes/<episode_id>.jsonl`, `checkpoints/` — written by
  both trainers in the same shape (ADR-0006).
- `viz/backend/`: reads `runs/` and serves it as JSON over local HTTP; no
  write path, no database.
- `viz/frontend/`: TypeScript, Canvas-based grid renderer following
  `third_party/ARC-AGI/apps/js`'s palette conventions (ADR-0007); consumes
  the backend's JSON, which mirrors the JSONL schemas 1:1 so no translation
  layer is needed beyond parsing.
- `third_party/arc-dsl/`, `third_party/re-arc/`, `third_party/ARC-AGI/`:
  vendored, read-only, not modified in place. All MIT-licensed except
  `ARC-AGI` (its own upstream license, already vendored as-is).
  `numpy`/`torch` (BSD-style) and `gymnasium` (MIT) are the only other
  Python runtime deps; **install torch's CPU-only wheel explicitly**
  (`--index-url https://download.pytorch.org/whl/cpu`) rather than the
  default PyPI wheel — the default pulls a CUDA build that cost `arc-ngps`
  6.9GB of disk for a machine with no GPU (see repo-cleanup history).
- `legacy/` and `research/arc-ngps/` are not deleted by this plan, but are
  off the path to the shipped agent (ADR-0001) — a later cleanup pass, not a
  slice here, can decide their final disposition.

## Testing approach

- **Env/executor correctness**: replay all 400 of `arc-dsl`'s known-correct
  solver programs through our env's executor; each must reproduce its
  task's expected output exactly. This is a strong, free regression test
  since the ground truth already exists (ADR-0001).
- **PPO sanity**: a smoke test that a single per-task policy (ADR-0008) can
  learn a single trivial single-action task (e.g. `vmirror`) within a small,
  fixed step budget — guards against a silently-broken hand-rolled
  implementation (ADR-0004), not a claim about ARC-AGI-1 accuracy.
- **GP sanity**: a smoke test that genetic programming finds a program
  matching a trivial task's train pairs within a bounded generation budget.
- **Visualizer replay**: a fixture `episodes/*.jsonl` file renders the
  expected sequence of grid states (golden/snapshot test on the frontend).
- **Dashboard aggregation**: a fixture `metrics.jsonl` produces the expected
  reward/success-rate curve data the frontend consumes.

Per-slice test plans (with concrete end-to-end acceptance criteria) live in
`SLICES.md`.

## Assumed defaults

| ID | Assumed | Cost if wrong |
| --- | --- | --- |
| Q1 | Solo user, no multi-actor conflicts | Low — nothing in the design assumes single-user; would just need auth/multi-run isolation added later |
| Q3 | Task/run identity via `task_id` (ARC filename stem) + timestamped `run_id`; JSONL trajectory schema as described in ADR-0006 | Medium — a schema change means a `schema_version` bump and a visualizer compatibility shim, already planned for |
| Q4 | All state on local disk under `runs/`, no database | Low — flat files remain inspectable/diffable either way; a DB migration would be additive, not corrective |
| Q5 | No concurrency control needed (independent `run_id` directories, no shared mutable state) | Low — true unless we later want two trainers writing into the *same* run, which isn't planned |
| Q7 | Invalid/out-of-bounds actions become a no-op with a small penalty; episodes hard-terminate at a max-step budget | Medium — if this makes exploration too easy/hard, only the env's step-handling code changes, not the trainers |
| Q9 | CPU-only training on this machine, confirmed no CUDA available | Low — confirmed by direct check (`torch.cuda.is_available()` unreachable, no `nvidia-smi`), not a guess |
| Q10 | Success = task-exact-match accuracy on the scoped subset, calibrated against the ~20% brute-force-search floor (not expected to be met this milestone) | Low — a measurement convention, not a design constraint |
| Q11 | No secrets/PII anywhere in this project | Low — true by the nature of the data (public ARC-AGI-1 puzzles) |
| Q12 | `schema_version` field in `run_meta.json` covers future log-format migration | Low — the whole point of adding it now is to make this cheap to revisit |

## Open risks

All five below were written before V2-V4 existed to reveal their outcome;
each now carries what actually happened, appended rather than rewritten.

- **PPO may show no measurable learning even on the scoped subset.** The one
  existing ARC RL paper (ARCLE) needed human-demonstration behavior cloning
  to make PPO tractable at all; we have no such dataset. Reward shaping
  (ADR-0005) mitigates but may not fully solve this. Earliest slice to reveal
  it: V2.
  **Resolved in V2**: it does learn. `tests/test_train_ppo.py`'s PPO-sanity
  check (mean eval reward over the last 10% of updates strictly beats a
  100-episode random-policy baseline) passes, and single-action tasks
  converge in well under two minutes of wall-clock training (README `What
  actually works right now`). Not yet tested past this 16-task subset.
- **The scratch-canvas + commit/crop action for variable output shape might
  be awkward in practice** (e.g., large unproductive exploration of canvas
  space before painting anything useful). Earliest slice to reveal it: V3.
  **Confirmed in V3, still open**: neither trainer reliably solves tasks
  needing `commit`'s exact four-argument combination. The actual failure
  mode is narrower than feared — not unproductive exploration, but a reward
  cliff: similarity scores zero for any wrong-shape output, so there's no
  gradient until the agent stumbles onto the right dimensions by chance
  (README). Getting the shape right first is the concrete open problem, not
  addressed by any slice yet.
- **Genetic programming over ~150 primitives may need real constraint/typing
  enforcement to avoid combinatorial explosion**, since there's no existing
  benchmark to calibrate population size/generation budget against. Earliest
  slice to reveal it: V4.
  **Did not materialize in V4**: GP's shipped design — a flat list of
  (action, argument) pairs over the same 23-action curated space PPO uses,
  not arc-dsl's full ~150 primitives, and no AST/typing to enforce (ADR-0003,
  `trainers/gp/genome.py`) — sidesteps this by construction. GP finds
  solutions in single-digit milliseconds on anything the action space can
  express in a handful of steps (README).
- **TypeScript frontend build tooling adds setup overhead** relative to a
  Python-only dashboard — a one-time cost, surfaced immediately in V1.
  **Paid in V1**: one-time setup cost as expected; not a live risk since.
- **Per-task training cost scales linearly with the number of tasks
  attempted** (ADR-0008) — solving N tasks means N independent training
  runs, which bounds how large V2/V3's curated task subset can practically
  be on this CPU-only machine. Earliest slice to reveal it: V2.
  **Confirmed and accepted**: this is why the curated subset is capped at 16
  tasks (`arc_env/task_loader.py`), not a defect to fix.
