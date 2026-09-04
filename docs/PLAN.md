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
  is not a slice in this milestone. Design decided (ADR-0009, 2026-08-29:
  opt-in `--warm_start_from` flag, same-task only) but not yet implemented —
  a future slice, not part of V1-V4.
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
  **Shape half fixed, position half still open (KAN-1178, 2026-09-05)**:
  `arc_env/reward.py`'s `SHAPE_MATCH_CREDIT` amendment (2026-08-29) gave
  `commit`'s `height`/`width` args a real gradient — Manhattan distance
  between current and target shape, smooth and separable per argument — so
  GP/PPO can now hill-climb those two dimensions. `row`/`col` never got the
  same treatment: once shape matches, `similarity` falls back to raw
  cell-for-cell content match, which is flat/noisy across wrong positions
  for an arbitrary ARC grid (no reason a 3x3 crop one cell off should share
  more pixels with the target than one five cells off) — confirmed
  empirically on `5bd6f4ac` (`crop(I, (0, 6), (3, 3))`, the one curated task
  neither trainer has ever solved): sampling every valid `(row, col)` with
  `height`/`width` pinned at the correct 3x3 gives similarity in a flat
  0.44-0.61 band everywhere except the single correct cell, which spikes to
  1.0. That makes `(row, col)` a genuine needle in a ~900-combination
  haystack (`RAW_ARG_RANGE=30` per axis) with no partial credit to search
  on. `trainers/gp/genome.py`'s `mutate` already resamples one gene
  argument at a time (not all four at once) — this was already the design
  from GP's first commit, not a gap KAN-1178 found — and it doesn't help
  here: a lucky correct `row` isn't fitter than an incorrect one on its own
  (flat landscape), so selection can't lock it in independently of `col`.
  A 25x larger search budget (`population_size=1000, n_generations=500` vs.
  the standard `200`/`100`, 3 seeds) still found 0% exact matches, plateauing
  at 0.71-0.73 similarity — evidence this is a genuine search-space-size
  problem given the current reward signal, not something a bigger budget or
  a finer-grained mutation operator fixes on its own. A real fix would need
  a smarter reward/search signal for spatial position specifically (e.g.
  cross-correlation-style partial credit, or seeding candidate crop
  windows from the input directly) — out of scope for a "does the search
  operator need tweaking" investigation, and not attempted here.
  **A different GP zero-success task, `ea32f347`, turned out to be a
  different failure mode (KAN-1179, 2026-09-05)**: its known-correct 5-step
  program (`replace(5,4)`, `select_largest`, `recolor_selected(1)`,
  `select_smallest`, `recolor_selected(2)`) *does* have a real, monotonic
  partial-credit gradient — replaying successive prefixes gives similarity
  0.0 → 0.34 → 0.34 (unchanged; selecting doesn't touch the grid) → 0.79 →
  0.79 → 1.0 — so this is not `5bd6f4ac`'s flat-landscape problem. The
  actual cause is a deceptive local optimum: a structurally unrelated,
  easier-to-find single action, `replace(5,1)` (equivalently `switch(1,5)`),
  scores a *higher* immediate similarity (0.4486) than the true program's
  own first step (0.34), purely by coincidence of which raw colors happen to
  overlap this task's diff cells — sampling 2000 random programs found only
  0.15% beating 0.34 at all, and the single best score among them belonged
  to this decoy, not any prefix of the real solution. Because the decoy is
  reachable in one gene (no select+color combo needed) and higher-scoring,
  tournament selection converges the whole population onto decoy lineages
  (`replace`/`switch` plus opportunistic `fill_cell` patchwork) within the
  first ~5 generations — the actual baseline run's `best_similarity` jumps
  to 0.45 by generation 5, then creeps to only 0.53 over the remaining 95.
  From then on the true program's lower early partial fitness can't win a
  tournament against the incumbent decoy lineage, so no selection pressure
  ever favors rebuilding the correct sequence from scratch — "premature
  convergence to a deceptive local optimum," not a missing gradient. (The
  decoy is also a genuine dead end on its own terms: each train pair's diff
  mask is only 12-16 cells out of 100, but `max_program_length=6` caps a
  `fill_cell`-patchwork strategy to retargeting at most ~6 cells directly,
  so it can never reach exact match that way.) A secondary, compounding risk
  is real but not the root cause: a successful ordinary `"transform"` action
  clears the current selection (`arc_env/actions.py`'s `execute()`, ADR-0011)
  — confirmed directly by inserting `identity` between `select_largest` and
  `recolor_selected(1)` and observing the latter turn invalid — so a stray
  mutation/crossover-inserted transform between a `select_*` and its paired
  `recolor_selected` silently breaks that segment, whenever a correct-path
  lineage does get a foothold. A 25x larger GP budget
  (`population_size=1000, n_generations=500`, 3 seeds) reaches exact match
  in 2 of 3 seeds (generations 67 and 140) but the third plateaus at 0.79
  for the full 500 generations — meaningfully better than the standard
  budget's 0%, but not reliable, and not adopted as a new default (it would
  5-25x compute cost across all 30 curated tasks, 26 of which already solve
  in single-digit milliseconds at the standard budget — KAN-1183's full-pass
  numbers below). Isolating the
  selection-pressure lever alone (`tournament_size` 3 → 2, population still
  200) reaches the 0.79 plateau more often (2/3 seeds vs. 0/3) but never
  exact match even with 3x more generations (300) — population breadth, not
  generation count or selection pressure alone, is the effective lever. No
  code change made; a real fix needs a search mechanism that doesn't let a
  single scalar similarity score be dominated by a same-shape decoy (e.g.
  novelty search or explicit diversity preservation), not a budget or
  selection-pressure tweak.
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
  **Confirmed and accepted**: this is why the curated subset is deliberately
  capped rather than grown without a design pass each time
  (`arc_env/task_loader.py`) — 16 tasks at V2/V3, 24 after ADR-0010 Phase 1,
  29 after ADR-0011/0012's object-selection menu, 30 after ADR-0013's two
  more `objects(...)` connectivity variants — not a defect to fix.
- **A new curated action doesn't automatically mean PPO learns to use it.**
  Not one of the original five (added here since it surfaced only once
  ADR-0010/0011/0012's task-coverage-scaling work began, past V1-V4).
  ADR-0011/0012's object-selection mechanism landed cleanly by every design/
  verification measure (a solvers.py audit, unit tests, curated-task
  regression tests), but a full 26-task training pass (2026-08-31, README
  `What actually works right now`) shows PPO scoring a flat 0% on both of
  ADR-0011's new object-selection tasks across the whole run, and ADR-0012's
  own small validation runs reproduce the same pattern on its new fixture
  tasks (`25ff71a9`, `ea32f347`) — GP finds a working program instantly for
  `25ff71a9`, but not for `ea32f347` (GP 0%, see KAN-1179 below). **Open**:
  whether this is a training-budget problem (these
  runs use the same short update budget as every other task, not tuned per-
  task) or something more structural about how a freshly-added, rarely-
  successful action gets discovered by on-policy exploration. The concrete
  next step to try is ADR-0009's opt-in GP-to-PPO warm-start, since GP
  already has a working program for all but one of the tasks PPO currently
  fails outright.
  **Update (2026-09-05):** Tried against all 8 same-task zero-PPO-success
  tasks a training pass had surfaced (`0d3d703e`, `1f85a75f`, `23b5c85d`,
  `5614dbcf`, `b1948b0a`, `c8f0f002`, `d511f180`, `25ff71a9` — the two other
  zero-PPO tasks, `5bd6f4ac`/`ea32f347`, are excluded since GP fails on
  those too, so there's no demonstration to warm-start from). A fresh
  same-task GP run supplied the demonstration for each (100% success in
  every case), then PPO trained for the identical 25-update budget that
  produced the original flat-0% baseline, warm-started via
  `--warm_start_from`. **Result: warm-start helps, but doesn't reliably fix
  the problem.** Two tasks (`1f85a75f`, `d511f180`) reach ~100% success and
  hold it. Three more (`23b5c85d`, `5614dbcf`, `b1948b0a`) reach substantial
  partial success (44-82% depending on how it's measured) but never fully
  stabilize. The remaining three (`0d3d703e`, `c8f0f002`, `25ff71a9`) end at
  0% on the literal final-update success rate — though two of those
  (`c8f0f002` up to 86%, `25ff71a9` up to 45% at some update) show real, if
  unstable, success mid-training that decays away by the final checkpoint,
  the same "regressed away from a working policy by final checkpoint"
  pattern already seen in the un-warm-started 26-task pass (README). Only
  `0d3d703e` — the longest demonstration (6 steps) and the one with the
  worst warm-start pretrain fit (final behavior-cloning loss 1.82, versus
  0.13-0.87 for every other task) — looks like a genuine non-improvement.
  There's no clean split by whether the task's solution uses the
  object-selection actions: the cleanest full win (`1f85a75f`) and one of
  the weakest results (`25ff71a9`) are both selection-based programs;
  program length and how tightly an action's arguments must be pinned (an
  exact color pair for `switch`, an exact direction for `move_selected`)
  look like better predictors of instability than "selection or not."
  **Still open**: the one-time pretrain phase (ADR-0009's chosen design,
  versus the continuous BC-auxiliary-loss alternative it explicitly
  deferred) gets several tasks to a working policy during training but
  doesn't reliably keep them there — a periodic re-anchoring to the
  demonstration, or simply selecting the checkpoint with the best observed
  eval success rather than always the final update, are the next things
  worth trying before concluding warm-start alone is the wrong lever.
  **KAN-1177 update (2026-09-05):** investigated the "regressed away from a
  working policy by final checkpoint" pattern directly (`6fa7a44f`,
  `9172f3a0`, `a416b8f3`, `d10ecb37` un-warm-started; `c8f0f002`, `25ff71a9`
  warm-started, all reaching 17-86% at some intermediate update then 0% by
  the last one). Reproduced 3 of the 6 (`6fa7a44f`, `a416b8f3` un-warm-
  started; `c8f0f002` warm-started) at the standard config (`--n_updates 25
  --rollout_steps 128 --eval_every 5 --re_arc_prob 0.5 --max_steps 25
  --seed 0`). **Finding: this is largely a metric-definition artifact, not
  policy collapse.** Every one of those percentages comes from
  `metrics.jsonl`'s `success_rate` field, which is `train.py`'s mean of
  `RolloutBuffer.episode_successes` — a handful of stochastic-policy
  episodes (`n_episodes` ranged 5-30 across the reproductions) drawn from
  that update's own shifting mix of re-arc-generated instances (a random
  difficulty band *per episode*, `make_next_pair_fn`) and native train
  pairs. It is not a measurement of the policy's competence on the fixed
  pair `log_eval_episode` actually replays with the *greedy* policy every
  `eval_every` updates. Checking that fixed-pair outcome separately (the
  `episodes/eval-update*.jsonl` traces) across all 3 reproductions, it
  never regressed — e.g. `c8f0f002`'s eval pair was solved (`success=True`)
  at every logged checkpoint (updates 0, 5, 10, 15, 20, 24) even as its
  rollout `success_rate` swung from 0.80 (update 23, `n_episodes=25`) to
  0.00 (update 24, `n_episodes=5`) one update later. `entropy`/`approx_kl`/
  `clip_frac` at that exact swing were unremarkable (entropy 1.49→1.56,
  approx_kl 0.005→0.006, clip_frac 0.02→0.04) — no destructive update, no
  entropy collapse — ruling out the policy-collapse and destructive-update
  hypotheses for these 3 cases. Nor did training on generated variants pull
  the policy away from the literal eval pair: it stayed solved throughout.
  **Caveat**: only 3 of the 6 originally-cited tasks were reproduced, and
  this doesn't rule out genuine fixed-pair eval regression elsewhere — the
  three still-unstable warm-start tasks above (`23b5c85d`, `5614dbcf`,
  `b1948b0a`) and non-improving `0d3d703e` remain open questions this
  investigation didn't re-examine. **Fix applied (low risk, additive,
  `train.py`)**: `metrics.jsonl` rows now also carry `eval_success`/
  `eval_reward` — the fixed-pair greedy-policy outcome, previously only
  visible by opening the separate per-checkpoint episode trace — so this
  metric confusion is easier to catch by eyeballing `metrics.jsonl` alone
  next time. `train_ppo` also now tracks a best-eval-reward-so-far
  checkpoint (`checkpoints/best.pt`, selection logic in
  `is_new_best_eval`, unit-tested in `tests/test_train.py`) as a safety net
  for cases where the fixed-pair eval genuinely does regress — on these 3
  reproductions `best.pt` ended up identical to the final checkpoint every
  time (since none of them actually regressed on the fixed pair), so there
  is no before/after improvement to show here; it's a diagnostic and
  safety-net change validated end-to-end
  (`tests/test_train_ppo.py::test_best_eval_checkpoint_is_tracked_separately_from_the_noisy_rollout_stat`),
  not a demonstrated fix for the three still-unstable warm-start tasks.
  **KAN-1183 update (2026-09-05): first comprehensive, current-config,
  all-30-task pass since this whole thread of investigation started** (one
  plain, non-warm-started PPO run + one GP run per task, `--seed 0` on both,
  the standard budgets documented in README's `What actually works right
  now`). **Aggregate numbers**: GP fully solves 26/30 (87%) — the same two
  total failures as before (`5bd6f4ac`, `ea32f347`, see above) plus two
  tasks — `46f33fce` and `d10ecb37`, both already part of the original
  26-task set and reported solved there — now landing on partial credit
  across their own train pairs at this run's `--seed 0` (33% and 67%
  respectively) instead of the 100% the 2026-08-31 pass recorded. No
  `trainers/gp/` code changed between the two passes (KAN-1178/KAN-1179
  were docs-only investigations, no fix applied), so this reads as GP's
  stochastic search landing in a worse local optimum at this particular
  seed rather than a regression — plausible given the standard budget
  (`population_size=200, n_generations=100`) is exactly the scale KAN-1178
  showed can plateau below 100% on a hard fitness landscape — but not
  separately re-investigated here; worth a multi-seed rerun if GP
  reliability at the standard budget becomes its own question. PPO fully
  solves 12/30 (40%) by `eval_success` — and,
  notably, `eval_success` was identical at the final logged update and at
  the best-ever logged update for all 30 tasks with no exception, the
  cleanest evidence yet for this section's KAN-1177 finding that the
  earlier "regressed away from a working policy by the final checkpoint"
  pattern was a `success_rate` sampling artifact, not real policy collapse:
  in a full run at this scale, zero tasks show real eval-pair regression.
  Reinforcing that same point from the other direction, the older
  `success_rate` metric never once hit literal 100% for any of the 30
  tasks at any logged update in this pass, which would read as "PPO fully
  solved 0/30" if `success_rate` were still (wrongly) treated as the
  solved/not-solved signal — exactly the metric-definition trap KAN-1177
  fixed `eval_success` to avoid. **Both of ADR-0011's original
  object-selection tasks remain unsolved by plain PPO** (`1f85a75f`,
  `23b5c85d`, `eval_success` false for both) — still consistent with this
  bullet's original finding — **but with a wrinkle**: neither is *flat*
  0% this time the way the 26-task pass reported (`1f85a75f`'s
  `success_rate` is 67%, `23b5c85d`'s is 28%), i.e. partial rollout success
  now shows up along the way even though the fixed-pair eval still never
  locks in. Of PPO's 18 `eval_success` failures, GP fully solves 14 of them
  (everything except its own four non-100% tasks above) — the same
  warm-start target list as before, just re-confirmed at the current
  30-task scale; the three still-unstable warm-start tasks (`23b5c85d`,
  `5614dbcf`, `b1948b0a`) and non-improving `0d3d703e` are all present
  again in this plain (non-warm-started) baseline's failure list, consistent
  with warm-start being the open, not-yet-reliable mitigation for them
  rather than something this baseline pass would fix on its own. Full
  per-task table in README's `What actually works right now`.
