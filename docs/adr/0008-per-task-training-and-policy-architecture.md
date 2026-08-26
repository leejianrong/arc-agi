# ADR-0008: Per-task, solve-time PPO training; policy/value architecture; no representation-pretraining this milestone

- Status: Accepted
- Date: 2026-08-27
- Deciders: repo owner

## Context

Two different things were undecided after ADR-0004 fixed the RL library
choice: (1) does PPO train one shared policy across all ARC tasks, or a
dedicated policy per task using that task's own train pairs; and (2) does
the policy need a pretrained representation (e.g. an autoencoder/VAE trained
on `re-arc`-generated grids) to embed grids/tasks before PPO can use them.
These two questions turn out to be the same decision: the answer to (1)
determines whether (2) is even a sensible thing to build.

The one existing purpose-built ARC RL paper, ARCLE (`docs/research/rl-evolutionary-survey.md`),
only ever demonstrated per-task learning, not cross-task generalization — and
that matches how every other solver in this repo's history has worked:
`legacy/baseline.py`, `research/arc-ngps`'s synthesis approach, and
`arc-dsl`'s own 400 solver programs (ADR-0001) all treat each task as an
independent problem to be fit against its own train pairs.

## Decision

**1. Per-task, solve-time training.** For a given `task_id`, train a fresh
PPO policy using only that task's train pairs plus `re-arc`-generated
variations of the same task concept (ADR-0006's V2 slice) for additional
practice instances. Apply the resulting policy once to the task's held-out
test input(s) for scoring; the test input/output is never used during
training. Solving N tasks means N independent training runs, one
`runs/<run_id>/` per task, run sequentially or in parallel across this
machine's 16 CPU cores.

**2. Policy/value network.** A small, CPU-trainable network:
- Color-embed each cell (10 colors → a learned vector, dim ~16-32) over the
  30×30 scratch canvas (ADR-0002), with an explicit active-region mask
  channel since grids vary in size.
- A few conv/residual blocks plus 1-2 self-attention layers, so the network
  can pick up long-range relations (symmetry, "the other object like this
  one") that convolution's local receptive field alone would miss.
- A factored action head: pick the primitive (categorical over the curated
  action subset) first, then its typed arguments — coordinates via a
  pointer/attention lookup over the spatial feature map, colors via a small
  categorical head — matching `arc-dsl`'s typed-primitive-plus-args shape.
- A linear value head off the same pooled features for PPO's critic.

**3. No separate representation-pretraining this milestone.** No
autoencoder/VAE (or any other unsupervised pretraining stage) on
`re-arc`-generated grids. The grid encoder in (2) is learned end-to-end from
the PPO reward signal on each task, which is sufficient because the
network's job is narrow — represent this one task's handful of instances —
not general ARC perception across hundreds of unrelated tasks.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Single shared policy across all tasks, conditioned on a task embedding derived from the train pairs | Requires solving two hard problems at once — inducing an arbitrary transformation from a few examples, and executing it — with no existing RL result (ARCLE included) demonstrating this works at meaningful accuracy. A much larger, riskier bet than this milestone calls for. |
| Pretrain an autoencoder/VAE on `re-arc`-generated grids for a warm-start embedding | Solves a different problem (general ARC perception) than per-task RL needs. Reconstruction is also a weak proxy for the relational/object-level structure ARC problems hinge on (object count, symmetry, adjacency) — a known weakness, and part of why `arc-ngps`'s unfinished Perceiver-encoder approach (superseded by ADR-0001) never got fully validated. Adds a whole new training pipeline for uncertain payoff, with no current slice requiring it. |
| Multi-task RL (train one policy jointly across many tasks using the RL reward itself, no separate pretraining stage) | A legitimate future warm-start path, and cheaper to justify than reconstruction pretraining since it reuses the same reward signal end-to-end — but it's a fast-follow contingent on per-task training (this ADR) working first, not part of this milestone. |

## Consequences

- Compute cost scales linearly with the number of tasks attempted — a real
  resource constraint on the curated task-subset size for V2/V3, not just an
  implementation detail (see `PLAN.md` Open risks).
- `train.py --algo ppo` operates on one `task_id` at a time; running it over
  a set of tasks means looping or parallelizing across independent
  `runs/<run_id>/` directories.
- Cross-task generalization (a policy that solves genuinely novel,
  never-trained-on tasks in one shot) remains out of scope this milestone —
  already true per `PLAN.md` Scope, but this ADR makes explicit *why*: the
  training lifecycle described here produces a per-task solver, not that
  capability.
- If per-task PPO doesn't clearly outperform genetic programming (ADR-0003)
  or brute-force search on the scoped subset, multi-task warm-starting
  becomes the natural next architectural bet — documented here as a
  deliberate fast-follow, not a silently dropped idea.
