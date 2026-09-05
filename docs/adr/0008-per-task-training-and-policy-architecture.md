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

## Amendment (2026-09-05): mask the 5 `act_on_selection` primitives when nothing is selected

`ArcEnv`'s object-selection mechanism (ADR-0011/ADR-0012) has 5 actions
whose `kind` is `"act_on_selection"`: `commit_selection`, `delete_selected`,
`recolor_selected`, `move_selected`, `paint_selected_at`. Per
`arc_env/actions.py`'s `execute`, every one of these is an unconditional
no-op (`valid=False`) whenever there's no current selection. Until now,
`trainers/ppo/network.py`'s `ActorCritic` sampled the primitive uniformly
over all `N_ACTIONS` (`Categorical(logits=self.primitive_head(pooled))`)
with no awareness of whether a selection currently exists, so the policy
could — and, per a prior investigation of this repo's training runs, did —
spend rollout steps sampling one of these 5 actions with nothing selected,
always invalid, always paying the invalid-action penalty (`arc_env/
reward.py`).

This is a PPO-specific problem, not a GP one: PPO does step-by-step credit
assignment, so an action that is *always* worse than a no-op when sampled
with an empty selection gets its whole region of action space pushed down
by gradient descent — the policy learns to avoid `commit_selection`/
`recolor_selected` entirely, including the states where they're the
*correct* next action (both are required to solve several curated tasks).
Worse, the correct first half of the winning program in those cases (a
`select_*` action) reward-ties with `identity`, since selecting doesn't
change the grid — so there was never a reward gradient pulling the policy
toward discovering the `select_*` → `act_on_selection` pairing at all, only
one pushing it away from the second half. GP never hits this: it only
scores whole finished programs, never assigns credit to an individual
step, so an invalid mid-program step is just diluted into that program's
overall fitness, not specifically taught-against.

**Decision**: mask, not just penalize. The observation already carries
what's needed for free — channel 1 of the `(2, 30, 30)` observation
(`arc_env/env.py`'s `_selected_mask`) is a binary "currently selected"
mask, already an `ActorCritic._encode` input. `ActorCritic` now:

- Precomputes a module-level `IS_ACT_ON_SELECTION` boolean tensor
  (`trainers/ppo/network.py`), one entry per `actions.ACTIONS` index,
  `True` at the 5 `act_on_selection` primitives — the same "derive a
  lookup tensor from `actions.ACTIONS`" pattern `ARITY_BY_PRIMITIVE`
  already established.
- In `get_action_and_value` and `get_greedy_action`, before constructing
  `primitive_dist`/taking the argmax, computes per batch row whether
  `obs[:, 1, :, :]` is all-zero, and sets `primitive_logits` to `-inf` at
  the 5 masked indices for exactly those rows (`_mask_act_on_selection_logits`).
  Nothing else is masked — `arg_heads` are untouched, since arguments are
  only sampled/scored conditioned on the already-chosen primitive.

This turns "always worse than a no-op, so credit assignment teaches the
policy to avoid this whole region" into "literally unsampleable, so credit
assignment can no longer teach avoidance of something that was never a
comparable alternative to begin with." The environment's own invalid-action
handling in `arc_env/actions.py`/`arc_env/env.py`/`arc_env/reward.py` is
unchanged — this only restricts what the *policy* can choose to sample, not
what the environment does when it receives an action.

Verified directly (not assumed) that `torch.distributions.Categorical`
handles `-inf` logits cleanly: masked entries get exactly zero sampling
probability, `.entropy()` and `.log_prob()` on the *unmasked* entries stay
finite with no NaN leakage, and gradients through unmasked logits are
well-defined. (Scoring a masked entry as the target action does still
produce `-inf`/`inf` log-prob/loss, as expected — see the warm-start
interaction below for why this is a real ordering hazard, not a
theoretical one.) See `tests/test_network.py` for the standalone checks
this amendment adds, including that masking is applied per-row in a mixed
batch and that `get_greedy_action`'s argmax avoids a masked primitive even
when it holds the batch's single highest raw (pre-mask) logit.

**Cross-reference — interaction with `trainers/ppo/warm_start.py`
(ADR-0009)**: `pretrain_from_demonstration` calls the exact same
`ActorCritic.get_action_and_value(obs, action=target)` path this masking
lives in, to compute `-log_prob` of a demonstrated action as its
supervised loss. Before the fix in PR #32
(`fix/warm-start-selection-channel`), `load_demonstration`'s reconstructed
selection channel was always zero regardless of what was actually selected
at that step in the logged trace — so a demonstration containing a real
`select_by_color` → `commit_selection` pairing (e.g. `1f85a75f`,
`23b5c85d`) would present the `commit_selection` step's observation as
"nothing selected," and this masking would then force that step's
primitive probability to exactly zero, making its log-prob `-inf` and the
pretrain loss `inf`/NaN. This PR is deliberately branched on top of PR
#32's fix rather than merged independently, precisely to avoid landing a
change that only works correctly once the selection-channel reconstruction
is already fixed. Separately confirmed that `ppo_update`
(`trainers/ppo/ppo.py`) recomputes `log_prob`/entropy for already-taken
rollout actions by calling `get_action_and_value(mb_obs, action=mb_action)`
against the *same* stored `mb_obs` the action was originally sampled from —
since this masking is a pure function of `obs` (no external or cached
selection flag anywhere in the codebase), the mask in effect at
recompute-time is guaranteed identical to the mask in effect when the
action was sampled during rollout collection.
