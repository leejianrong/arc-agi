# ADR-0009: GP-to-PPO behavior-cloning warm-start, opt-in via `--warm_start_from`

- Status: Accepted
- Date: 2026-08-29
- Deciders: repo owner, via conversation 2026-08-29

## Context

ADR-0003 documented behavior-cloning warm-start of PPO from GP-found solving programs
as a future option, made possible by both trainers sharing one trajectory log format
(ADR-0006): "GP-found programs are a documented future option as behavior-cloning
demonstrations to warm-start the RL policy... out of scope for this milestone's
slices, but the shared trajectory format is what makes it possible later without
rework." `docs/research/rl-evolutionary-survey.md` (§1, §4) found this is exactly the
mitigation ARCLE (the one existing purpose-built ARC RL paper) needed to make PPO
tractable at all: sparse/hard-to-reach reward alone stalls PPO, and they fixed it
with behavior cloning from a human-demonstration dataset we don't have. GP is a free
substitute for that dataset on any task GP happens to solve, and GP already finds
solving programs in milliseconds on the tasks it can solve at all.

Whether to wire this in as an always-on pipeline stage, an opt-in flag, or a
continuous auxiliary loss was an open architectural fork - resolved by the repo owner
in the 2026-08-29 conversation in favor of the smallest mechanism first, consistent
with how V1-V4 each started at the smallest viable slice before growing.

## Decision

**Opt-in only, one-time supervised pretrain phase, same-task only.**

- New flag: `train.py --algo ppo --task_id <id> --warm_start_from <gp_run_dir>`,
  where `<gp_run_dir>` is an existing `runs/<run_id>/` produced by a prior
  `train.py --algo gp --task_id <id>` run **for the same `task_id`** - cross-task
  warm-starting is not this decision; ADR-0008's per-task training scope is
  unchanged.
- Mechanism: before the normal PPO loop starts, load that GP run's
  `episodes/best-program.jsonl` trajectory (ADR-0006's schema - the same shape
  PPO's own eval episodes already produce, so no new parsing logic beyond what
  `viz/backend/server.py` already has), and run a small number of supervised
  pretraining epochs: for each logged `(grid_before, action)` step, minimize
  cross-entropy between the policy's factored action distribution (primitive head,
  then typed-argument heads - ADR-0008 point 2's network) and the demonstrated
  action. The value head is left untouched by this phase (a demonstration trace has
  no logged returns/advantages to regress a critic against).
- After pretraining completes, the standard PPO rollout/update loop proceeds
  unmodified.
- Omitting `--warm_start_from` leaves `train.py --algo ppo` byte-identical to
  today - this is strictly additive.
- GP's best-found program need not have reached `exact_match` to be used - a
  partial/incomplete GP attempt is still a directionally useful demonstration, and
  gating on "GP fully solved it" would rule out warm-starting on exactly the harder
  tasks where GP itself struggles most and PPO could most use the head start.

## Alternatives considered

| Option | Why not (this decision) |
|--------|--------------------------|
| Always-on pipeline: `train.py --algo ppo` automatically runs a small, bounded GP search first, warm-starting whenever GP solves it | Couples the two trainers' CLI paths together and spends a GP budget on every PPO invocation whether or not warm-starting turns out to help - a bigger commitment than this still-unproven idea calls for as a first slice. Worth revisiting once the opt-in mechanism is proven. |
| Continuous BC-auxiliary-loss / DAgger-style mixing throughout PPO training, not just a one-time pretrain | More powerful in principle (keeps pulling the policy back toward the demonstration as it drifts) but meaningfully more complex to implement and tune correctly, for an idea with no existing evidence yet in this repo that even a simple pretrain helps. |
| Don't build this at all right now | Rejected - the reward-shape fix (see this ADR's sibling amendment to ADR-0005) may reduce how badly PPO needs this, but it doesn't eliminate ARCLE's underlying finding that RL benefits from demonstration data when available, and the mechanism is cheap relative to the payoff if it works. |

## Consequences

- No changes to `train.py --algo ppo`'s default (no-flag) behavior.
- Requires new code, not yet built: loading a GP run's episode trace outside
  `viz/backend/server.py`'s existing read path (or reusing it), and a supervised
  pretraining loop against `trainers/ppo/network.py`'s factored action heads. This
  ADR records the design decision, not the implementation - a future slice
  (`SLICES.md`, not yet written) should scope the concrete build plan and test
  criterion.
- A reasonable end-to-end test for that future slice: BC-pretrained PPO on a
  GP-solvable, commit-requiring task (e.g. `d10ecb37`) reaches nonzero eval
  reward/success measurably faster (fewer updates, same rollout/update budget) than
  cold-start PPO on the same task - a relative, not absolute, pass/fail criterion,
  mirroring V2's PPO-sanity test pattern (`PLAN.md` Testing approach).
- If GP hasn't been run for a task, or found nothing better than a trivial/near-zero
  fitness program, `--warm_start_from` still loads whatever trajectory is on disk;
  callers are responsible for pointing it at a GP run worth cloning from - this ADR
  doesn't add a fitness-quality gate.

### Implementation (2026-08-31)

Landed as designed: `trainers/ppo/warm_start.py` (`load_demonstration`,
`pretrain_from_demonstration`) plus `train.py --algo ppo --warm_start_from
<gp_run_dir> [--warm_start_epochs --warm_start_batch_size --warm_start_lr]`.

- `load_demonstration` reuses `viz.backend.server.read_episode` verbatim to parse
  `<gp_run_dir>/episodes/best-program.jsonl` - no new parsing logic, per this ADR's
  original wording. Each logged step's `action.name`/`action.args` (the *decoded*
  values `arc_env.env.ArcEnv.step` logs, not the raw `Discrete(RAW_ARG_RANGE)`
  values the policy's arg heads predict over) are mapped back to a raw value via
  the exact inverse of each `ArgSpec.kind`'s `decode` function. `color`/`factor`
  decode (`raw % 10` / `2 + raw % 3`) is many-to-one, so the inverse picks the
  canonical smallest raw value in that decode's preimage - any raw value in the
  same preimage decodes identically, so this loses no reachable behavior, only
  fixes one arbitrary representative among equivalent raw encodings for the
  policy to imitate.
- The supervised pretrain loss turned out to need no new network code: the target
  action's negative log-probability under the network's existing factored
  `Categorical` distributions is exactly what `ActorCritic.get_action_and_value
  (grid, action=target)` already computes for PPO's own ratio calculation (arg-head
  masking by the target primitive's arity falls out of the same code path for
  free) - `pretrain_from_demonstration`'s loss is literally `-sample.log_prob.
  mean()` over that call, no bespoke cross-entropy head needed.
- Pretraining uses its own short-lived Adam optimizer, separate from the main PPO
  optimizer constructed immediately afterward - keeps PPO's own optimizer state
  clean (no residual pretrain-phase momentum) rather than reusing one Adam
  instance across both phases.
- `--warm_start_from` is validated against the target run's `run_meta.json`
  (`train.check_warm_start_compatible`): must be an `algo: "gp"` run whose
  `task_ids` is exactly `[task_id]`, erroring via `argparse` otherwise. Combining
  `--warm_start_from` with `--resume_from` skips the pretrain phase entirely (the
  resumed checkpoint's weights would immediately overwrite it anyway).
- End-to-end test (`tests/test_warm_start_e2e.py`) uses `67a3c6ac` (`vmirror`), not
  this ADR's suggested `d10ecb37` example: `d10ecb37`/`5bd6f4ac` need `commit`'s 4
  raw args to land exactly right *simultaneously* for the crop to match at all -
  `arc_env.actions.execute`'s bounds check is all-or-nothing, so there's no partial-
  credit gradient toward "almost right" args for random GP mutation to climb, and
  it did not reliably solve either commit-based task within a fast test's budget in
  practice (verified empirically, not merely suspected). `vmirror` is GP-solvable in
  a handful of generations and sufficient to demonstrate the mechanism; across 3
  seeds, 5-update BC-pretrained PPO reached mean eval success rate 1.0 vs.
  cold-start's ~0.0-0.1 on the identical budget.

### Selection-channel fix (2026-09-05)

The implementation above landed with a known, documented gap: `_pad_grid`'s
selection channel (channel 1 of `ArcEnv`'s ADR-0011 observation) was always
zero, because the logged trajectory only carries `grid_before`/`grid_after`,
not the env's selection state. At the time, no curated task's solver used
`select_*`/`commit_selection` together with `--warm_start_from`, so this was
inert rather than silently wrong. `docs/PLAN.md`'s 2026-09-05 warm-start
experiment made it live: `1f85a75f` (`select_by_color` + `commit_selection`)
and `23b5c85d` are both selection-based programs used as real
`--warm_start_from` demonstrations.

Fixed by threading the previous step's logged `"selected"` field
(`arc_env.episode_log.EpisodeWriter.step`'s `"selected"` - the *post*-step
selection state) through `load_demonstration`'s loop over
`episode["steps"]`, so each step's reconstructed observation carries the
selection state as it actually was *before* that step ran: the prior step's
`"selected"` value, or `None`/empty for the first step (episodes always
start with nothing selected, `ArcEnv.reset`'s `self._selected = None`).
`_pad_grid` now takes that selection list and renders it into the mask the
same way `arc_env.env`'s private `_selected_mask` does, duplicated locally
per this module's existing convention rather than imported across modules.
No change to `pretrain_from_demonstration`, `_encode_action`, or the
public `load_demonstration` signature.
