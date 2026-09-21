# ADR-0030: Pivot to LLM-driven program synthesis; freeze the curated-DSL RL/GP/object tracks

- Status: Accepted
- Date: 2026-09-22
- Deciders: repo owner, via conversation (software-product-director review,
  2026-09-22)
- Supersedes: ADR-0001 (arc-dsl as the RL/GP action space), ADR-0003 (GP over
  that action space), ADR-0004/ADR-0008 (the hand-rolled PPO trainer),
  ADR-0009 (GP→PPO warm-start), ADR-0029 (the object-centric typed-grammar
  rewrite) — all as the project's active development track, not as a
  historical record. None of the superseded ADRs are retracted; they stand as
  what was decided and why, at the time.

## Context

The project's goal changed. It is no longer personal research/exploration —
it is now to (1) get the best achievable ARC-AGI performance by following
actually state-of-the-art methods, and (2) field a real, competitive entry to
the [ARC Prize 2026](https://arcprize.org/competitions/2026) competition
(ARC-AGI-2 and/or ARC-AGI-3 prediction tracks on Kaggle), including the
[Paper Track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track).

A 2026-09-22 review (`software-product-director` skill) audited the repo
against that new goal and found three disqualifying facts, all confirmed
directly, not assumed:

1. **The Paper Track is not standalone.** Per the official rules, an entrant
   must first hold "a valid submission that you previously provided for
   either the ARC-AGI-2 or ARC-AGI-3 prediction competitions," submitted as a
   public, open-sourced Kaggle notebook, scored against the hidden set. The
   paper documents that submission — it isn't a separate essay track. This
   repo has zero ARC-AGI-2/3 presence (`find . -iname "*arc-agi-2*"` returns
   nothing) and no Kaggle submission tooling of any kind. `docs/QUESTIONS.md`
   Q2 explicitly scoped both out at project start. **There is currently no
   path to entering either underlying competition**, which means no path to
   the Paper Track either, independent of research quality.
2. **The action space's core growth mechanism cannot run at scoring time.**
   Task coverage (91/400 ARC-AGI-1 training tasks, per `CLAUDE.md`'s own
   `arc_env/` bullet) grew across ~20 ADRs, each one a human (or an agent)
   tracing one specific task's `re-arc` generator and hand-adding a bespoke
   curated action for it. A Kaggle-scored held-out task gets no such
   treatment — there is no generator source to trace and no human in the
   loop. This is a training-set-fitting process, not a generalizing solver,
   by construction.
3. **The one active bet on breaking the generalization ceiling just failed
   its own validation gate.** `docs/results/training-pass-runpod-v6-20260922b.md`
   (2026-09-22, same day as this ADR): V6's typed-grammar object GP scored
   0/8 on the exact set-op cluster ADR-0029 built it to solve, and is
   *below* parity elsewhere (2/8 vs. flat GP's 5/8) on the V5 fixture basket.

Separately, and independent of the above: every ARC-AGI approach that has
scored competitively on the public leaderboard since 2024 is LLM-centric —
either test-time-trained/fine-tuned models, or LLM-guided program synthesis
with an execution-feedback loop. This project's founding premise
(`docs/PLAN.md` Scope, F12 in `docs/QUESTIONS.md`) explicitly excluded
LLM-in-the-loop "by the project's own premise." That exclusion is now in
direct conflict with the stated goal of following state-of-the-art methods,
so it is reversed by this ADR, not worked around.

## Decision

**Pivot the project's primary method to LLM-driven program synthesis with an
execution-feedback refinement loop, inference-only — no test-time training or
fine-tuning.** Concretely, for a given task:

1. An LLM (hosted API, frozen weights) is prompted with the task's
   demonstration input/output pairs and proposes a Python program (a
   plain function over a grid, using ordinary Python/numpy — see Assumed
   defaults below) hypothesized to solve it.
2. The program is executed against the task's own training inputs (not the
   hidden test outputs — this reuses `arc_eval`'s existing blind boundary,
   see Consequences).
3. The predicted outputs are diffed against the known training outputs. Any
   discrepancy (wrong shape, wrong cells, a raised exception) is fed back to
   the LLM as concrete, structured feedback — not just "try again."
4. The LLM revises the program in light of that feedback. Repeat until the
   program matches every training pair (or a fixed attempt/token budget is
   exhausted), then apply the final program to the task's test input(s) for
   submission.

**Test-time training / weight fine-tuning is explicitly out of scope for now**
(user, 2026-09-22: "a bag of worms," deliberately deferred, not rejected
forever — a later revisit is a new fork, not a reversal of this one). This
keeps the new track inference-only: no GPU training runs, no fine-tuning
infrastructure, cost bound by LLM API spend rather than compute-hours.

**Freeze, don't delete, the curated-DSL RL/GP/object tracks.** Same
disposition this repo has always given superseded work (`legacy/`,
`research/arc-ngps/`): `arc_env/`, `trainers/ppo/`, `trainers/gp/`,
`trainers/gp_object/`, `object_env/`, and the training-dashboard/replay half
of `viz/` stay in the repo, their tests keep running as a regression safety
net, but no further feature work, ADRs, or training passes land on them.
`third_party/arc-dsl/` and `third_party/re-arc/` remain vendored — they may
yet be useful as an optional library the synthesized Python programs import
(see Assumed defaults), not as an RL/GP action space.

**V6 (`trainers/gp_object`, the object-grammar GP) is stopped permanently, no
further attempts.** ADR-0029 commitment #4 (demonstration/curriculum seeding
as a follow-up attack on V6's fitness-landscape problem) is explicitly not
pursued. This is a stronger stop than a mere pause: the user was direct that
V6 gets no more budget, not "one more bounded attempt."

**New scope, in:**

- Vendor `ARC-AGI-2` (`arcprize/ARC-AGI-2` on GitHub, Apache-2.0, same JSON
  task format as ARC-AGI-1: 1,000 public training tasks, 120 public
  evaluation tasks) under `third_party/ARC-AGI-2/`, same read-only vendoring
  convention as `third_party/ARC-AGI/`.
- Build an actual Kaggle submission pipeline: a notebook (or a script a
  notebook thinly wraps) that drives the new LLM-synthesis solver through
  `arc_eval`'s existing `Challenge`/`attempt_1`/`attempt_2` boundary — the
  scoring shape `arc_eval/evaluator.py` already mirrors, which is real
  reusable infrastructure from the old track.
- Run the frozen RL/GP/curated-DSL stack, as-is, against ARC-AGI-2's public
  evaluation set once vendored, and keep that number on record. Not to
  compete with it — to document, with evidence rather than assertion, why
  this pivot happened. That baseline belongs in `docs/results/` alongside
  the existing training-pass records.
- Retire "curated ARC-AGI-1 training tasks solved" (`arc_env/task_loader.py`'s
  `CURATED_TASK_IDS` count) as the project's north-star metric — it measures
  fit to known training tasks via hand-curated actions, not generalization.
  Replace it with: **held-out ARC-AGI-1 evaluation-split exact-match score**
  (via `arc_eval`, already unblocked by the Phase 0 locked-dataset-splits work,
  PR #79) as the near-term number, and **ARC-AGI-2 public-evaluation-split
  score** as the number that actually matters once the new solver exists.

**Scope, still out (unchanged by this ADR):** ARC-AGI-3 (a different,
interactive-game-shaped benchmark, not a program-synthesis-over-a-static-grid
problem — a separate future decision if pursued), test-time training/
fine-tuning (see above), distributed/multi-GPU training (moot now — the new
track is API-bound, not compute-bound).

## Assumed defaults

| ID | Assumed | Cost if wrong |
|----|---------|----------------|
| A1 | Synthesized programs are plain Python/numpy functions, not required to route through `arc-dsl`'s primitives. `third_party/arc-dsl/` stays vendored and *available* as an optional import, not a mandatory action space. | Low — an early implementation choice in the new solver package, not a data/schema decision; changing it later means changing the system prompt's allowed-imports guidance, not the harness. |
| A2 | LLM calls go through a hosted API (frozen weights), not a self-hosted/local model. | Low-medium — mainly a cost/latency choice; swapping providers or self-hosting later doesn't change the propose→execute→diff→refine loop shape. |
| A3 | Per-task refinement runs a fixed attempt/token budget, tuned empirically once real cost data exists. | Low — a config constant, same as the old system's `max_steps`/`n_updates`. |

## Alternatives considered

| Option | Why not |
|--------|---------|
| Test-time training / fine-tuning on ARC-AGI tasks | The dominant approach among the highest-scoring public ARC-AGI entries, but the user explicitly wants to stay away from it for now ("a bag of worms") — deferred, not ruled out forever, but not this pivot. |
| Keep growing the curated-DSL action space (more ADRs like 0020–0026) | Directly falsified by this ADR's own Context: the growth mechanism requires per-task human curation that cannot run against held-out/hidden tasks, which is exactly what a Kaggle-scored entry needs. |
| Give V6 (object-grammar GP) one more bounded attempt at ADR-0029 commitment #4 before deciding | Considered in the review that preceded this ADR; the user was explicit that V6 gets no further attempts, full stop, not one more dated try. |
| Do nothing; keep the RL/GP system as the primary track and treat Kaggle/ARC-AGI-2 as a much later, separate effort | Rejected — the user's stated goal is to be competitive now, on a ~7-week clock to the 2026-11-09 Paper Track deadline; the current track's own architecture disqualifies it from the entry vehicle the goal requires. |

## Consequences

- `arc_env/`, `trainers/{ppo,gp,gp_object}/`, `object_env/` are frozen in
  place: their existing tests keep running in CI as a regression safety net
  (no code there is deleted), but `CLAUDE.md`'s repo-layout description of
  them should be marked frozen/superseded so a future agent doesn't build on
  them by default.
- `viz/backend`'s training-dashboard/replay routes and `viz/frontend`'s
  matching UI are frozen along with the trainers they visualize — not
  deleted, since the F13/F16 "Play"/editor write path is a separate,
  potentially still-useful piece (human demonstration collection) whose fate
  is F16's own decision, not this ADR's.
- No further RunPod GPU training-pass work is expected for this project's
  mainline development; the new track's compute cost is hosted-LLM API
  spend, which needs its own cost-tracking/budget discipline (a new
  operational concern this ADR surfaces but doesn't resolve).
- A new solver package (name TBD — not `arc_env`/`trainers`, since those now
  denote the frozen track) will house the propose→execute→diff→refine loop,
  the ARC-AGI-2 loader, and the Kaggle submission entrypoint.
- The paper-track goal now has a real, buildable path: build the ARC-AGI-2
  submission, open-source it (already Apache-2.0 licensed, per this repo's
  existing `LICENSE`), get a hidden-set score, then write the paper
  documenting it — in that order, per the competition's own rules.
- Full history of this decision and its follow-ups lives in
  [`docs/questions/f17-llm-program-synthesis-pivot.md`](../questions/f17-llm-program-synthesis-pivot.md),
  linked from `docs/QUESTIONS.md`'s F17 row, same pattern as F11–F16.
