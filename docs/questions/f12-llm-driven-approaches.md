# F12: whether to pursue LLM-driven approaches

**Status:** deferred. Not needed this milestone; revisits Q2's "no
LLM-in-the-loop" scope boundary for a later one.

Whether and how to pursue LLM-driven approaches, program synthesis,
hypothesis testing, on-the-fly reward or primitive design, as a track
toward higher ARC-AGI-1 coverage and, eventually, ARC-AGI-2/3.

## The user's 2026-09-05 framing

Three options, recorded for later:

1. An LLM designs reward functions or primitives on the fly.
2. An LLM writes arbitrary Python/DSL-composition scripts per task,
   verified against train pairs.
3. An LLM models a single task's rules explicitly and iterates via an
   exploring agent that tests hypotheses.

Assessment: all three are a genuinely different architecture family from
this project's GP/PPO-over-curated-flat-DSL substrate, not an increment on
it. Options 2 and 3 in particular sidestep the exact restriction (no
`Callable`/higher-order primitives, ADR-0001) that
[F11](f11-task-coverage.md)'s 2026-09-05 refresh found blocks 73% of
ARC-AGI-1's remaining uncurated tasks. And ARC-AGI-2 was deliberately
designed to defeat compositional search over a small, fixed DSL menu, the
current architecture's whole category. So options 2 and 3 look like the
more credible path toward either goal than more curated-DSL growth.

Option 1 is valid, but it's in tension with ADR-0005's
one-reward-shared-by-both-trainers invariant, which is this project's own
comparative-research value. If pursued, treat it as its own experiment
track with its own comparability story, not a bolt-on to the existing
GP/PPO harness.

Recommendation: keep this a clearly separate initiative, its own docs, its
own ADRs, rather than folding it into F11's incremental-coverage loop, if
and when it's picked up.

## Refinement: LLM-seeded search, not LLM-designed reward (2026-09-06)

A narrower, lower-risk cut of option 2 that doesn't touch ADR-0005's
shared-reward invariant at all: an LLM proposes a candidate action
sequence drawn from the existing curated DSL (`arc_env.actions`, never
arbitrary code) for a given task. That sequence feeds GP's initial
population and/or PPO's ADR-0009 warm-start as a second demonstration
source, alongside, not instead of, GP's own search, applied symmetrically
to both trainers so the GP-vs-PPO comparison this project is built around
stays intact.

Small experiment shape: for a handful of curated tasks GP/PPO currently
struggle with (or a few from F11's near-miss buckets), have an LLM propose
a `CURATED_TASK_IDS`-style action sequence from the task's train pairs and
the `arc_env.actions` menu, verify it against train and test pairs the
same way `tests/test_dsl_regression.py` does, and if valid, seed it into
GP's population and PPO's warm-start, comparing against a no-seed
baseline.

## ADR-0018: running the experiment (2026-09-06)

Built the two code mechanisms this refinement needs:
`trainers/gp/evolve.py`'s `run_gp(..., seed_programs=...)` (seeding
generation 0's population rather than pure random init), and widened
`train.py`'s `check_warm_start_compatible` from a hardcoded
`algo == "gp"` check to an explicit
`WARM_START_COMPATIBLE_ALGOS = {"gp", "llm-seed"}` allowlist. Plus a
harness, `scripts/llm_seed_search.py` (`describe_task`, `verify`,
`sequence_to_program`, `write_seed_episode`).

Then ran the actual small experiment on 3 curated tasks GP/PPO currently
struggle with: `ea32f347` and `5bd6f4ac` (GP's only two standard-budget
zero-success tasks), and `5614dbcf` (GP solves it, PPO doesn't, and its
existing GP-based warm-start was separately documented as unstable; see
`docs/PLAN.md`'s KAN-1239 section).

We reasoned through each task's train pairs directly (pixel-level diffs)
and proposed a `CURATED_TASK_IDS`-style sequence for each. All 3 verified
against every train and test pair, and, notably, all 3 matched
`CURATED_TASK_IDS`'s existing entry exactly. One honesty caveat: the same
pass's code-change work required reading `task_loader.py`, so this isn't a
fully blinded derivation.

**GP-seeding result** (standard budget, 3 seeds each): unseeded, 0/6 exact
matches across both tasks, reproducing the already-documented
standard-budget failure. Seeded, 6/6, every seeded run solving at
generation 0. That's strictly better than `docs/PLAN.md`'s previously
tried 25x-larger-budget mitigation, which itself only reached 2/3 seeds on
`ea32f347` and 0/3 on `5bd6f4ac`, though this is the least surprising
possible outcome of seeding with an already-perfect demonstration, not a
test of noisier or partial seeds.

**PPO warm-start result on `5614dbcf`** (single seed, 3 arms): plain PPO
never reaches `eval_success=True` at any checkpoint (ends at
`eval_reward=0.12`). Both a fresh GP warm-start and the LLM-seed warm-start
do reach `eval_success=True` and hold it through the final checkpoint. The
GP arm gets there immediately, its randomly found demonstration this run
happened to have a low 0.34 behavior-cloning loss. The LLM-seed arm gets
there only from update 20 onward, its cleaner 2-step demonstration had a
harder-to-imitate 1.10 bc loss, closer to KAN-1239's own reported 1.21 for
this task.

Read plainly: the LLM-seed mechanism is a viable alternative demonstration
source that also turns this task from a plain-PPO failure into a stable
final-checkpoint solve. It's not proven better or worse than GP's own
demonstration from this one run alone. GP's demonstration difficulty
varies run to run, and a proper comparison needs multiple GP seeds, an
acknowledged limitation of this single-seed-per-arm pass.

All new and existing tests pass (`uv run pytest -m "not slow" -q` and the
full `uv run pytest`, `uv run ruff check .`). See
`docs/research/llm-seeded-search-experiment.md` for exact counts.

Overall: seeding helped in every comparison run this pass, with the
PPO-side caveat above stated rather than oversold.

**Landed by:** ADR-0018, `docs/research/llm-seeded-search-experiment.md`.
