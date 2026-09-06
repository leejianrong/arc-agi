# ADR-0018: GP population seeding and a warm-start algo allowlist (F12's LLM-seeded-search mechanisms)

- Status: Accepted
- Date: 2026-09-06
- Deciders: repo owner (delegated implementation, per F12's 2026-09-06
  "LLM-seeded search, not LLM-designed reward" refinement), via conversation
  2026-09-06

## Context

`docs/QUESTIONS.md` F12 records a narrow, lower-risk cut of "LLM-in-the-loop"
that doesn't touch ADR-0005's shared-reward invariant: an LLM proposes a
candidate action sequence, drawn only from the existing curated DSL
(`arc_env.actions`, never arbitrary code), for a task GP and/or PPO
currently struggle with; that sequence feeds GP's initial population and/or
PPO's ADR-0009 warm-start as a *second* demonstration source, alongside (not
instead of) GP's own search. Two mechanisms were missing to make this
possible at all:

1. **GP has no population-seeding mechanism.** `trainers/gp/evolve.py`'s
   `run_gp` always initializes generation 0 purely at random
   (`[random_program(rng, config.max_program_length) for _ in
   range(config.population_size)]`) - there is no way to hand it a
   candidate program to include in that initial population.
2. **`train.py`'s `check_warm_start_compatible` hardcodes `algo == "gp"`.**
   `--warm_start_from` only accepts a run directory whose `run_meta.json`
   records `algo: "gp"` - an LLM-seeded demonstration, written out with any
   other `algo` value, is rejected outright, even though the mechanism it
   feeds (`trainers/ppo/warm_start.py`'s `load_demonstration`) only actually
   cares about the episode's *shape* (ADR-0009's own design), not which
   trainer produced it.

Both are small, additive, backward-compatible changes - not a redesign of
either trainer. This ADR covers only these two mechanisms; the harness that
proposes/verifies/writes a candidate sequence (`scripts/llm_seed_search.py`)
and the one-off experiment run built on top of both are not ADR-level
decisions themselves (per F12's framing) and are written up in
`docs/research/llm-seeded-search-experiment.md` instead.

## Decision

### 1. `run_gp(task, config, seed_programs=None)`

A new, optional keyword argument on `trainers/gp/evolve.py`'s `run_gp` (not
a new `GPConfig` field - see Mechanism below for why): a `list[Program] |
None`. When given and non-empty, generation 0's population is

```python
seeds = list(seed_programs)[:config.population_size]
n_random = config.population_size - len(seeds)
population = seeds + [random_program(rng, config.max_program_length) for _ in range(n_random)]
```

i.e. the seed(s) **replace** some of generation 0's random slots, never
appended on top of `population_size` - a seeded run costs exactly the same
per-generation compute as an unseeded one at the same config. Extra seeds
beyond `population_size` are dropped (capped, not an error), and any
shortfall is filled with `random_program` exactly as before - a single weak
seed still leaves normal random search as the fallback.

Omitting `seed_programs` (the default `None`, and an explicit empty list)
reproduces today's exact `random.Random(config.seed)` call sequence for
generation 0's population, byte-for-byte - strictly additive, opt-in,
covered by
`tests/test_gp_evolve.py::test_omitting_seed_programs_reproduces_todays_exact_population`.

### 2. `WARM_START_COMPATIBLE_ALGOS = {"gp", "llm-seed"}`

`train.py`'s `check_warm_start_compatible` now checks `meta.get("algo") in
WARM_START_COMPATIBLE_ALGOS` instead of `meta.get("algo") != "gp"` - a small
explicit allowlist, not simply deleting the check. `"llm-seed"` is the new
value `scripts/llm_seed_search.py`'s `write_seed_episode` writes to a run's
`run_meta.json`. The same-task-only constraint (`task_ids == [task_id]`) is
unchanged.

## Mechanism

**Why `seed_programs` is a `run_gp` argument, not a `GPConfig` field.**
`GPConfig.to_dict()` (via `asdict`) is serialized verbatim into
`run_meta.json`'s `config` (`train.py`'s `train_gp`) - a flat, JSON-trivial
hyperparameter record every other field in it already is. A list of
`Program`s (nested tuples of primitive indices and raw args) doesn't belong
in that record the same way `population_size`/`n_generations`/etc. do, and
serializing it there would bloat `run_meta.json` for every GP run whether or
not seeding is in play. Keeping it a `run_gp`-only argument (like `task`
itself) means unseeded runs are completely unaffected by this change at
every level, not just in the population-construction code path.

**Why the warm-start check is an allowlist, not a removed check.** The
check's real purpose (per ADR-0009) was always "does this run dir hold a
same-task `best-program`-shaped demonstration episode", not "was it
literally produced by the GP trainer" - `"gp"` was simply the only producer
that existed when ADR-0009 landed. Broadening it to a small explicit set
keeps out anything that isn't a deliberately-produced demonstration run
(e.g. a plain `"ppo"` run has no `best-program` episode at all, and would
fail later at `load_demonstration` with a confusing file-not-found error
instead of this check's clear message). `"human"` (`viz/backend/play.py`,
ADR-0017) is a real demonstration-shaped run too, but wiring it into
`--warm_start_from` is left as an explicit future decision (F13 Stage 1/2's
own territory), not an accidental side effect of this change - covered by
`tests/test_warm_start.py::test_check_warm_start_compatible_still_rejects_a_human_run`.

**No change to `load_demonstration`, `pretrain_from_demonstration`, or any
other part of ADR-0009's pipeline** - once a run passes
`check_warm_start_compatible`, warm-starting from an `"llm-seed"` run goes
through the exact same code path as a `"gp"` run always has, because both
write the identical `episodes/best-program.jsonl` shape via the same
`EpisodeWriter`.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Add `seed_programs` to `GPConfig` instead of `run_gp`'s signature | Rejected - see Mechanism above: would force a non-JSON-trivial field into `run_meta.json`'s config record for every GP run, seeded or not. |
| Remove `check_warm_start_compatible`'s algo check entirely, keep only the same-task check | Rejected - the point of a demonstration-source check isn't bureaucratic; it catches a plain PPO run or a garbage directory early with a clear message instead of an obscure downstream parse failure. A short allowlist keeps that value while accepting the one new legitimate producer. |
| Also add `"human"` to the allowlist now, since `viz/backend/play.py` already produces the identical schema | Rejected for this pass - F13's own Stage 0 ADR (ADR-0017) explicitly deferred "should `warm_start_from` accept a human episode ID" as a Stage 1/2 question; folding it in here as a byproduct of an unrelated change would pre-empt that decision rather than make it deliberately. |
| Seed GP's population by mutating/appending to it *after* generation 0 (e.g. injecting the seed at generation 1) | Rejected - generation 0 is the first point tournament selection/elitism act on the population; seeding any later generation would need extra bookkeeping (which generation, how it interacts with elitism's carry-over) for no benefit over simply including it from the start. |

## Consequences

- `trainers/gp/evolve.py::run_gp`'s signature grows by one optional
  argument; every existing call site (`train.py`'s `train_gp`, every
  existing test) is unaffected since it defaults to `None`.
- `train.py`'s `WARM_START_COMPATIBLE_ALGOS` is a new small, named module
  constant - any future demonstration-producing write path (beyond `"gp"`/
  `"llm-seed"`) adds itself here explicitly rather than loosening the check
  further.
- Both changes are covered by new unit tests
  (`tests/test_gp_evolve.py`'s 4 new `seed_programs` tests,
  `tests/test_warm_start.py`'s 2 new allowlist tests) alongside the existing
  suites, which pass unchanged.
- Enables (but does not itself run) the experiment in
  `docs/research/llm-seeded-search-experiment.md`: GP-seeding comparisons on
  `ea32f347`/`5bd6f4ac`, and a 3-arm PPO warm-start comparison on
  `5614dbcf` (plain vs. GP-warm-started vs. LLM-seed-warm-started). See that
  doc for the actual numbers, including where seeding did and didn't help.
- `docs/QUESTIONS.md`'s F12 entry is appended (not rewritten) with this
  pass's outcome, mirroring how ADR-0015/0016/0017 each appended to
  F11/F13.
