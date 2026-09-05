# ADR-0014: GP replay logs a generation-interval snapshot series, not just the best program

- Status: Accepted
- Date: 2026-09-05
- Deciders: repo owner (delegated design), via conversation 2026-09-05 (visualizer overhaul, Slice 4)

## Context

`viz/backend/server.py`'s `GET /api/runs/<id>/episodes` only ever returns one
episode, `best-program`, for a GP run - `trainers/gp/replay.py`'s
`program_to_episode_trace` is only ever called once, on `GPResult.best_program`
(the single best program `trainers/gp/evolve.py`'s `run_gp` ever found).
There's no way to watch the population evolve the way PPO's `eval-update*`
episodes already let you compare an early- vs. late-training checkpoint - GP
has no equivalent "early vs. late" story at all.

`run_gp`'s generational loop already computes each generation's own best
program (`gen_best_program`, from sorting that generation's scored
population) - it's just discarded after `GenerationRecord` extracts its
scalar fitness/similarity for `metrics.jsonl`. Capturing it costs nothing
extra at search time; the only added cost is replaying it later
(`program_to_episode_trace` - one more `ArcEnv` rollout per captured
generation, and each program is at most `max_program_length` steps, so this
is cheap in absolute terms).

The frontend already has everything needed to *show* a series like this:
`viz/frontend/src/main.ts`'s `PlayerPanel.setRun` already lets a user pick
among an arbitrary list of episode IDs for one run, and defaults Panel A to
the alphabetically-first episode and Panel B to the alphabetically-last one -
exactly the "early vs. late" comparison this needs, already built for PPO's
`eval-update00000`/.../`eval-update000NN` series.

## Decision

**`run_gp` snapshots each generation's own best program at a configurable
interval, and `train_gp` replays and logs one episode per snapshot -
reusing the existing episode-schema and frontend machinery, not a new
"scrub across generations" widget.**

- `GPConfig` gains `snapshot_interval: int = 10`. `GPResult` gains
  `snapshots: list[tuple[int, Program]]` - `(generation, program)` for every
  generation whose index is a multiple of `snapshot_interval`, always
  including generation 0 and the final generation actually run (whether that
  final generation is `n_generations - 1` or an earlier one from GP's
  existing early-stop-on-perfect-fitness behavior). No change to
  `GenerationRecord`'s scalar fields or `metrics.jsonl`'s shape - this is
  additive, not a schema change to what already exists.
- `train_gp` replays each snapshot via the existing, unmodified
  `program_to_episode_trace` and writes it as episode ID
  `f"{generation:05d}-gen"` (e.g. `00000-gen`, `00010-gen`, `00090-gen`) -
  zero-padded so the IDs sort chronologically. It still separately replays
  and writes `result.best_program` as episode `best-program`, exactly as
  before.
- Default `snapshot_interval=10`: for the existing `n_generations=100`
  default this is ~11 snapshots (11 extra `ArcEnv` rollouts + JSONL files) -
  enough to see real generation-over-generation change without every
  generation's near-identical neighbor cluttering the episode picker. Also
  exposed as `train.py --algo gp`'s `--snapshot_interval`, matching every
  other `GPConfig` field already exposed there.

**Why the numeric-leading name (`00000-gen`, not e.g. `gen-00000`), and why
`best-program` keeps its exact existing name:** `trainers/ppo/warm_start.py`'s
`load_demonstration(gp_run_dir, episode_id="best-program")` (ADR-0009's
warm-start path) hardcodes that literal episode ID - renaming or dropping it
would silently break `--warm_start_from`. But `viz/backend/server.py`'s
`list_episode_ids` returns `sorted(...)` filenames, and `main.ts` picks the
alphabetically-first/last episode ID for its two comparison panels. A name
like `gen-00000` sorts *after* `best-program` (`g` > `b`), which would have
put the final/best program in Panel A ("earliest") and a mid-run snapshot in
Panel B ("latest") - backwards. Every snapshot name here starts with a digit
(`0`-`9`, all below ASCII `b`), so the full series always sorts
chronologically *and* strictly before `best-program` - Panel A lands on
generation 0, Panel B lands on `best-program`, matching PPO's own
earliest-eval/latest-eval default with no frontend change at all.

**Why generation-interval snapshots reuse the existing episode picker rather
than a new "scrub across generations" UI:** the existing multi-episode
picker already solves "pick an early one and a late one to compare" for
PPO; GP needing the same comparison is the same problem, not a new one. A
dedicated generation-slider widget would be a real UI investment for a
capability the frontend already has.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Snapshot every single generation | Up to 100 extra `ArcEnv` rollouts + JSONL files per GP run, most of them near-identical to their neighbor thanks to elitism's monotonic improvement (see Consequences) - all cost, little signal. `snapshot_interval` makes this a knob, not a decision to relitigate per run. |
| A new frontend "generation scrubber" component, separate from the existing episode picker | The existing episode picker/two-panel comparison already does exactly what's needed (pick any two episodes from a list, per-run) - building a parallel, GP-specific widget would duplicate that mechanism for no new capability. |
| Rename `best-program` to fit a uniform `gen-*` naming scheme, updating `warm_start.py` to match | Rejected: couples this visualizer-only change to ADR-0009's training-pipeline contract for no benefit - `warm_start.py` doesn't care what a GP run's *other* episodes are named, only that this one specific ID still resolves. Keeping `best-program` frozen and instead choosing a snapshot-naming scheme that sorts correctly around it is strictly lower-risk. |
| Snapshot the true running best-so-far program at each interval, rather than that generation's own best | Provably the same sequence of programs as generation-own-best, given elitism (see Consequences) - no behavioral difference, and generation-own-best needed no extra state to track alongside the existing loop. |

## Consequences

- `GenerationRecord`/`metrics.jsonl` are unchanged; `GPConfig.to_dict()` now
  includes `snapshot_interval`, which flows harmlessly into `run_meta.json`'s
  logged config the same way every other GP hyperparameter already does.
- Elitism (`config.elitism >= 1`, true for every curated task's default
  config) guarantees each generation's own best fitness is non-decreasing
  generation-over-generation: the previous top `elitism` genomes always
  carry over into the next population unchanged, and fitness evaluation is
  deterministic for a fixed task, so they re-score identically. This means
  the snapshot series is itself a genuinely monotonically-improving trace
  (never "worse, then better again"), and the final snapshot's program is
  always identical to `best_program` - confirmed directly by
  `tests/test_gp_evolve.py`'s `test_final_snapshot_program_matches_the_best_program_found`
  and `test_snapshots_survive_early_stop_on_perfect_fitness`.
- Disk/compute cost per GP run grows by roughly
  `n_generations_run / snapshot_interval` extra episode replays and JSONL
  files - small in absolute terms (each program is at most
  `max_program_length` steps), but not free; a much smaller
  `snapshot_interval` than the default should be a deliberate choice, not an
  accident.
- `viz/backend/server.py` and the frontend needed **no changes at all** -
  `GET /api/runs/<id>/episodes` already lists whatever's in `episodes/`, and
  `main.ts`'s existing episode picker/panel-default logic already handles an
  arbitrary-length list correctly. This is the entire point of the
  numeric-leading naming choice above.
