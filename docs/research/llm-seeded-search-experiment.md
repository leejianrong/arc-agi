# LLM-seeded search: a small experiment (F12 refinement)

**What this is, and — more importantly — what this is NOT.** `docs/
QUESTIONS.md` F12's 2026-09-06 refinement ("LLM-seeded search, not
LLM-designed reward") asks: does seeding GP's population / PPO's ADR-0009
warm-start with an LLM-proposed action sequence (drawn only from the
existing curated DSL, `arc_env.actions`) help those trainers succeed more
reliably/cheaply than their own unseeded search, on tasks where that search
currently struggles?

**This experiment does NOT test "can an LLM discover solutions to
previously-unsolved ARC-AGI-1 tasks."** All 3 target tasks below
(`ea32f347`, `5bd6f4ac`, `5614dbcf`) are already in `arc_env/task_loader.py`
's `CURATED_TASK_IDS`, each with a known-correct sequence already on file.
A proposed sequence that reproduces every train/test pair exactly is not a
new discovery here - it's a candidate demonstration for a search mechanism
that already has a curated action space capable of expressing the answer.
"Can an LLM find genuinely new solutions to tasks this DSL structurally
excludes" is a different, much bigger question, explicitly out of scope -
F11's own 2026-09-05 refresh already found 73% of the remaining uncurated
tasks need a higher-order primitive this architecture excludes by design
(ADR-0001), and no part of this pass changes that. State this distinction
plainly because it's easy to conflate "the LLM's proposal matched the
curated answer" with "the LLM solved a new task" - it did the former, not
the latter, on all 3 tasks.

Code changes this experiment runs on top of are covered by ADR-0018 (GP
`seed_programs`, `train.py`'s warm-start `algo` allowlist) - this doc covers
only the harness, the proposed sequences, and the actual comparison
results.

## Methodology

1. **Harness** (`scripts/llm_seed_search.py`): `describe_task` loads a
   task's train/test pairs plus the curated action menu
   (`arc_env.actions.ACTIONS`) for reasoning from; `verify` replays a
   candidate `[(name, args), ...]` sequence through
   `arc_env.actions.execute`'s raw-args path against every train + test
   pair (mirroring `tests/test_dsl_regression.py`'s own replay logic
   exactly, reusing its `_encode` helper rather than reimplementing it);
   `sequence_to_program` converts a verified sequence into a GP `Program`
   for use as a `run_gp(..., seed_programs=[...])` entry; `write_seed_episode`
   writes a verified sequence out as a real `runs/<run_id>/episodes/
   best-program.jsonl` + `run_meta.json` (`algo="llm-seed"`) via the
   unmodified `EpisodeWriter`/`write_run_meta`, replayed through `ArcEnv.step`
   (`trainers.gp.replay.program_to_episode_trace`) so the logged trace
   matches real step-by-step env semantics, not a bare-function replay.
2. **Proposing the 3 sequences**: for each target task, its train pairs were
   loaded and reasoned about directly (pixel-level diffs between input and
   output grids, described below per task) against the curated action
   menu, then verified computationally with `verify` before being compared
   to `CURATED_TASK_IDS`'s existing entry.

   **Caveat on independence**: deriving `scripts/llm_seed_search.py` and
   the two ADR-0018 code changes required reading `arc_env/task_loader.py`
   (which contains `CURATED_TASK_IDS`, including each target task's
   existing entry) and `arc_env/actions.py` as ordinary, unavoidable
   groundwork for the code-change part of this pass. The reasoning below
   was still done directly against each task's actual train-pair grids
   (dumped and inspected cell-by-cell, shown per task below), not by
   copying the known entry - but a fully blinded derivation (an LLM context
   that had never seen `task_loader.py` at all) would be a stronger
   independence guarantee than this pass can honestly claim. Recorded here
   rather than glossed over.
3. **GP-seeding comparison** (`ea32f347`, `5bd6f4ac`): `run_gp` with vs.
   without the verified seed program, standard budget
   (`population_size=200, n_generations=100, max_program_length=6,
   tournament_size=3, crossover_rate=0.7, mutation_rate=0.3, elitism=2` -
   `docs/PLAN.md`/README's documented standard config), 3 seeds per arm
   (0, 1, 2).
4. **PPO warm-start comparison** (`5614dbcf` only): 3 arms - (a) no
   warm-start, (b) warm-started from a freshly-run standard-budget GP
   result for this task (today's existing ADR-0009 mechanism), (c)
   warm-started from the LLM-seed episode. Standard PPO config
   (`--n_updates 25 --rollout_steps 128 --eval_every 5 --re_arc_prob 0.5
   --max_steps 25`, confirmed against README's "What actually works right
   now" table), single seed (0) per arm - PPO's higher per-run cost makes
   a 3-seed sweep here a real statistical-power limitation, noted honestly
   rather than glossed over.

## The 3 proposed sequences

### `ea32f347`

Train pairs recolor connected same-color (5) shapes by relative size: after
recoloring every `5` to a shared color, the largest connected component
becomes one color and the smallest becomes another, with any remaining
middle-sized component(s) left at the shared color. Reasoning through
train pair 0 directly: the input has three disjoint vertical/L-shaped
runs of color 5 (sizes 6, 5, 3 cells); the output recolors the size-6 run
to 1, leaves the size-5 run unrecolored (at color 4), and recolors the
size-3 run to 2. This is consistent across all 4 train pairs and the test
pair (verified computationally, not just eyeballed on pair 0).

Proposed sequence:

```python
[("replace", (5, 4)), ("select_largest", ()), ("recolor_selected", (1,)),
 ("select_smallest", ()), ("recolor_selected", (2,))]
```

`replace(5, 4)` turns every color-5 cell to 4 first (a shared color so
`objects()` still segments the three disjoint runs as separate
same-color components); `select_largest` then finds the size-6 run and
`recolor_selected(1)` recolors it; with that component gone,
`select_smallest` now finds the size-3 run (the smallest of the two
remaining) and `recolor_selected(2)` recolors it, leaving the size-5 run
at color 4 untouched - exactly the observed output.

**Result: matches `CURATED_TASK_IDS["ea32f347"]` exactly**, verified by
`scripts/llm_seed_search.py verify --task_id ea32f347` (5/5 pairs pass:
4 train + 1 test).

### `5bd6f4ac`

Every train pair's 3x3 output appears verbatim as a sub-block of the 9x9
input. Checking where: pair 0's output rows are `970`/`484`/`400`; input
row 0 is `300700970` (columns 6-8 = `970`), row 1 is `840660484` (columns
6-8 = `484`), row 2 is `170000400` (columns 6-8 = `400`) - an exact match
at rows 0-2, columns 6-8. The same crop window (rows 0-2, columns 6-8)
was checked against pair 1 (output `060`/`081`/`445` vs. input rows 0-2
columns 6-8) and matches there too - the crop window is fixed across
every pair, not input-dependent.

Proposed sequence:

```python
[("commit", (0, 6, 3, 3))]
```

**Result: matches `CURATED_TASK_IDS["5bd6f4ac"]` exactly**, verified 5/5
pairs (4 train + 1 test).

### `5614dbcf`

Each 9x9 input is a 3x3 macro-grid of 3x3 blocks; each block is either a
solid "background" color (uniformly 0) or a solid "fill" color with a
handful of stray noise cells of color 5 scattered in it. The 3x3 output is
each block's fill color (0 for a background block). Checking how a plain
`downscale(factor=3)` (which, per `third_party/arc-dsl/dsl.py`'s actual
implementation, samples the top-left corner pixel of each `factor x factor`
block, not a majority vote) would fare: on pair 0 the 9 corner pixels
(rows/cols 0, 3, 6) happen to avoid every noise cell, giving the correct
answer already. But on pair 1, the block at macro-position (1, 0) has its
noise pixel sitting exactly at its own top-left corner (input row 3, column
0 is `5`, not that block's true fill color 0) - a plain corner-sampling
`downscale` would read this block as color 5, but the expected output is
0. Replacing every color-5 cell with 0 first removes this failure mode
without changing any other block's corner pixel (color 5 is never used as
an actual block fill color in this task family, only as noise), fixing the
mismatch.

Proposed sequence:

```python
[("replace", (5, 0)), ("downscale", (3,))]
```

**Result: matches `CURATED_TASK_IDS["5614dbcf"]` exactly**, verified 3/3
pairs (2 train + 1 test). (`docs/PLAN.md`'s KAN-1239 section separately
notes GP's own found solution for this task is a different, also-valid
path - `select_smallest` -> `move_selected` -> `downscale(factor=3)` -
confirming there is more than one curated-DSL-expressible way to solve it;
this pass's proposal happens to match the `arc-dsl` solver-derived entry,
not GP's.)

## GP-seeding comparison: results

Standard budget (`population_size=200, n_generations=100,
max_program_length=6, tournament_size=3, crossover_rate=0.7,
mutation_rate=0.3, elitism=2`), 3 seeds (0, 1, 2), unseeded vs. seeded with
each task's verified `Program` (via `sequence_to_program`):

| Task | Seed | Arm | Solved (exact match) | Generation solved | Best fitness (exact_match_fraction, similarity) | Wall time |
|---|---|---|---|---|---|---|
| `ea32f347` | 0 | unseeded | No | - (ran all 100) | (0.0, 0.694) | 17.2s |
| `ea32f347` | 0 | seeded | **Yes** | 0 | (1.0, 1.0) | 0.17s |
| `ea32f347` | 1 | unseeded | No | - (ran all 100) | (0.0, 0.660) | 22.1s |
| `ea32f347` | 1 | seeded | **Yes** | 0 | (1.0, 1.0) | 0.16s |
| `ea32f347` | 2 | unseeded | No | - (ran all 100) | (0.0, 0.660) | 25.9s |
| `ea32f347` | 2 | seeded | **Yes** | 0 | (1.0, 1.0) | 0.20s |
| `5bd6f4ac` | 0 | unseeded | No | - (ran all 100) | (0.0, 0.689) | 15.1s |
| `5bd6f4ac` | 0 | seeded | **Yes** | 0 | (1.0, 1.0) | 0.28s |
| `5bd6f4ac` | 1 | unseeded | No | - (ran all 100) | (0.0, 0.669) | 19.8s |
| `5bd6f4ac` | 1 | seeded | **Yes** | 0 | (1.0, 1.0) | 0.33s |
| `5bd6f4ac` | 2 | unseeded | No | - (ran all 100) | (0.0, 0.669) | 18.8s |
| `5bd6f4ac` | 2 | seeded | **Yes** | 0 | (1.0, 1.0) | 0.23s |

**Aggregate: unseeded 0/6 exact matches (0%), seeded 6/6 (100%), every
seeded run solving at generation 0.** The unseeded arm reproduces
`docs/PLAN.md`'s already-documented standard-budget 0% finding for both
tasks almost exactly (plateau similarity 0.66-0.69 here vs. that doc's
reported 0.71-0.73 for `5bd6f4ac` and 0.45-0.53 for `ea32f347` from earlier
investigation passes - close enough to read as the same underlying
landscape, not a contradiction; exact plateau values are seed- and
run-history-dependent).

**Why every seeded run solves at generation 0, and what that does and
doesn't prove.** Since the seed program is itself a full, verified solution
(`fitness = (1.0, 1.0)`), it scores perfectly the moment generation 0 is
evaluated, and `run_gp`'s existing early-stop-on-perfect-fitness check fires
immediately - no evolution happens at all in the seeded arm. This is a
real, honest result (the mechanism is exactly what ADR-0018 built: include
a demonstration in the initial population, let ordinary tournament
selection/elitism keep it if it's fit, which a perfect solution always is),
but it's also the least-surprising possible outcome of seeding with an
already-perfect program - it does not test "does GP still benefit from a
noisier, incomplete, or subtly-wrong LLM proposal," a harder and more
realistic question this small experiment didn't scope (F12's brief was
explicit: propose a sequence, verify it, seed it - not stress-test partial
or wrong seeds). What this result does establish cleanly: a cheap,
one-off LLM proposal (17-26 seconds each) reliably fixes both of GP's only
standard-budget zero-success curated tasks, 3/3 seeds, at a cost
(generation 0, under a third of a second of GP compute) far below
`docs/PLAN.md`'s already-tried alternative of a 25x larger budget
(`population_size=1000, n_generations=500`), which itself only reached 2/3
seeds on `ea32f347` and 0/3 on `5bd6f4ac`. Seeding strictly dominates that
budget-increase mitigation on both cost and reliability, for these 2 tasks.

## PPO warm-start comparison: results (`5614dbcf`)

Standard PPO config (`--n_updates 25 --rollout_steps 128 --eval_every 5
--re_arc_prob 0.5 --max_steps 25`), single seed (0). 3 arms: (a) no
warm-start; (b) warm-started from a fresh standard-budget GP run for this
task (`--warm_start_from` an `algo="gp"` run); (c) warm-started from the
verified LLM-seed episode (`--warm_start_from` the `algo="llm-seed"` run
`write_seed_episode` wrote).

**Demonstration used by each warm-started arm.** (b)'s fresh GP run solved
at generation 0 (population 200, standard config) with a different,
5-gene program than the LLM-seed proposal - not the clean 2-step
`replace(5,0) -> downscale(3)` this pass proposed, but a longer, messier
program GP's random initialization happened to land on: `[(31, (2, 16, 15,
13)), (20, (12, 23, 27, 0)), (7, (13, 19, 21, 29)), (8, (24, 4, 28, 0)),
(7, (22, 13, 7, 1))]` (primitive indices/raw args as logged). (c) is the
2-step LLM-seed sequence itself (`replace(5, 0)`, `downscale(3)`).
Behavior-cloning pretrain loss (`--warm_start_epochs 50`, final epoch)
diverged sharply between the two: **GP's 5-step demonstration reached bc_loss
0.34**; **the LLM-seed's 2-step demonstration reached bc_loss 1.10** - a
harder demonstration to imitate, consistent with `docs/PLAN.md`'s KAN-1239
section separately noting this task's warm-start bc loss (1.21, for what
reads as the same textbook-style short solution) was the highest of its
14-task warm-start pass.

**`eval_success`/`eval_reward` across every logged checkpoint** (per this
repo's own convention - `eval_success`, not the noisier rollout
`success_rate`, is what "solved" means, per `CLAUDE.md`):

| Update | (a) no warm-start | (b) GP warm-start | (c) LLM-seed warm-start |
|---|---|---|---|
| 0  | False (-0.250) | **True** (1.722) | False (-0.250) |
| 5  | False (-0.449) | **True** (1.722) | False (-0.250) |
| 10 | False (-0.500) | **True** (1.722) | False (-0.250) |
| 15 | False (-0.500) | **True** (1.722) | False (-0.250) |
| 20 | False (-0.020) | **True** (1.722) | **True** (1.752) |
| 24 (final) | False (0.120) | **True** (1.722) | **True** (1.752) |

**Result: both warm-start arms solve `5614dbcf` by the final checkpoint;
plain PPO never does.** (a) never reaches `eval_success=True` at any
logged checkpoint, ending at `eval_reward=0.120` (some partial progress
by the final update, but no exact match). (b) reaches `eval_success=True`
immediately at update 0 (the easy-to-fit demonstration already gets the
pretrained policy to solve the fixed eval pair before any PPO update runs)
and holds it, completely stable, through update 24 - no regression at any
checkpoint. (c) starts `eval_success=False` through update 15 (consistent
with its harder-to-fit demonstration - the pretrained policy alone isn't
enough), then locks in at update 20 and holds through the final update 24,
ending at a marginally higher `eval_reward` (1.752 vs. 1.722) than the GP
arm.

**This is a genuinely useful, if nuanced, result for F12's actual
question** ("does an LLM-derived demonstration do any better than the
existing GP-derived one, on a task where ADR-0009's warm-start was
previously documented as unstable?"): the LLM-seed demonstration reaches
the same end state (solved, stable, at the final checkpoint) as the GP
demonstration, despite needing a harder demonstration to imitate (bc_loss
3x higher) and converging later in training (update 20 vs. update 0). It
does not do *better* than GP's demonstration in this single-seed run - GP
got lucky with an atypically easy-to-imitate program this particular run,
not because GP's demonstrations are inherently easier on this task (recall
`docs/PLAN.md`'s own KAN-1239 section calls `5614dbcf`'s warm-start bc loss
1.21 - close to this pass's LLM-seed bc_loss of 1.10, not to this run's GP
bc_loss of 0.34 - suggesting GP's *typical* found program on this task is
closer in difficulty to the LLM-seed's than this one lucky run's 5-gene
solution was). The more robust reading: **both a GP-derived and an
LLM-derived demonstration turn `5614dbcf` from a plain-PPO failure into a
solved, stable-by-the-final-checkpoint task** - direct evidence the
LLM-seed mechanism (ADR-0018) is a viable *alternative* demonstration
source when a GP demonstration either isn't available or (per the earlier
documented instability) doesn't reliably help, not merely a same-shape
duplicate of what GP already provides.

**Statistical-power caveat, stated plainly per the brief**: this is a
single seed per arm. GP's demonstration difficulty varies run-to-run (its
search is stochastic - a different GP seed could easily find the clean
2-step program instead of this run's messier 5-step one, or something
harder still), so this single comparison should not be read as "GP
warm-start is more reliable than LLM-seed warm-start on this task" or vice
versa - only as "both demonstration sources got PPO to a solved, stable
final checkpoint in this one run, where plain PPO did not."

## Did seeding help?

**Yes, on both fronts tested, though the GP-seeding result is a cleaner win
than the PPO one.**

- **GP-seeding (`ea32f347`, `5bd6f4ac`)**: unambiguous. 0/6 unseeded exact
  matches vs. 6/6 seeded, every seeded run solving at generation 0. This
  strictly dominates `docs/PLAN.md`'s previously-tried 25x-larger-budget
  mitigation on both cost and reliability for these 2 tasks - though, as
  noted above, this is the least-surprising outcome of seeding with an
  already-perfect demonstration, not a test of noisier/partial seeds.
- **PPO warm-start (`5614dbcf`)**: also a clear win over plain PPO (which
  never solves the eval pair at any checkpoint), and the LLM-seed
  demonstration reached the same solved-and-stable final state as the
  GP demonstration, despite being harder to imitate (3x the bc loss) and
  taking longer to converge during training. It did not outperform GP's
  demonstration outright in this single run, but this run's GP
  demonstration happened to be atypically easy - a fairer comparison would
  need multiple GP seeds to characterize its typical difficulty on this
  task, which this pass's single-seed-per-arm design (an explicit,
  acknowledged limitation) doesn't provide.
- **Nothing tested here showed seeding hurting** - no arm did worse than
  its corresponding unseeded/non-warm-started baseline.

**What would make this a stronger result**: multiple GP seeds for the PPO
comparison (to characterize GP-demonstration difficulty variance on
`5614dbcf` and see whether the LLM-seed's fixed difficulty is typically
better, worse, or comparable); a GP-seeding test with a genuinely partial
or subtly-wrong LLM proposal, not only a fully-verified one, to see whether
seeding helps GP even short of a perfect demonstration; and a fully
blinded sequence-derivation setup (see the Methodology caveat above) for a
cleaner independence claim on the "did the LLM actually reason this out"
question, even though it doesn't change the results here (both are `1.0`/
verified either way).

## Test results

- `uv run pytest -m "not slow" -q`: **476 passed** (up from the pre-pass
  baseline; +16 new tests this pass across the 3 files below).
- `uv run pytest -q` (full, including `slow`-marked e2e tests): **483
  passed** (476 fast + 7 slow).
- `uv run ruff check .`: **clean, no findings.**
- New tests added this pass: `tests/test_gp_evolve.py` (4 new
  `seed_programs` tests), `tests/test_warm_start.py` (2 new allowlist
  tests), `tests/test_llm_seed_search.py` (10 new harness tests, new file).
- No regressions - every test that passed before this pass still passes.

## Judgment calls / deviations from the brief

- `seed_programs` was added as a `run_gp` keyword argument, not a new
  `GPConfig` field (the brief explicitly left this as "your call") - see
  ADR-0018's Mechanism section for why (keeps `run_meta.json`'s config
  record JSON-trivial).
- The warm-start allowlist (`WARM_START_COMPATIBLE_ALGOS = {"gp",
  "llm-seed"}`) deliberately does not also add `"human"` (ADR-0017's
  write path), even though it produces an identically-shaped episode -
  left as an explicit future decision per ADR-0018's Alternatives, not
  folded in as an unplanned side effect of this pass.
- The "independently derived" proposals carry the caveat described above
  in Methodology point 2: the same agent that wrote the code changes also
  read `CURATED_TASK_IDS` as ordinary groundwork before reasoning about the
  train pairs pixel-by-pixel. All 3 proposals were verified computationally
  against the raw train/test grids before being compared to the existing
  entries, but this is not a fully blinded derivation.
