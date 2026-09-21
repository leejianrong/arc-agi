# ARC-AGI RL/Evolutionary Agent: Slices

Vertical increments. Each ends in something you can demonstrate. Slice 1
confronts the riskiest unknown: does the whole env → trajectory-log →
visualizer-replay chain actually work, before any learning is involved.

## V1: See it move

**Delivers:** R1 (partial: curated action subset, same-shape only), R4

**Build plan**

1. Vendor `arc-dsl` (`dsl.py`, `arc_types.py`, `constants.py`) into
   `third_party/arc-dsl/` (ADR-0001).
2. Build `arc_env/`: Gymnasium-style `step`/`reset`, a curated action subset
   (exclude higher-order primitives per ADR-0001; exclude canvas/crop per
   ADR-0002 — that's V3), a task loader over a hand-picked same-shape-only
   subset of `third_party/ARC-AGI/data/training` (~10-20 tasks).
3. Random-policy rollout script that steps the env and writes
   `runs/<run_id>/episodes/<episode_id>.jsonl` (ADR-0006).
4. `viz/backend`: serve a run directory's episode files as JSON over local
   HTTP.
5. `viz/frontend`: TypeScript + Canvas grid renderer (palette from
   `third_party/ARC-AGI/apps/js`, per ADR-0007) with play/pause/step/speed
   controls, reading one episode from the backend.

**Demo:** run the random-policy rollout script against one task, open the
visualizer, and watch the grid change one DSL action at a time, exactly like
stepping through a human's moves in `third_party/ARC-AGI/apps/testing_interface.html`.

**Rests on assumptions:** Q3 (trajectory JSONL schema) — if wrong, only the
env's logging code and the frontend's parsing need to change, not the env
logic itself.

### Test plan

#### End-to-end

- Running the rollout script against a fixture task produces an
  `episodes/*.jsonl` file the visualizer can load and step through without
  error, showing the correct starting grid and correct grid after each
  logged action.

#### Integration

- The env's executor reproduces the exact expected output when stepped
  through one of `arc-dsl`'s known-correct solver programs for a same-shape
  task (see PLAN.md Testing approach).

#### Unit

- Each curated DSL primitive wrapped as an action produces the same grid as
  calling the vendored `arc-dsl` function directly, for representative
  inputs.
- The frontend's grid renderer draws the correct palette color for each of
  the 10 ARC colors.

## V2: It learns something

**Delivers:** R0 (first evidence), R2, R5, R7

**Build plan**

1. Build `trainers/ppo/`: the color-embedding + conv/attention encoder and
   factored action head (ADR-0008), rollout buffer, GAE, clipped surrogate
   objective (ADR-0004). `train.py --algo ppo --task_id <id>` trains one
   fresh policy per task at solve-time (ADR-0008) — no shared/pretrained
   encoder across tasks; running the whole curated subset means looping (or
   parallelizing across cores) over `task_id`s, one `runs/<run_id>/` each.
2. Implement the dense delta-shaped reward (ADR-0005) in `arc_env/`.
3. Vendor `re-arc` (`third_party/re-arc/`) and use it to generate additional
   training instances per task, beyond ARC's native ~3-5 train pairs, for
   PPO's rollouts — this is the concrete mitigation for the sample-efficiency
   risk flagged in `docs/research/rl-evolutionary-survey.md` (ARCLE's own
   finding that sparse ARC signal alone stalls PPO).
4. Wire periodic checkpointing and periodic evaluation-episode logging
   (every N updates, log one eval episode per task to `episodes/`) plus
   `metrics.jsonl` (reward, success rate per update) to `runs/<run_id>/`.
5. `viz/frontend`: add the training dashboard (reward curve, success-rate
   curve) polling `metrics.jsonl`, and a run/episode picker so late-training
   eval episodes can be replayed next to early-training ones.
6. `train.py --algo ppo --task_id <id> [hyperparameter flags]` CLI entrypoint.

**Demo:** launch `train.py --algo ppo`, watch the dashboard's reward/
success-rate curves update over the run, then replay an early-training vs.
late-training episode on the same task side by side to see qualitative
improvement (or a documented lack thereof, per PLAN.md Open risks).

**Rests on assumptions:** Q7 (invalid actions are a no-op + penalty, episodes
hard-terminate at a max-step budget) — if the budget is badly tuned,
episodes may end before or long after the agent could plausibly finish;
only a config constant needs to change.

### Test plan

#### End-to-end

- Running `train.py --algo ppo` against the single-task PPO-sanity fixture
  (PLAN.md Testing approach) for the configured update budget produces a
  mean evaluation-episode reward, over the last 10% of updates, strictly
  greater than the mean reward of 100 random-policy episodes on the same
  task measured in V1 — a concrete, non-subjective pass/fail comparison
  logged in `metrics.jsonl` and checkable by script, not by eyeballing a
  chart.

#### Integration

- `metrics.jsonl` rows parse into the dashboard's expected schema and render
  a monotonically-timestamped curve.
- A periodic checkpoint can be loaded and used to resume training without
  error.

#### Unit

- The reward function returns the expected delta for a hand-constructed
  before/after grid pair.
- GAE advantage computation matches a hand-computed value for a small
  fixed trajectory.

## V3: Full action space, variable output shape

**Delivers:** R1 (complete)

**Build plan**

1. Add the 30×30 scratch canvas and `commit`/`crop` action to `arc_env/`
   (ADR-0002).
2. Extend the task subset to include variable-output-shape tasks from
   `third_party/ARC-AGI/data/training`.
3. Extend the visualizer's grid renderer to show the scratch canvas during
   an episode and the committed (cropped) final grid at episode end.
4. Retrain PPO (V2's trainer, unchanged) on the extended task subset.

**Demo:** replay an episode for a variable-output-shape task and see the
agent paint within the scratch canvas, then commit/crop to the final output
shape, rendered correctly in the visualizer.

**Rests on assumptions:** none beyond ADR-0002 itself, which this slice
exists to validate.

### Test plan

#### End-to-end

- A fixture variable-output-shape task's known-correct `arc-dsl` solver,
  replayed through the env with the canvas/commit mechanism, produces the
  exact expected output shape and content.

#### Integration

- The visualizer correctly distinguishes "scratch canvas mid-episode" from
  "committed final grid" when rendering a fixture episode that uses `commit`.

#### Unit

- `commit`/`crop` on the scratch canvas produces the expected sub-grid for a
  range of hand-constructed painted regions (corners, full canvas, single
  cell).

## V4: Evolutionary fast-follow

**Delivers:** R0 (second evidence source), R3, R8 (format only, not the BC step itself)

**Build plan**

1. Build `trainers/gp/`: population of DSL-program ASTs (same action subset
   as V1/V3), crossover/mutation operators, fitness = fraction of train
   pairs matched (falling back to the ADR-0005 similarity measure as a
   tiebreaker) (ADR-0003).
2. Reuse `arc_env/`'s executor to run a candidate program against a task's
   train pairs for fitness evaluation.
3. Log GP run metrics (best fitness per generation) and the best-found
   program's execution trace as an `episodes/*.jsonl` file, in the same
   format V1-V3 already produce (ADR-0006), so the visualizer needs no
   changes to replay it.
4. `train.py --algo gp --task_id <id> [hyperparameter flags]` CLI entrypoint.

**Demo:** run `train.py --algo gp` against a handful of tasks, watch best-
fitness-per-generation in the dashboard (same charting code as PPO's reward
curve), and replay a GP-found solving program's trajectory in the exact same
visualizer used for PPO episodes.

**Rests on assumptions:** none new — this slice is the direct payoff of
ADR-0003's shared-substrate decision, and its test is whether that sharing
actually worked with zero visualizer changes.

### Test plan

#### End-to-end

- GP run against a fixture task with a known short solving program (e.g.
  `vmirror`) finds a matching program within a small, fixed generation
  budget.

#### Integration

- A GP-found program's execution trace, logged as `episodes/*.jsonl`, loads
  and replays in the visualizer with no code changes beyond V1's.

#### Unit

- Crossover and mutation operators always produce a syntactically valid
  (type-correct) program given valid parents/inputs.
- Fitness evaluation on a hand-constructed program/task pair matches a
  hand-computed expected value.

---

# The object-representation rewrite (V5–V9, ADR-0029)

A **new, parallel track**, not an in-place edit of V1–V4. It builds an
object-centric representation with a *typed, compositional action grammar*
alongside the shipped single-grid agent, which keeps running and stays the
benchmark until the new track reaches parity (V9's cutover). The ordering is
riskiest-first, and the risk this whole track exists to resolve is named up
front by the F15 POC (`research/arc-object-poc/`): the object model *expresses*
the set-op family cleanly (8/8), but a *free-form* search over it fails exactly
like the flat 85-action space (0/8) — only a *typed grammar* over the same
actions discovers solutions (8/8). So **the grammar is the headline deliverable,
not the object model alone**, and V5–V6 confront the open question the POC left:
does that grammar generalize past the 8 tasks it was validated on.

Note this deliberately reintroduces the typed composition ADR-0001 flattened
away ("a flat list of `(primitive, args)`, no AST"). The POC is the evidence
that flatness — not the choice of primitives — is what starved the search; the
grammar is the correction, over an object substrate.

## V5: Object substrate + typed action grammar

**Delivers:** ADR-0029 commitments #1 (grammar) and #2 (object model), as a
parallel `object_env/` package with no dependency from the shipped `arc_env/`.

**Build plan**

1. Promote the POC's object model into a real package (e.g. `object_env/`):
   an `Obj` with queryable attributes (color, cells, bbox, shape signature,
   region membership, role) and an `ObjState` (grid as ground truth + named
   object/region/index-set slots), plus segmentation and region derivation.
   Reuse `arc_env._dsl` primitives as the executor underneath (ADR-0001
   generalized, not discarded).
2. Define the **typed action grammar**: every action declares typed input/output
   slots over a small type set (`Grid`, `Region`, `IndexSet`, `Object`,
   `Color`, `SetOp`, `Axis`), and only type-valid compositions are constructible.
   Generalize the vocabulary *past* the set-op family (the POC's 5 verbs) to
   object selection-by-attribute, move/recolor/paint-object, and canvas/commit —
   enough that the grammar isn't a single-cluster tool.
3. Add **derived-color actions** (ADR-0029 #3): attribute queries like "the
   region's background / least-common color", so args can be grid-derived
   instead of hardcoded.
4. Promote the POC's `verify.py` into the track's regression harness.

**Demo:** replay a typed-grammar program solving a set-op task
(e.g. `6430c8c4`) through `object_env` from the CLI, and show the same grammar
expressing a non-set-op task (an object move/recolor) — evidence the vocabulary
generalizes past the cluster it was born from.

**Rests on assumptions:** that a single typed grammar spans task families beyond
the 8 the POC validated — the central open risk this slice exists to test. If it
doesn't, the grammar's type set / verb set changes, not the object substrate.

**Outcome (delivered 2026-09-16, `object_env/`):** the risk cleared — one typed
grammar (17 verbs) expresses a 16-task fixture spanning 5 families (8 set-op +
move / recolor / crop / canvas / transform), all reproduced exactly through
`object_env`. Two type-set adjustments surfaced (the "verb/type set changes, not
the substrate" outcome the slice anticipated): `Direction` (4-way move, vs the
2-way split `Axis`) and `Size` (canvas dims). Fork decisions as approved: named
typed slots (not SSA); moderate basket; a shipped enumerator. The discovery
proof (`object_env/search.py`) also sharpened the POC's Gate-2 finding: the
shallow families are discovered *from scratch* by type-directed search, but the
deep depth-5 programs (set-op, and a depth-5 recolor) are a needle in ~170k
type-valid depth-5 skeletons — so uniform from-scratch discovery of them is the
deceptive-landscape problem ADR-0029 hands to V6's biased/learned search (#4).
Given the grammar-*derived* skeleton (not hardcoded), arg-search rediscovers all
16 in a few thousand samples — the POC's fixed-skeleton 8/8, generalized.

### Test plan

#### End-to-end
- Every curated task with a known object-space program is reproduced exactly by
  running that program through `object_env` (the object-track analogue of
  `tests/test_dsl_regression.py`).

#### Integration
- A type-invalid composition is rejected at construction time (the grammar
  actually constrains the space), and a type-valid one executes.

#### Unit
- Segmentation + attribute extraction on hand-built grids yields the expected
  objects (color, bbox, shape signature, region).
- Each derived-color action returns the expected grid-derived color for
  representative inputs.

## V6: GP over the object grammar (discovery)

**Delivers:** first evidence the rewrite breaks the ceiling — a GP trainer over
the typed grammar that solves set-op tasks the current GP gets 0/8 on, at
parity elsewhere.

**Build plan**

1. A GP trainer over a **grammar-typed genome** (each gene constrained by the
   grammar's types/positions, not a flat pick-any-of-N list) — productionizing
   `research/arc-object-poc/{gp_object,grammar_search}.py` into a real trainer.
2. Reuse the ADR-0005 similarity reward for fitness; log the best program's
   trace as `episodes/*.jsonl` in a schema V8 formalizes.
3. Lean on the derived-color actions so a found program generalizes across
   `re-arc` training instances, not just the real pairs.
4. Run serially locally for smoke; the full sweep on **RunPod** (`runpod-jobs`),
   per the compute directive.

**Demo:** object-track GP solves the 8 set-op tasks (0/8 for the current GP in
the 2026-09-15 pass) and is at parity on a sample of already-solved curated
tasks — a head-to-head table.

**Rests on assumptions:** that grammar-typed search generalizes past the 8
(V5's risk, now under search rather than hand-written programs).

**Outcome (2026-09-22, `docs/results/training-pass-runpod-v6-20260922b.md`,
PR #82) — negative, and stopped, not deferred:** `trainers/gp_object` scored
**0/8** on the exact set-op cluster this slice was built to solve (matching
flat GP's own 0/8), and is *below parity* on the non-set-op sample (2/8 vs.
flat GP's 5/8). Both halves of V6's own demo ask came back no. ADR-0029
commitment #4 (demonstration/curriculum seeding as a follow-up attack on the
fitness-landscape problem this result points to) is **not pursued** — the
user decided V6 gets no further attempts, a harder stop than a bounded
one-more-try. See ADR-0030/F17 (`docs/questions/f17-llm-program-synthesis-pivot.md`):
this result, together with two other disqualifying facts unrelated to V6
itself, is part of why the project pivoted off this whole track (V5-V9)
rather than continuing to iterate on it. V7-V9 below are not started and will
not be; kept as the historical plan for what this track intended.

### Test plan

#### End-to-end
- GP over the object grammar finds an exact solution for a fixture set-op task
  within a fixed generation budget (the object-track analogue of V4's
  `vmirror` test).

#### Integration
- A GP-found object program's logged trace reloads and re-executes to the same
  result.

#### Unit
- The typed crossover/mutation operators only ever produce grammar-valid
  programs.

## V7: PPO over the object representation (encoder + typed head)

**Delivers:** ADR-0029 commitment #4 — a set encoder and a grammar-constrained
action head, trained per task, warm-startable from object-GP.

**Build plan**

1. An object-**set encoder** (per-object embedding + relational attention),
   replacing V2's CNN-over-2-channels for this track.
2. A **grammar-constrained action head**: the policy proposes `(verb,
   object-ref, arg)`, masked to type-valid choices given the current object
   slots — the structure the POC showed is decisive, now in the RL head.
3. Object-GP → PPO warm-start, the object-track analogue of ADR-0009.
4. Eval against the current PPO on a shared task sample.

**Demo:** PPO learns a set-op task on the object track; warm-start rescues a
task cold PPO can't, mirroring the ADR-0009 result on the new representation.

**Rests on assumptions:** that a set encoder + masked typed head learns from the
ADR-0005 reward — new architecture risk; falls back to GP-only (V6) for the
track's coverage claim if PPO underperforms, exactly as the current pipeline
leans on GP+warm-start.

### Test plan

#### End-to-end
- Object-track PPO beats random-policy mean reward on a single-task sanity
  fixture (the object analogue of V2's non-subjective PPO-sanity test).

#### Integration
- The action mask never admits a type-invalid action; a checkpoint round-trips.

#### Unit
- The set encoder is permutation-equivariant over object order for a
  hand-built object set.
- Advantage/rollout bookkeeping matches a hand-computed value on a small
  trajectory.

## V8: Object-aware episode log + replay

**Delivers:** the logging schema and minimal replay the object track needs;
the human-facing editor is F16/ADR-0028, built on this same object model.

**Build plan**

1. Extend the episode-log schema to carry the object set per step plus the
   `annotation` field (additive; `runs/` is local + gitignored, a normal schema
   evolution, same ethos as ADR-0020).
2. Minimal object-aware replay in the visualizer (objects/layers overlay) —
   enough to debug and demo a run; the full layers/objects editor UI is F16.

**Demo:** replay an object-track episode and step through object-level actions
with the objects/layers overlay.

**Rests on assumptions:** none beyond the schema shape, which is local and
disposable.

### Test plan

#### End-to-end
- An object-track run written by V6/V7 reloads through the read path and
  replays without error, showing correct per-step object sets.

#### Integration
- The frontend parses the extended schema and renders the object overlay for a
  fixture episode.

#### Unit
- The log round-trips an object set + annotation losslessly.

## V9: Parity, coverage, and cutover

**Delivers:** the go/no-go on making the object track the default — a full
head-to-head against the 2026-09-15 baseline.

**Build plan**

1. Full curated-set pass on the object track (RunPod), head-to-head vs the
   2026-09-15 numbers (GP 70/91, PPO+warm 69/91), plus a run at the 21
   currently-unsolved tasks.
2. A combined results doc under `docs/results/`, same shape as
   `training-pass-2026-09-15-combined.md`.
3. **Cutover** only if the object track is ≥ baseline on the curated set *and*
   clears some of the 21: make it the default `train.py` path; keep the
   single-grid track available (or retire it) per the results.

**Demo:** a results table showing the object track at or above baseline on the
curated set and N of the previously-unsolved 21 now solved — the ceiling moved.

**Rests on assumptions:** that parity is actually reached — this slice is the
decision point; a miss means the object track stays a parallel experiment and
the single-grid agent remains default, with the gap documented.

### Test plan

#### End-to-end
- The full-pass harness runs both tracks and emits the combined results doc
  a script (not eyeballing) can check the parity/coverage claims against.

#### Integration
- The cutover leaves `make test` green and the shipped read-only visualizer
  paths working.

#### Unit
- The results aggregation reproduces a hand-computed solved-count from a small
  fixture of per-task outcomes.

---

# The LLM program-synthesis pivot (V10+, ADR-0030)

**V5-V9 above are frozen as of 2026-09-22 — not deleted, not restarted.** V6's
own negative result (its Outcome note above), plus two disqualifying facts
unrelated to V6 (no path to a Kaggle/ARC-AGI-2 entry existed at all, and the
curated-DSL action space's core growth mechanism can't run against held-out
tasks), are why the project pivoted rather than continuing to iterate on the
object-grammar track. Full rationale: `docs/adr/0030-llm-program-synthesis-pivot.md`,
`docs/questions/f17-llm-program-synthesis-pivot.md`.

The new track: an LLM (hosted API, frozen weights — no test-time training)
proposes a Python program hypothesized to solve a task from its demonstration
pairs, the program runs against the task's own training inputs, discrepancies
between predicted and expected outputs are fed back to the LLM, and it
revises the program. Repeat to a fixed budget, then apply the final program
to the task's test input(s). Inference-only, so no GPU training runs and no
`runpod-jobs` compute provisioning the way V2/V6 needed it — cost is LLM API
spend instead.

Sequencing is risk-first, same philosophy as V1-V9: get a real, scored
competition entry path working end to end (even a weak one) before optimizing
the synthesis loop itself, since "no entry vehicle exists at all" was one of
the three facts that forced this pivot.

## V10: ARC-AGI-2 vendored + frozen-stack baseline on record

**Delivers:** the data this whole pivot needs to measure against, plus a
documented, evidence-based answer to "how did the old system actually do on
the benchmark that matters" instead of an assumption.

**Build plan**

1. Vendor `arcprize/ARC-AGI-2` (Apache-2.0, same JSON task-pair format as
   ARC-AGI-1: 1,000 public training tasks, 120 public evaluation tasks) under
   `third_party/ARC-AGI-2/`, read-only, same convention as
   `third_party/ARC-AGI/`.
2. Extend (not fork) `arc_eval/` and any shared loader code so a `tasks-dir`
   can point at either `third_party/ARC-AGI/data/*` or
   `third_party/ARC-AGI-2/data/*` uniformly.
3. Run the frozen `arc_env`/`trainers/{ppo,gp}` stack, unmodified, against
   ARC-AGI-2's 120 public evaluation tasks (no re-curation, no new ADRs
   adding actions for them — that would defeat the point) and record the
   result under `docs/results/`.

**Demo:** `python -m arc_eval --tasks-dir third_party/ARC-AGI-2/data/evaluation
--submission <old-stack-output>.json` produces a real, low (expected
near-zero) pass@1/pass@2 number, written up alongside the reasoning for why —
the evidence backing this pivot's own ADR, not just an assertion in it.

**Rests on assumptions:** none new — `arc_eval`'s scoring boundary already
exists and already speaks the right submission shape; this slice is data
plus a measurement, not new mechanism.

### Test plan

#### End-to-end
- `arc_eval` scores a small fixture of ARC-AGI-2-shaped tasks (train/test
  pairs, same JSON shape as ARC-AGI-1) identically to how it already scores
  ARC-AGI-1 fixtures — confirms the loader generalizes across both datasets
  by construction, not by a parallel code path.

#### Integration
- The frozen-stack baseline run against the real 120-task public evaluation
  set completes and produces a submission file `arc_eval` can score without
  modification.

#### Unit
- `third_party/ARC-AGI-2/` task files parse into the same in-memory shape
  `third_party/ARC-AGI/` files already do.

## V11: Kaggle submission pipeline

**Delivers:** the actual, currently-missing entry vehicle — a public,
open-sourceable notebook (or a thin notebook wrapper around a scripted
pipeline) that turns any solver satisfying `arc_eval`'s `Challenge` interface
into a scored Kaggle submission.

**Build plan**

1. A submission-format writer: given a solver callable, produce the
   `attempt_1`/`attempt_2`-shaped JSON Kaggle's ARC-AGI-2/3 tracks expect,
   for every test input across every task in a directory — reusing
   `arc_eval`'s existing boundary and shapes, not a second implementation of
   them.
2. A minimal notebook (or a script structured so a notebook can thinly wrap
   it) that runs end to end inside Kaggle's own notebook environment
   constraints (no assumption of this project's local `uv`/`make` tooling
   being present).
3. A dry run against the V10 baseline solver first (even though it's frozen
   and expected to score near-zero) — proves the pipeline itself works
   before the real LLM-synthesis solver exists, isolating "does the
   submission mechanism work" from "does the new solver work."

**Demo:** a notebook run, using the V10 frozen-stack solver as a placeholder,
produces a submission file scorable by `arc_eval` with no manual
post-processing.

**Rests on assumptions:** that Kaggle's notebook environment doesn't need
anything beyond what a from-scratch notebook can install/import — unverified
until a real notebook is drafted; if wrong, only the packaging changes, not
`arc_eval`'s scoring contract.

### Test plan

#### End-to-end
- The full pipeline (solver → submission JSON → `arc_eval` scoring) runs
  against a small fixture task set with a known expected score.

#### Integration
- The submission writer's JSON validates against the exact shape
  `arc_eval/evaluator.py` already parses, for tasks with multiple test
  inputs and variable output shapes (the same edge cases the existing
  evaluator's docstring already calls out).

## V12: LLM propose→execute→diff→refine synthesis loop

**Delivers:** the real new solver — ADR-0030's core mechanism, replacing the
old system's curated action space and per-task RL/GP training with an
inference-only program-synthesis loop.

**Build plan**

1. A prompt that gives the LLM a task's demonstration input/output pairs and
   asks for a Python function (plain Python/numpy per ADR-0030's assumed
   default A1; `third_party/arc-dsl/` available as an optional import, not
   mandatory).
2. A sandboxed executor that runs the proposed program against the task's
   own training inputs only (never the hidden test outputs — reuses
   `arc_eval`'s blind boundary) and captures either the predicted grids or a
   structured error (exception, wrong shape, wrong cells at which
   positions).
3. A feedback formatter that turns an execution result into concrete,
   structured input for the next LLM call — not a bare "wrong, try again."
4. A fixed per-task attempt/token budget (ADR-0030 A3); on exhaustion, submit
   the best-scoring program found so far (by training-pair match fraction),
   not a blank/default answer.
5. Provider/model choice and per-task cost budget (ADR-0030 A2) — a
   provisional choice to unblock this slice, revisited once real cost data
   exists.

**Demo:** the loop solves a held-out ARC-AGI-1 evaluation-split task end to
end from its own demonstration pairs alone, logging every attempted program
and the feedback that drove each revision — replayable evidence of the
propose→execute→diff→refine cycle actually happening, not just a final
answer.

**Rests on assumptions:** that iterative execution-feedback refinement
meaningfully improves over single-shot proposal within a practical budget —
the central open risk this slice exists to test, same role V2 played for
whether PPO learns anything at all.

### Test plan

#### End-to-end
- The loop reaches an exact match on a fixture task deliberately chosen to
  need at least one refinement round (the first proposal is wrong, a later
  one is right) — proves the feedback loop does something, not just that a
  single LLM call can get lucky.

#### Integration
- The executor never lets a proposed program see a task's hidden test
  outputs, verified the same way `arc_eval`'s existing boundary is tested.
- A program that raises an exception, times out, or returns the wrong shape
  produces a structured (not crash-the-harness) feedback message.

#### Unit
- The submission-format writer output for this solver validates identically
  to V11's placeholder-solver output — no special-casing per solver type.

## V13: Score, iterate, and the north-star metric switch

**Delivers:** the first real, trackable numbers on the metrics that now
matter, replacing "curated ARC-AGI-1 tasks solved" for good.

**Build plan**

1. Run V12's solver against the full held-out ARC-AGI-1 evaluation split and
   record pass@1/pass@2 via `arc_eval` — the near-term north-star metric
   (ADR-0030).
2. Run it against ARC-AGI-2's 120 public evaluation tasks (vendored in V10)
   and record the same — the metric that actually matters for the
   competition goal.
3. Compare against the V10 frozen-stack baseline on both, in one combined
   results doc under `docs/results/`, same shape as prior full-pass records.
4. Decide the next iteration lever from where the numbers actually land
   (prompt/feedback design, attempt-budget tuning, or — if this plateaus well
   below competitive — reopening the test-time-training question F17
   explicitly left as a future fork, not a closed door).

**Demo:** a results table with three columns (frozen-stack baseline, new
LLM-synthesis solver) across two benchmarks (ARC-AGI-1 eval split,
ARC-AGI-2 public eval) — the pivot's own scorecard.

**Rests on assumptions:** none beyond V10-V12 already landing; this slice is
measurement and a decision point, not new mechanism.

### Test plan

#### End-to-end
- The full-pass harness runs both the frozen-stack baseline and the new
  solver across both benchmarks and emits the combined results doc a script
  can check the reported numbers against.

#### Unit
- The results aggregation reproduces a hand-computed pass@1/pass@2 from a
  small fixture of per-task outcomes, for both benchmarks.
