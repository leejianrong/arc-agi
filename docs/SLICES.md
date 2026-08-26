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
6. `train.py --algo ppo --config ...` CLI entrypoint.

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
4. `train.py --algo gp --config ...` CLI entrypoint.

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
