# arc-agi

Revamping this project to tackle ARC-AGI-1 with a reinforcement-learning and/or
evolutionary-algorithm agent, plus a local visualizer to watch training runs and
watch a trained agent "play" a task step-by-step like a game.

Planning artifacts (read these before making architectural changes):
- `docs/QUESTIONS.md` — decision register: every open question, its status, and
  its answer. A few questions carry a long running history instead of a short
  one; those live in their own file under `docs/questions/`, linked from the
  register row.
- `docs/PLAN.md`, `docs/SLICES.md`, `docs/adr/` — the agreed plan, vertical
  slices, and the ADRs behind them. Implement against these; don't
  re-litigate a decision that's already recorded there.

## Repo layout

- `third_party/ARC-AGI/` — vendored official ARC-AGI-1 dataset
  (`data/training`, `data/evaluation`) and the human testing interface. Plain
  tracked files (not a submodule) — treat as read-only upstream content.
- `third_party/arc-dsl/` — vendored Michael Hodel's `arc-dsl` (ADR-0001): the
  action-space DSL/executor (`dsl.py`, `arc_types.py`, `constants.py`) plus
  `solvers.py` (400 known-correct per-task solver programs, used as a free
  regression-test fixture). Read-only, see `third_party/arc-dsl/README.md`.
- `third_party/re-arc/` — vendored Michael Hodel's `re-arc` (per-task
  synthetic-instance generators). Two deliberate deviations from a verbatim
  vendor (own `dsl.py` kept separate from `arc-dsl`'s; trimmed
  `matplotlib`-free `utils.py`) — see that dir's README.
- `arc_env/` — the Gymnasium-style ARC environment: the curated
  `arc-dsl`-primitive action space (`actions.py` — 85 actions as of
  ADR-0026: structural transforms including the 4 self-concatenation
  actions, `fill_cell`, `canvas`, `canvas_mostcolor`,
  `swap_two_least_colors`, `commit`, plus the object-selection
  mechanism's 14 actions (`select_largest`/`select_smallest`/
  `select_by_color`/`select_unique_color`/`select_largest_no_diag`/
  `select_tallest`/`select_largest_multicolor`/`commit_selection`/
  `crop_to_selection`/`delete_selected`/`recolor_selected`/`move_selected`/
  `paint_selected_at`/`stamp_selected`) threading a "currently selected
  patch" side-channel, ADR-0002/ADR-0010/ADR-0011/ADR-0012/ADR-0013/
  ADR-0015/ADR-0016/ADR-0019 — `crop_to_selection` is the same crop
  `commit_selection` does but, deliberately, under a different action name
  so it does *not* end the episode, letting a further transform run on the
  cropped result (ADR-0011/`fitness.py` both key episode-termination off the
  literal action name, not `Action.kind`); `stamp_selected` (ADR-0016) shifts
  the selected indices by one of 8 fixed directions (its own menu, separate
  from `move_selected`'s 4-direction one) and fills the grid with a color
  there, without clearing the original selected cells and without
  invalidating the selection, so one selection is reused across several
  stamp calls in a row; `canvas_mostcolor`/`swap_two_least_colors`
  (ADR-0019) are two more derived actions (`canvas` fed a grid-derived color
  instead of an agent-chosen one; a fully self-contained zero-arg swap of a
  grid's two least-common colors) — same "derived, not drawn 1:1 from a
  single `dsl` call" pattern `fill_cell`/`canvas` already established; and
  ADR-0020 adds a region-scoped, second named selection slot ("a"/"b",
  keyed internally by int 0/1) via 4 more actions —
  `select_by_color_in_region`, `combine_slots` (intersect/union/symdiff),
  `fill_new_canvas`, `fill_onto_region` — unlocking 8 more curated tasks
  that combine indices from two grid halves via a set op; and ADR-0021 adds
  6 more derived actions — `select_leastcolor`, `select_all`,
  `select_by_size`, `select_largest_multicolor_no_diag`,
  `switch_least_most_colors`, `fractal_expand_cellwise` — unlocking 7 more
  curated tasks; and ADR-0022 widens the region menu from 4 halves to 7
  entries (adding `left_third`/`middle_third`/`right_third`) and adds 2
  more actions, `fill_slot_onto_region` and `replace_region_and_fill`,
  unlocking 2 more curated tasks; and ADR-0023 (Bucket C) adds 3 more
  derived `"transform"`-kind actions, no new mechanism, no new
  `Action.kind` — `fill_inbox_by_dot_color` (finds an agent-chosen dot
  color's cells, takes their bounding subgrid, and fills the *interior* box
  with that subgrid's own trimmed least-common color),
  `fill_quadrant_from_colors` (reads each of 3 quadrants' own agent-chosen
  color and fills its found indices onto the 4th, bottom-right quadrant,
  sequentially), and `repeat_mirror_tile` (zero-arg, fully derived:
  `vconcat`s the grid with its own hmirror, minus the shared edge row,
  twice) — unlocking 3 more curated tasks. A broader audit prompted by F13
  Stage 2's last deferred cluster ("Bucket C") found none of that cluster's
  tasks actually needs a third-generation selection mechanism ("hold the
  pre-crop original grid alongside a derived crop"); every genuine win was
  reachable as an ordinary derived action referencing `grid` more than once
  inside one atomic function body, the same correction ADR-0021 already
  made once for `007bbfb7`/`80af3007`/`8f2ea7aa` — `7c008303`, `c9f8e694`,
  and `017c7c7b` remain uncurated, a reasoned no-go (each needs real
  structural-detection logic to generalize past its own fixed json, not a
  parameterization swap)); and ADR-0024 (a re-check of the `free_by_name`
  bucket the ADR-0023 audit left unexamined) adds 4 more zero-arg
  `"transform"`-kind actions, no new mechanism, no new `Action.kind` —
  `quad_rotate_tile` (2x2-tiles the grid with its own `rot90`/`rot270`/
  `rot180`), `quad_mirror_tile` (stacks `hconcat_self_vmirror`'s own block
  with that block's `hmirror`), `stack3_vmirror_tile` (a related but
  distinct tiling — concat order and stack count differ enough that it
  isn't reducible to `quad_mirror_tile`), and `left_third` (zero new logic
  at all: it's ADR-0022's existing private `_left_third` helper, already
  used internally by the region-scoping menu's `_REGIONS`, now also
  exposed as its own standalone action) — unlocking 7 more curated tasks
  (`46442a0e`/`7fe24cdd` via `quad_rotate_tile`, `3af2c5a8`/`62c24649`/
  `67e8384a` via `quad_mirror_tile`, `8d5021e8` via `stack3_vmirror_tile`,
  `2dee498d` via `left_third`); two more `free_by_name` candidates
  (`0520fde7`, `a699fb00`) were checked and found not to generalize
  against fresh `re-arc` instances, left uncurated same as ADR-0023's
  no-gos; and ADR-0025 (a same-day follow-up working the audit's next two
  highest-value buckets, `one_new_primitive` and a `free_by_name`
  re-check) adds 10 more zero/low-arity `"transform"`-kind actions, no new
  mechanism, no new `Action.kind`, and — for the first time since
  ADR-0001 — 5 brand-new `arc-dsl` primitives this module had never called
  before: `delta`, `frontiers`, `palette`, `dedupe`, `asobject`.
  `fill_delta_by_color(dot_color, fill_color)` (two agent-chosen
  `COLOR_ARG`s) fills a color's bounding box minus the color's own cells
  (`delta`); `fill_frontiers` fills every full-width/height single-color
  row/column (`frontiers`) with a fixed color; `switch_palette_then_
  zero_five` reads a two-color grid's palette (`palette`), switches the
  two colors, then zeroes the surviving reserved `5`;
  `tile_alternating_column_mirror` re-derives its tile unit and count from
  the grid's own height/width rather than the literal solver's hardcoded
  crop shape, needing zero new primitives; `dedupe_grid_both_axes` crops
  to the bounding box of the grid's literal non-zero cells (not
  `dsl.objects`'s `mostcolor`-based background autodetection, which picks
  the wrong background here) then collapses repeated rows/columns on both
  axes (`dedupe`); `fill_holes_in_object_bbox` crops to the grid's single
  foreground object's bounding box (using `diagonal=True` connectivity,
  not the literal solver's `diagonal=False`, since `re-arc`'s generator
  grows the shape via 8-connected neighbors and a 4-connected segmentation
  can wrongly split it), replaces the grid's own most-common color with a
  fixed fill color inside the crop, then composites the recolored crop
  back onto the original grid via `asobject`+`shift`+`paint`;
  `tile_by_mostcolor` tiles the grid by its own height/width (not the
  literal solver's hardcoded 3x3) using that same `asobject`+`shift`+
  `paint` composition; `paint_vmirrored_righthalf_onto_lefthalf` needs no
  new primitive at all; `fill_nonsingleton_foreground` auto-detects the
  grid's one foreground color in plain Python (not a hardcoded color) and
  recolors every non-singleton object of that color; `recolor_objects_by_
  size` fills size-1/2/3 foreground objects to three fixed colors by
  explicit size rather than the literal solver's background-color-
  dependent final replace. Unlocks 10 more curated tasks: `32597951`,
  `c1d99e64`, `f76d97a5`, `e9afcf9a`, `90c28cc7`, `6d75e8bb`, `c3e719e8`,
  `e3497940`, `67385a82`, `e8593010`. Two `one_new_primitive` candidates
  (`05f2a901`/`gravitate`, `3de23699`/`fgpartition`) initially looked
  promising (~96%/~94% on fresh `re-arc` instances) but a dedicated
  root-cause follow-up found each failure mode inherent to the primitive
  itself, not fixable by any action-level design, and left them uncurated
  same as prior ADRs' no-gos; 6 more `free_by_name` candidates (`11852cab`,
  `3618c87e`, `47c1f68c`, `aedd82e4`, `cce03e0d`, `e98196ab`) were also
  checked and found not to generalize. See ADR-0025 for the full audit; and
  ADR-0026 (a same-day follow-up working the audit's next-highest-value
  bucket, `two_new_primitives`, folded together with one leftover
  `one_new_primitive` task too small to be its own pass) adds 13 more
  zero-arg `"transform"`-kind actions, no new mechanism, no new
  `Action.kind`, and 10 more brand-new `arc-dsl` primitives: `numcolors`,
  `backdrop`, `outbox`, `shoot`, `center`, `normalize`, `leastcommon`,
  `box`, `hfrontier`, `connect`. The dominant finding this pass, more
  pronounced than ADR-0025's: for most candidates the literal solver's
  *choice of primitive* — not just its hardcoded constants — was wrong for
  `re-arc`'s actual generative concept, so every design below was traced
  against the actual `re-arc` generator source, not just a closer solver
  read — e.g. `67a423a3`'s real design needs `outbox` over a
  perpendicular-band intersection, not `neighbors` around a single hole
  cell, and `7b7f7511`'s real rule is a `tophalf`==`bottomhalf` structural-
  equality check, not a `portrait` (aspect-ratio) branch.
  `canvas_by_symmetry` builds a 1x1 canvas colored by whether the grid is
  symmetric under any of `hmirror`/`vmirror`/`dmirror`/`cmirror`;
  `tophalf_or_lefthalf_by_equality` returns `tophalf` if it equals
  `bottomhalf`, else `lefthalf`; `mirror_crop_by_rectangle_marker`
  auto-detects the marker color as whichever color's own cells form an
  exact solid rectangle and crops that color's bbox out of whichever of
  `hmirror`/`vmirror` no longer contains it; `diagonal_canvas_by_object_
  count` builds an NxN canvas (N = foreground object count) colored by the
  grid's own `mostcolor`/`leastcolor`, diagonal-filled in plain Python;
  `canvas_row_by_foreground_count` builds a 1-row canvas sized and colored
  by the grid's one foreground color's own cell count;
  `bar_chart_canvas_by_size4_count` builds a fixed-total-5 "bar chart"
  canvas from a count of size-4 objects; `upscale_by_numcolors_minus_one`
  (`dsl.upscale(grid, dsl.numcolors(grid) - 1)`) is shared by two tasks;
  `fill_outbox_of_band_intersection` finds two perpendicular color-bands by
  column/row bounding-box span (not connectivity-object identity, which
  fragments once one band cuts across the other) and rings their
  intersection's `outbox` with a fixed color; `shoot_diagonals_from_
  objects` shoots a diagonal from every same-colored object's own
  `ulcorner` (not one merged blob per color); `stamp_shape_at_singleton_
  echoes` normalizes an anchor shape and stamps it — shifted by each
  singleton marker echo's own center minus the anchor's own center, traced
  directly from the `re-arc` generator rather than the literal solver's
  fixed offset — at every singleton echo of the marker color;
  `crop_to_leastcommon_quadrant` returns whichever of the grid's 4
  quadrants is the statistical mode by value; `fill_backdrop_and_box_by_
  rarity` auto-detects the dot/background colors by rarity and classifies
  the remaining two colors into outline vs. interior by adjacency-to-
  background (thickness-independent, unlike a rarity ranking or a
  shape-equality test) before filling the merged bbox's interior/outline;
  `mirror_border_decoration` classifies two marker dots by position (not
  fixed color identity) and draws each half's own three-sided border plus
  a through-dot frontier. Unlocks 14 more curated tasks: `44f52bb0`,
  `7b7f7511`, `ff805c23`, `d0f5fe59`, `d631b094`, `1fad071e`, `ac0a08a4`,
  `b91ae062`, `67a423a3`, `5c0a986e`, `88a10436`, `88a62173`, `b548a754`
  (~96.7% on fresh `re-arc`, a documented degenerate edge case matching
  ADR-0025's `f76d97a5` precedent), `1bfc4729`. One candidate (`77fdfe62`)
  was checked and left uncurated: even the literal official solver crashes
  against fresh `re-arc` instances, since its hardcoded marker color is
  actually randomly chosen per instance — the real generator concept needs
  genuine structural detection, not a parameterization fix, same
  disposition as `7c008303`/`c9f8e694`/`017c7c7b`/ADR-0025's `3de23699`.
  See ADR-0026 for the full audit, verification numbers, and every
  design's exact composition. The
  task loader (`task_loader.py` — 91 curated tasks, 34 same-shape
  + 57 variable-shape), `env.py` (2-channel observation:
  grid + selection mask, now with values in {0,1,2} per ADR-0020's dual
  slots; `get_selected()` exposes the selection for episode
  logging), ADR-0005's dense reward (`reward.py`), extra
  practice-instance generation via `re-arc` (`re_arc.py`), and the JSONL
  trajectory/run-meta writers (`episode_log.py`), per ADR-0004/ADR-0006.
  `info["exact_match"]`, not the broader `terminated`, is what "solved"
  means once `commit` can end an episode without matching.
- `trainers/ppo/` — the ADR-0008 policy/value network (`network.py`),
  rollout collection with truncation-aware GAE (`rollout.py`, `gae.py`),
  the clipped-surrogate PPO update (`ppo.py`), and the ADR-0009 opt-in
  GP-to-PPO behavior-cloning warm-start (`warm_start.py`,
  `train.py --algo ppo --warm_start_from <gp_run_dir>`).
- `trainers/gp/` — the ADR-0003 evolutionary trainer: DSL-program genomes
  as flat gene lists (`genome.py` — no separate AST, same non-compositional
  action space PPO uses), fitness evaluation reusing `arc_env`'s executor
  and reward similarity (`fitness.py`), the generational loop (`evolve.py` —
  per ADR-0014, also snapshots each generation's own best program at
  `GPConfig.snapshot_interval`, not just the final one), and program replay
  for logging (`replay.py`).
- `object_env/` — the V5 object substrate + **typed action grammar**
  (ADR-0029, SLICES.md V5), a *new parallel track* to the shipped `arc_env/`
  single-grid agent (which stays the benchmark until V9 cutover). It must NOT
  be imported *by* `arc_env/`; reuse flows the other way (`object_env` leans on
  `arc_env._dsl` as the executor, and `arc_env.{task_loader,reward,re_arc}` in
  its harness). The headline is the grammar, not "objects instead of pixels":
  the F15 POC showed a *free-form* search over object actions fails exactly like
  the flat 85-action space (0/8), while a *typed grammar* over the same actions
  finds 8/8. Pieces: `types.py` (the `ArgType` set — dataflow types
  Grid/Region/IndexSet/Object plus parameter types Color/SetOp/Axis, extended
  by V5 with Direction/Size — and the first-class `Obj`); `state.py`
  (`ObjState` — bounded *named typed slots* grid/region_a·b/set_a·b/obj, the
  fork-1 decision over an SSA register file; `SLOT_TYPES` is the type-checker's
  source of truth); `objects.py` (segmentation + attribute selectors, baking in
  the same `dsl.objects(...)` connectivity variant each shipped selector uses);
  `colors.py` (derived colors — `DerivedColor("most"/"least")`, ADR-0029 #3, so
  args generalize across re-arc palettes); `actions.py` (the typed verb
  vocabulary — the POC's 5 set-op verbs plus select-by-attribute / object
  move·recolor·crop / canvas·transform — each an `Action` declaring its
  parameter slots and its symbolic read/write/clear footprint); `grammar.py`
  (the lever: `Program` **type-checks at construction** — a step referencing an
  unfilled slot raises `GrammarError`, so the skeleton *emerges from the types*
  rather than being hardcoded as the POC did; plus the type-directed
  `enumerate_skeletons`/`iter_fills`/`sample_program` search surface V6's
  GP/PPO will consume as an action mask); `programs.py` (the ~16-task fixture
  basket spanning families — 8 set-op + move/recolor/crop/canvas/transform, the
  fork-2 moderate basket; `build(task_id)` type-checks each); `verify.py` (the
  promoted re-arc generalization harness); `search.py` (the discovery proof —
  shallow families discovered *from scratch*, and every family rediscovered by
  arg-search within a grammar-*derived* skeleton, the POC's 8/8 method
  generalized: `uv run python -m object_env.search`); `cli.py` (the SLICES V5
  demo: `uv run python -m object_env.cli replay <task_id>`, a set-op *and* a
  non-set-op task through one grammar). Fully local/serial/seconds; GP over the
  grammar is V6 (RunPod). Tests: `tests/test_object_grammar_regression.py`
  (E2E, the object-track analogue of `test_dsl_regression.py`) and
  `tests/test_object_grammar.py` (grammar construction + enumerator validity,
  segmentation/attribute/derived-color units, fast discovery smoke).
- `train.py` — `train.py --algo ppo|gp --task_id <id>`: trains one
  dedicated PPO policy (ADR-0008) or evolves one dedicated GP population
  (ADR-0003) per task, logging to `runs/<run_id>/` in the same shape either
  way. GP now logs one `episodes/<id>.jsonl` per snapshotted generation
  (`00000-gen`, `00010-gen`, ... — zero-padded so they sort chronologically
  and, deliberately, before `best-program`) plus `best-program` itself
  (ADR-0014) — the visualizer's existing multi-episode picker shows this as
  an early-vs-late comparison with no GP-specific UI. `best-program` keeps
  its exact historical name/content unchanged, since `trainers/ppo/
  warm_start.py`'s `--warm_start_from` (ADR-0009) hardcodes that episode ID.
  PPO's `metrics.jsonl` rows carry two distinct signals per update:
  `success_rate`/`mean_reward` (noisy — averaged over that update's own
  small, shifting mix of re-arc-generated + native training-rollout
  episodes) and `eval_success`/`eval_reward` (the fixed held-out pair's
  greedy-policy outcome, only set on `eval_every` update rows) — don't read
  a `success_rate` swing as the policy regressing on the task without also
  checking `eval_success` (KAN-1177: a `success_rate` crash to 0% between
  adjacent updates was often just a small-sample artifact, `eval_success`
  unaffected). PPO also keeps a `checkpoints/best.pt` — the checkpoint with
  the best `eval_reward` seen so far in the run, not necessarily the last
  one (`is_new_best_eval` in `train.py`).
- `scripts/rollout_random.py` — random-policy rollout script (no training);
  writes `runs/<run_id>/` (gitignored, generated locally).
- `scripts/prune_runs.py` — `runs/` accumulates fast (a full curated-task
  pass alone is 50+ dirs) and nothing else cleans it up since it's
  gitignored; `make prune-runs` (dry run) / `make prune-runs YES=1` (delete)
  keeps only runs from the most recent `created_at` date(s) seen (see the
  script's docstring for the exact rule).
- `scripts/print_task.py` — prints an ARC-AGI-1 task's train/test grids
  straight to the terminal (`make print-task TASK=<id>`, or
  `uv run python scripts/print_task.py <id> [--pair train|test N] [--no-color]`)
  so an agentic coding session and the person reading its terminal output
  can look at the same grids without opening the visualizer or the raw
  JSON — searches both `third_party/ARC-AGI/data/training` and
  `data/evaluation`, validates `task_id` against an 8-hex-digit allowlist
  before touching the filesystem (same convention as `viz/backend/play.py`'s
  own `task_id`/`run_id` checks), and renders truecolor 2-char-wide cells
  using the exact same 10-color palette `viz/frontend/src/palette.ts`
  already transcribes from `third_party/ARC-AGI/apps/css/common.css`
  (ADR-0007) — falls back to a plain bracketed-digit rendering when stdout
  isn't a TTY, `--no-color` is passed, or `NO_COLOR` is set.
- `viz/backend/` — local HTTP server exposing `runs/` as JSON, including
  `metrics.jsonl` (`server.py`); also serves `viz/frontend/dist` so one
  process runs the whole visualizer. The run-browsing/dashboard routes are
  read-only, per ADR-0006/ADR-0007. `play.py` (F13 Stage 0, ADR-0017) is a
  separate, narrow write path: an in-memory `session_id -> ArcEnv` session
  store (guarded by a `threading.Lock()`, no persistence until an explicit
  save) behind `GET /api/actions` and `POST /api/play/start|<id>/step|<id>/
  save`, letting a human solve any of the 400 training tasks by hand,
  one curated action at a time, and save the result as a real `runs/
  <run_id>/` dir through the *unmodified* `EpisodeWriter` schema
  (`algo="human"`, automatically warm-start-compatible per ADR-0009).
  `server.py` only dispatches HTTP into `play.py`'s functions; `task_id`/
  `run_id` are validated against a path-traversal allowlist before either
  touches the filesystem.
- `viz/frontend/` — TypeScript + Canvas UI: training dashboard
  (reward/success-rate curves), dual side-by-side episode replay for
  early- vs. late-training comparison (Vite + Vitest), per ADR-0007, and a
  "Play" panel (`play.ts`, F13 Stage 0/ADR-0017) driving a live `/api/play/*`
  session so a human can solve a task by hand and save it as a run. Replay
  and Play both render the current object-selection (an amber outline over
  selected cells, `grid.ts`'s `computeCellRects`/`drawGrid`); Play's action
  picker is built generically from `GET /api/actions`, not hardcoded.
- `legacy/` — the original geometric-transform + color-bijection baseline
  (`baseline.py`, `evaluate.py`, `arc_io.py`). Kept as a reference/sanity-check
  baseline, not part of the new agent.
- `research/arc-ngps/` — a prior, half-built *supervised program-synthesis*
  scaffold. Superseded by `arc-dsl`/`arc_env` per ADR-0001; not deleted, but
  off the path to the shipped agent.
- `docs/` — planning artifacts (PLAN, ADRs, SLICES, QUESTIONS).

## Commands

- `make` (no target) — lists every available command; it is not `install` (that's just the first target in the Makefile, not the default goal).
- `make install` — `uv sync --group dev` (Python, includes `ruff`/`pytest`/`pip-audit`) + `npm ci` (frontend).
- `make test` (or `uv run pytest -m "not slow"` / `cd viz/frontend && npm run typecheck && npm test`) — the fast layer, no external services needed, ~10s (488 Python tests + 57 frontend). `make test-py-slow` (or `uv run pytest`) also runs the ~90s PPO-sanity e2e test (`tests/test_train_ppo.py`, marked `slow`; `test_train_gp.py`'s own e2e check is fast enough to already be in the default layer).
- `uv run ruff check .` — lint (config in `pyproject.toml`'s `[tool.ruff]`; excludes `third_party/`, `legacy/`, `research/` — only the shipped agent's own code is linted).
- `make rollout` — random-policy rollout over all curated tasks, writes `runs/demo/`.
- `make train` — `train.py --algo ppo --task_id 67a3c6ac --run_id demo` (edit the task_id, or pass `--algo gp`, for a different run).
- `make viz` — builds the frontend and starts the backend at `http://127.0.0.1:8000` (reads `runs/`; override the port with `make viz PORT=8001` if 8000 is taken).
- `make demo` — `train` + `viz` in one command: trains a fresh run, then opens the visualizer on it. If `runs/` already has something in it (`make rollout`/`make train` output, or any prior run), `make viz` alone is faster.
- `make print-task TASK=<id>` — prints an ARC-AGI-1 task's grids to the terminal (`scripts/print_task.py`); see that script's own `CLAUDE.md` bullet above.
- `git config core.hooksPath .githooks` — installs the pre-push hook: `ruff check .`, the fast test layer, frontend typecheck/tests, and a `gitleaks` secret scan (skipped with a warning if `gitleaks` isn't installed locally; CI runs it regardless). `.github/workflows/ci.yml` runs five jobs in parallel: `lint` (`ruff`), `python-tests`, `python-tests-slow`, `frontend` (adds `npm audit`), and `security` (`gitleaks` + `pip-audit --skip-editable`, skipping the local `arc-agi-agent` package and the CPU-only `torch` build since neither resolves on PyPI under those exact names/versions). Branch protection on `main` requires all five before merge. `.github/dependabot.yml` opens weekly update PRs for `uv`, `npm` (`viz/frontend`), and GitHub Actions.

## Git workflow

Commit, push, open PRs, and merge without asking first — this is standing
authorization, not a one-time approval. Concretely: after making changes,
commit them with a useful message (splitting into multiple logical commits/PRs
when changes have distinct risk profiles, e.g. a behavioral code change vs. a
docs-only change), push a branch, open a PR (`gh pr create`), wait for CI, and
merge once the required checks are green — all without pausing for
confirmation at each step. Still surface anything genuinely unusual (a failing
check that isn't a flake, a merge conflict, force-push, or anything else this
file's absence of a rule wouldn't obviously cover) rather than pushing through
silently.

## Research subagent policy

For ARC-AGI technical research tasks — DSL/action-space survey, RL or
evolutionary-algorithm literature and prior-art review, or similar deep-dive
research needed to settle an open decision in `docs/QUESTIONS.md` — spin up
**at most 2 subagents at a time**. This is a deliberate cap: keep research
focused and reviewable rather than fanned out into results nobody reads.
