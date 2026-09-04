# arc-agi

Revamping this project to tackle ARC-AGI-1 with a reinforcement-learning and/or
evolutionary-algorithm agent, plus a local visualizer to watch training runs and
watch a trained agent "play" a task step-by-step like a game.

Planning artifacts (read these before making architectural changes):
- `docs/QUESTIONS.md` — decision register: every open question, its status, and
  its answer.
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
  `arc-dsl`-primitive action space (`actions.py` — 36 actions as of
  ADR-0012: structural transforms including the 4 self-concatenation
  actions, `fill_cell`, `canvas`, `commit`, plus the object-selection
  mechanism's 9 actions (`select_largest`/`select_smallest`/
  `select_by_color`/`select_unique_color`/`commit_selection`/
  `delete_selected`/`recolor_selected`/`move_selected`/`paint_selected_at`)
  threading a "currently selected patch" side-channel, ADR-0002/ADR-0010/
  ADR-0011/ADR-0012), the task loader (`task_loader.py` — 29 curated tasks,
  14 same-shape + 15 variable-shape), `env.py` (2-channel observation:
  grid + selection mask; `get_selected()` exposes the selection for episode
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
  and reward similarity (`fitness.py`), the generational loop (`evolve.py`),
  and best-program replay for logging (`replay.py`).
- `train.py` — `train.py --algo ppo|gp --task_id <id>`: trains one
  dedicated PPO policy (ADR-0008) or evolves one dedicated GP population
  (ADR-0003) per task, logging to `runs/<run_id>/` in the same shape either
  way. PPO's `metrics.jsonl` rows carry two distinct signals per update:
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
- `viz/backend/` — read-only local HTTP server exposing `runs/` as JSON,
  including `metrics.jsonl` (`server.py`); also serves `viz/frontend/dist`
  so one process runs the whole visualizer.
- `viz/frontend/` — TypeScript + Canvas replay UI: training dashboard
  (reward/success-rate curves) and dual side-by-side episode replay for
  early- vs. late-training comparison (Vite + Vitest), per ADR-0007. Replay
  renders the current object-selection (an amber outline over selected
  cells, `grid.ts`'s `computeCellRects`/`drawGrid`).
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
- `make test` (or `uv run pytest -m "not slow"` / `cd viz/frontend && npm run typecheck && npm test`) — the fast layer, no external services needed, ~10s (364 Python tests + 33 frontend). `make test-py-slow` (or `uv run pytest`) also runs the ~90s PPO-sanity e2e test (`tests/test_train_ppo.py`, marked `slow`; `test_train_gp.py`'s own e2e check is fast enough to already be in the default layer).
- `uv run ruff check .` — lint (config in `pyproject.toml`'s `[tool.ruff]`; excludes `third_party/`, `legacy/`, `research/` — only the shipped agent's own code is linted).
- `make rollout` — random-policy rollout over all curated tasks, writes `runs/demo/`.
- `make train` — `train.py --algo ppo --task_id 67a3c6ac --run_id demo` (edit the task_id, or pass `--algo gp`, for a different run).
- `make viz` — builds the frontend and starts the backend at `http://127.0.0.1:8000` (reads `runs/`; override the port with `make viz PORT=8001` if 8000 is taken).
- `make demo` — `train` + `viz` in one command: trains a fresh run, then opens the visualizer on it. If `runs/` already has something in it (`make rollout`/`make train` output, or any prior run), `make viz` alone is faster.
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
