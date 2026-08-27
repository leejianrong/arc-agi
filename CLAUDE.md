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
- `arc_env/` — the Gymnasium-style ARC environment: the curated
  `arc-dsl`-primitive action space (`actions.py`), the task loader
  (`task_loader.py`), `env.py`, and the JSONL trajectory/run-meta writers
  (`episode_log.py`), per ADR-0004/ADR-0006.
- `scripts/rollout_random.py` — random-policy rollout script; writes
  `runs/<run_id>/` (gitignored, generated locally).
- `viz/backend/` — read-only local HTTP server exposing `runs/` as JSON
  (`server.py`); also serves `viz/frontend/dist` so one process runs the
  whole visualizer.
- `viz/frontend/` — TypeScript + Canvas replay UI (Vite + Vitest), per
  ADR-0007.
- `legacy/` — the original geometric-transform + color-bijection baseline
  (`baseline.py`, `evaluate.py`, `arc_io.py`). Kept as a reference/sanity-check
  baseline, not part of the new agent.
- `research/arc-ngps/` — a prior, half-built *supervised program-synthesis*
  scaffold. Superseded by `arc-dsl`/`arc_env` per ADR-0001; not deleted, but
  off the path to the shipped agent.
- `docs/` — planning artifacts (PLAN, ADRs, SLICES, QUESTIONS).

## Commands

- `make install` — `uv sync` (Python) + `npm ci` (frontend).
- `make test` (or `uv run pytest` / `cd viz/frontend && npm run typecheck && npm test`) — the fast test layers; no external services needed.
- `make rollout` — random-policy rollout over all curated V1 tasks, writes `runs/demo/`.
- `make viz` — builds the frontend and starts the backend at `http://127.0.0.1:8000` (reads `runs/`).
- `git config core.hooksPath .githooks` — installs the pre-push hook (mirrors CI's cheap jobs); `.github/workflows/ci.yml` is the CI backstop.

## Research subagent policy

For ARC-AGI technical research tasks — DSL/action-space survey, RL or
evolutionary-algorithm literature and prior-art review, or similar deep-dive
research needed to settle an open decision in `docs/QUESTIONS.md` — spin up
**at most 2 subagents at a time**. This is a deliberate cap: keep research
focused and reviewable rather than fanned out into results nobody reads.
