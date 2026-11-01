.PHONY: help arc-watch install test test-py test-py-slow test-ts rollout train viz build-frontend demo prune-runs print-task arc2-preflight

.DEFAULT_GOAL := help

PORT ?= 8000

help:
	@echo "Available targets:"
	@echo "  install         - uv sync (Python) + npm ci (frontend)"
	@echo "  test            - fast layer: test-py + test-ts (~4s)"
	@echo "  test-py         - uv run pytest -m 'not slow'"
	@echo "  test-py-slow    - uv run pytest (includes ~90s PPO e2e test)"
	@echo "  test-ts         - frontend typecheck + vitest"
	@echo "  build-frontend  - build the viz frontend"
	@echo "  rollout         - random-policy rollout over all curated tasks -> runs/demo"
	@echo "  train           - train.py --algo ppo --task_id 67a3c6ac --run_id demo"
	@echo "  viz             - build frontend + start the visualizer backend (http://127.0.0.1:$(PORT))"
	@echo "                    override the port with: make viz PORT=8001"
	@echo "  demo            - train + viz: one command to produce a run and open the visualizer on it"
	@echo "  prune-runs      - dry-run report of stale runs/ dirs (add YES=1 to actually delete)"
	@echo "  print-task      - print an ARC task's grids to the terminal: make print-task TASK=928ad970"
	@echo "  arc-watch       - open/bundle agent traces in the viewer: make arc-watch ARGS=\"open path/to/out\""
	@echo "  arc2-preflight  - go/no-go gate before pushing a Kaggle notebook (FAST=1 for the quick path)"

install:
	uv sync --group dev
	cd viz/frontend && npm ci

# Fast layer only (no PPO training runs) - what the pre-push hook and CI's
# quick job run. `make test-py-slow` / CI's full job also run the ~90s PPO
# smoke test (tests/test_train_ppo.py).
test: test-py test-ts

test-py:
	uv run pytest -m "not slow"

test-py-slow:
	uv run pytest

test-ts:
	cd viz/frontend && npm run typecheck && npm test

build-frontend:
	cd viz/frontend && npm run build

rollout:
	uv run python scripts/rollout_random.py --all --run_id demo

train:
	uv run python train.py --algo ppo --task_id 67a3c6ac --run_id demo

viz: build-frontend
	uv run python -m viz.backend.server --port $(PORT)

demo: train viz

# runs/ is gitignored and nothing else ever cleans it up - defaults to a dry
# run (scripts/prune_runs.py); `make prune-runs YES=1` actually deletes.
prune-runs:
	uv run python scripts/prune_runs.py $(if $(YES),--yes,)

print-task:
	uv run python scripts/print_task.py $(TASK)

arc2-preflight:
	uv run python scripts/arc2_preflight.py $(if $(FAST),--fast)

arc-watch:
	uv run python scripts/arc_watch.py $(ARGS)
