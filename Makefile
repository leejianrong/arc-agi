.PHONY: install test test-py test-ts rollout viz build-frontend

install:
	uv sync --group dev
	cd viz/frontend && npm ci

test: test-py test-ts

test-py:
	uv run pytest

test-ts:
	cd viz/frontend && npm run typecheck && npm test

build-frontend:
	cd viz/frontend && npm run build

rollout:
	uv run python scripts/rollout_random.py --all --run_id demo

viz: build-frontend
	uv run python -m viz.backend.server
