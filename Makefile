.PHONY: install test test-py test-py-slow test-ts rollout train viz build-frontend

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
	uv run python -m viz.backend.server
