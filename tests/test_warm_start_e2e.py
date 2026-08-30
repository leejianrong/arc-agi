"""End-to-end test for ADR-0009's own suggested test criterion: BC-pretrained
PPO (`train.py --algo ppo --warm_start_from <gp_run_dir>`) reaches nonzero
eval success measurably faster (same rollout/update budget, fewer updates)
than cold-start PPO on the same task - the "does the mechanism actually
help" question the ADR left for a future slice to answer.

Uses `67a3c6ac` (`vmirror`, PLAN.md's own PPO-sanity fixture) rather than
ADR-0009's `d10ecb37` example: `d10ecb37`/`5bd6f4ac` need `commit`'s 4 raw
args to land exactly right simultaneously for the crop to match at all (no
partial-credit gradient toward "almost right" args - see `arc_env.actions.
execute`'s all-or-nothing bounds check), which random GP mutation over a
30-wide raw-arg range does not reliably find within a fast test's budget.
`vmirror` is zero-arg and GP-solvable in a handful of generations (see
`tests/test_gp_evolve.py`), and is enough to demonstrate the warm-start
mechanism end-to-end without depending on GP happening to solve a much
harder search problem."""

import json

import pytest

from train import train_gp, train_ppo
from trainers.gp.evolve import GPConfig
from trainers.ppo.ppo import PPOConfig

pytestmark = pytest.mark.slow  # GP + 2x short PPO runs

TASK_ID = "67a3c6ac"
N_UPDATES = 5
ROLLOUT_STEPS = 64
MAX_STEPS = 10


def _mean_success_rate(run_dir) -> float:
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    rates = [m["success_rate"] for m in metrics if m["success_rate"] is not None]
    return sum(rates) / len(rates) if rates else 0.0


@pytest.fixture(scope="module")
def gp_run_dir(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("warm_start_e2e") / "gp"
    train_gp(
        task_id=TASK_ID, run_dir=run_dir,
        config=GPConfig(population_size=50, n_generations=20, max_program_length=4, seed=0),
        max_steps=25,
    )
    return run_dir


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_warm_started_ppo_beats_cold_start_ppo_on_the_same_seed(gp_run_dir, tmp_path_factory, seed):
    run_dir = tmp_path_factory.mktemp(f"warm_start_e2e_seed{seed}")
    config = PPOConfig(n_epochs=4, minibatch_size=64)

    cold_dir = run_dir / "cold"
    train_ppo(
        task_id=TASK_ID, run_dir=cold_dir, n_updates=N_UPDATES, rollout_steps=ROLLOUT_STEPS,
        eval_every=N_UPDATES - 1, re_arc_prob=0.5, max_steps=MAX_STEPS, seed=seed, config=config,
    )

    warm_dir = run_dir / "warm"
    train_ppo(
        task_id=TASK_ID, run_dir=warm_dir, n_updates=N_UPDATES, rollout_steps=ROLLOUT_STEPS,
        eval_every=N_UPDATES - 1, re_arc_prob=0.5, max_steps=MAX_STEPS, seed=seed, config=config,
        warm_start_from=gp_run_dir, warm_start_epochs=30, warm_start_batch_size=8, warm_start_lr=1e-2,
    )

    assert _mean_success_rate(warm_dir) > _mean_success_rate(cold_dir)
