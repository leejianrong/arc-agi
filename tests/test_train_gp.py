"""Integration test (SLICES.md V4): `train.py --algo gp`'s output - GP run
metrics and, per ADR-0014, one execution trace per snapshotted generation
plus the best-found program's own - loads through `viz/backend/server.py`'s
existing read functions with no GP-specific code (ADR-0006: same
`runs/<run_id>/` shape V1-V3 already produce)."""


from arc_env.task_loader import load_task
from train import train_gp
from trainers.gp.evolve import GPConfig
from viz.backend import server as backend

TASK_ID = "67a3c6ac"  # solved by a single vmirror - fast for GP, no `slow` marker needed


def test_train_gp_end_to_end_produces_a_backend_readable_run(tmp_path):
    run_dir = tmp_path / "gp-run"
    config = GPConfig(population_size=50, n_generations=20, max_program_length=4, seed=0)

    train_gp(task_id=TASK_ID, run_dir=run_dir, config=config, max_steps=25)

    runs_dir = run_dir.parent
    runs = backend.list_runs(runs_dir)
    assert runs == [{"run_id": "gp-run", "algo": "gp", "created_at": runs[0]["created_at"], "task_ids": [TASK_ID]}]

    metrics = backend.read_metrics(runs_dir, "gp-run")
    assert metrics, "GP should log at least one generation"
    expected_keys = {"update", "timestamp", "n_episodes", "mean_reward", "success_rate"}
    for row in metrics:
        assert expected_keys <= row.keys()
    updates = [row["update"] for row in metrics]
    assert updates == sorted(updates)  # same monotonic-curve requirement as PPO's metrics

    # ADR-0014: at least one generation snapshot alongside "best-program" -
    # "best-program" sorts last (digit-prefixed snapshot names < the letter
    # "b"), which is what lets main.ts's existing earliest/latest-episode
    # panel defaults land on "earliest generation" vs. "the final best" here,
    # the same way they already land on "earliest eval"/"latest eval" for PPO.
    episode_ids = backend.list_episode_ids(runs_dir, "gp-run")
    assert len(episode_ids) > 1
    assert episode_ids[-1] == "best-program"
    assert episode_ids[0] != "best-program"

    episode = backend.read_episode(runs_dir, "gp-run", "best-program")
    assert episode["start"]["task_id"] == TASK_ID
    assert episode["end"]["success"] is True  # GP found a perfect vmirror program
    assert episode["steps"][-1]["exact_match"] is True

    # The best program genuinely solves the task's held-out test pair too,
    # not just what fitness evaluation (train-pairs only) checked.
    task = load_task(TASK_ID)
    assert episode["start"]["target_grid"] == [list(r) for r in task.train[0].output]
