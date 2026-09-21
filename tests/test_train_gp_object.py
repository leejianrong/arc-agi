"""Integration test (SLICES.md V6): `train.py --algo gp_object`'s output -
GP-over-the-object-grammar run metrics and, per ADR-0014, one execution
trace per snapshotted generation plus the best-found genome's own - loads
through `viz/backend/server.py`'s existing read functions with no
object-track-specific code (the same ADR-0006 "one `runs/<run_id>/` shape"
guarantee `tests/test_train_gp.py` already proves for the flat-space
trainer)."""

from arc_env.tasks import load_task
from train import train_gp_object
from trainers.gp_object.evolve import GPConfig
from viz.backend import server as backend

TASK_ID = "b1948b0a"  # solved by a single replace_color(6, 2) - fast for GP


def test_train_gp_object_end_to_end_produces_a_backend_readable_run(tmp_path):
    run_dir = tmp_path / "gp-object-run"
    config = GPConfig(population_size=30, n_generations=15, max_program_length=3, seed=2)

    train_gp_object(task_id=TASK_ID, run_dir=run_dir, config=config, max_steps=25)

    runs_dir = run_dir.parent
    runs = backend.list_runs(runs_dir)
    assert runs == [
        {"run_id": "gp-object-run", "algo": "gp_object", "created_at": runs[0]["created_at"], "task_ids": [TASK_ID]}
    ]

    metrics = backend.read_metrics(runs_dir, "gp-object-run")
    assert metrics, "GP should log at least one generation"
    expected_keys = {"update", "timestamp", "n_episodes", "mean_reward", "success_rate"}
    for row in metrics:
        assert expected_keys <= row.keys()
    updates = [row["update"] for row in metrics]
    assert updates == sorted(updates)  # same monotonic-curve requirement as flat GP's/PPO's metrics

    episode_ids = backend.list_episode_ids(runs_dir, "gp-object-run")
    assert len(episode_ids) > 1
    assert episode_ids[-1] == "best-program"
    assert episode_ids[0] != "best-program"

    episode = backend.read_episode(runs_dir, "gp-object-run", "best-program")
    assert episode["start"]["task_id"] == TASK_ID
    assert episode["end"]["success"] is True
    assert episode["steps"][-1]["exact_match"] is True

    # The best genome genuinely solves the task's held-out test pair too,
    # not just what fitness evaluation (train-pairs only) checked.
    task = load_task(TASK_ID)
    assert episode["start"]["target_grid"] == [list(r) for r in task.train[0].output]
