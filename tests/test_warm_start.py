"""Tests for `trainers.ppo.warm_start` (ADR-0009): loading a GP run's
best-program trajectory as behavior-cloning demonstrations, and a
supervised pretrain pass against the policy's factored action heads."""

import numpy as np
import torch

from arc_env import actions
from arc_env.env import PAD_VALUE
from train import check_warm_start_compatible, train_gp
from trainers.gp.evolve import GPConfig
from trainers.ppo.network import ActorCritic
from trainers.ppo.warm_start import load_demonstration, pretrain_from_demonstration

TASK_ID = "67a3c6ac"  # solved by a single vmirror - fast for GP, no `slow` marker needed


def _gp_run(tmp_path):
    run_dir = tmp_path / "gp-run"
    config = GPConfig(population_size=50, n_generations=20, max_program_length=4, seed=0)
    train_gp(task_id=TASK_ID, run_dir=run_dir, config=config, max_steps=25)
    return run_dir


def test_load_demonstration_recovers_the_solving_program(tmp_path):
    run_dir = _gp_run(tmp_path)

    demonstration = load_demonstration(run_dir)

    assert len(demonstration) == 1  # a single vmirror solves 67a3c6ac
    step = demonstration[0]
    assert step["primitive"] == actions.ACTION_BY_NAME["vmirror"]
    assert step["grid"].shape == (actions.MAX_GRID_DIM, actions.MAX_GRID_DIM)
    assert step["grid"].dtype == np.int8
    # vmirror is zero-arg - the raw arg slots are present but unconstrained/masked.
    assert set(step) == {"grid", "primitive", "arg1", "arg2", "arg3", "arg4"}


def test_load_demonstration_pads_the_grid_like_the_env_observation():
    from arc_env.env import ArcEnv
    from arc_env.task_loader import load_task

    task = load_task(TASK_ID)
    env = ArcEnv()
    expected_obs, _ = env.reset(task_id=TASK_ID, pair=task.train[0])

    from trainers.ppo.warm_start import _pad_grid
    got = _pad_grid([list(row) for row in task.train[0].input])

    assert np.array_equal(got, expected_obs)
    assert (got == PAD_VALUE).sum() > 0  # the fixture grid is smaller than the 30x30 canvas


def test_encode_action_round_trips_through_decode_for_every_curated_arg_kind():
    """The raw value recovered from a logged decoded value must decode back
    to that exact value, for every arg `kind` in the curated action space -
    otherwise a demonstration would train the network toward a different
    action than what was actually logged."""

    from trainers.ppo.warm_start import _encode_action

    cases = [
        ("fill_cell", {"color": 7, "row": 3, "col": 4}),
        ("canvas", {"value": 0, "height": 5, "width": 5}),
        ("commit", {"row": 0, "col": 0, "height": 2, "width": 2}),
        ("hupscale", {"factor": 3}),
        ("replace", {"replacee": 1, "replacer": 2}),
    ]
    for action_name, decoded_args in cases:
        primitive_index, raw_args = _encode_action(action_name, decoded_args)
        action = actions.ACTIONS[primitive_index]
        for i, spec in enumerate(action.args):
            assert spec.decode(raw_args[i]) == decoded_args[spec.name]


def test_pretrain_from_demonstration_drives_the_policy_toward_the_demonstrated_action():
    network = ActorCritic()
    grid = np.zeros((actions.MAX_GRID_DIM, actions.MAX_GRID_DIM), dtype=np.int8)
    grid[5:8, 5:8] = 3
    vmirror_idx = actions.ACTION_BY_NAME["vmirror"]
    demonstration = [{
        "grid": grid, "primitive": vmirror_idx,
        "arg1": 0, "arg2": 0, "arg3": 0, "arg4": 0,
    }] * 8  # repeat one demonstrated step so a small batch has something to learn from

    losses = pretrain_from_demonstration(network, demonstration, n_epochs=30, batch_size=8, lr=1e-2)

    assert len(losses) == 30
    assert losses[-1] < losses[0]

    obs = torch.as_tensor(grid, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        primitive, _ = network.get_greedy_action(obs)
    assert int(primitive.item()) == vmirror_idx


def test_pretrain_from_demonstration_returns_empty_losses_for_no_demonstration():
    network = ActorCritic()
    assert pretrain_from_demonstration(network, [], n_epochs=10, batch_size=8, lr=1e-3) == []


def test_check_warm_start_compatible_rejects_a_non_gp_run(tmp_path):
    from arc_env.episode_log import RunMeta, write_run_meta

    run_dir = tmp_path / "ppo-run"
    write_run_meta(run_dir, RunMeta(run_id="ppo-run", algo="ppo", task_ids=[TASK_ID], config={}))

    error = check_warm_start_compatible(TASK_ID, run_dir)
    assert error is not None
    assert "not a GP run" in error


def test_check_warm_start_compatible_rejects_a_different_task(tmp_path):
    run_dir = _gp_run(tmp_path)

    error = check_warm_start_compatible("some_other_task", run_dir)
    assert error is not None
    assert "same-task only" in error


def test_check_warm_start_compatible_accepts_a_matching_gp_run(tmp_path):
    run_dir = _gp_run(tmp_path)
    assert check_warm_start_compatible(TASK_ID, run_dir) is None
