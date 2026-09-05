"""Tests for `trainers.ppo.warm_start` (ADR-0009): loading a GP run's
best-program trajectory as behavior-cloning demonstrations, and a
supervised pretrain pass against the policy's factored action heads."""

import numpy as np
import torch

from arc_env import actions
from arc_env.env import PAD_VALUE, ArcEnv
from arc_env.episode_log import RunMeta, write_run_meta
from arc_env.task_loader import load_task
from train import _write_episode, check_warm_start_compatible, train_gp
from trainers.gp.evolve import GPConfig
from trainers.gp.replay import program_to_episode_trace
from trainers.ppo.network import ActorCritic
from trainers.ppo.warm_start import load_demonstration, pretrain_from_demonstration
from viz.backend.server import read_episode

TASK_ID = "67a3c6ac"  # solved by a single vmirror - fast for GP, no `slow` marker needed


def _gp_run(tmp_path):
    run_dir = tmp_path / "gp-run"
    config = GPConfig(population_size=50, n_generations=20, max_program_length=4, seed=0)
    train_gp(task_id=TASK_ID, run_dir=run_dir, config=config, max_steps=25)
    return run_dir


def _write_demo_run(tmp_path, task_id: str, program: list) -> object:
    """A hand-picked program written out through the exact same path
    `train_gp` uses (`program_to_episode_trace` + `_write_episode`), rather
    than a real GP search - deterministic and independent of GP's RNG
    outcome for a given action-space size, which is all `load_demonstration`
    itself needs to be tested against."""

    run_dir = tmp_path / "demo-run"
    task = load_task(task_id)
    env = ArcEnv()
    write_run_meta(run_dir, RunMeta(run_id="demo-run", algo="gp", task_ids=[task_id], config={}))
    trace = program_to_episode_trace(env, program, task_id, task.train[0])
    _write_episode(run_dir, "best-program", env, task_id, task.train[0], trace)
    return run_dir


def test_load_demonstration_recovers_the_solving_program(tmp_path):
    vmirror_idx = actions.ACTION_BY_NAME["vmirror"]
    run_dir = _write_demo_run(tmp_path, TASK_ID, [(vmirror_idx, (0,) * actions.MAX_ARITY)])

    demonstration = load_demonstration(run_dir)

    assert len(demonstration) == 1  # a single vmirror solves 67a3c6ac
    step = demonstration[0]
    assert step["primitive"] == vmirror_idx
    assert step["grid"].shape == (2, actions.MAX_GRID_DIM, actions.MAX_GRID_DIM)
    assert step["grid"].dtype == np.int8
    # vmirror is zero-arg - the raw arg slots are present but unconstrained/masked.
    assert set(step) == {"grid", "primitive", "arg1", "arg2", "arg3", "arg4"}


def test_load_demonstration_recovers_a_select_then_act_on_selection_program(tmp_path):
    """ADR-0011: `1f85a75f`'s solver is `select_largest` + `commit_selection`
    - a `"select"` action (no grid change, updates state) followed by an
    `"act_on_selection"` action, the two new `Action.kind`s this ADR adds."""

    select_idx = actions.ACTION_BY_NAME["select_largest"]
    commit_idx = actions.ACTION_BY_NAME["commit_selection"]
    program = [(select_idx, (0,) * actions.MAX_ARITY), (commit_idx, (0,) * actions.MAX_ARITY)]
    run_dir = _write_demo_run(tmp_path, "1f85a75f", program)

    demonstration = load_demonstration(run_dir)

    assert len(demonstration) == 2
    assert demonstration[0]["primitive"] == select_idx
    assert demonstration[1]["primitive"] == commit_idx

    # The bug this test also guards against: the observation the policy sees
    # when predicting a step's action must carry the selection state going
    # into that decision, not the state that step's own action produces.
    # Step 0 (select_largest) runs with nothing selected yet - its input
    # selection channel must be all zero.
    assert demonstration[0]["grid"][1].sum() == 0
    # Step 1 (commit_selection) runs with whatever select_largest just
    # selected - its input selection channel must be non-empty and must
    # match the env's own post-select-step selection state exactly.
    episode = read_episode(run_dir.parent, run_dir.name, "best-program")
    selected_after_select = episode["steps"][0]["selected"]
    assert selected_after_select  # sanity: select_largest actually selected something
    expected_mask = np.zeros((actions.MAX_GRID_DIM, actions.MAX_GRID_DIM), dtype=np.int8)
    for i, j in selected_after_select:
        expected_mask[i, j] = 1
    assert np.array_equal(demonstration[1]["grid"][1], expected_mask)


def test_load_demonstration_transform_only_program_has_no_selection(tmp_path):
    """Regression check: a plain transform-only demonstration (no
    select_*/act_on_selection actions) must be unaffected by the
    previous-step-selection threading - the selection channel stays all
    zero throughout, as before."""

    # Two hmirrors, not one vmirror: 67a3c6ac is solved by a single vmirror
    # (see TASK_ID's comment above), which would terminate the episode after
    # step 1 and leave no second step for this test to exercise. hmirror
    # applied twice returns to the original grid, which doesn't match the
    # target, so the episode runs both steps without an early exact-match
    # termination.
    hmirror_idx = actions.ACTION_BY_NAME["hmirror"]
    program = [(hmirror_idx, (0,) * actions.MAX_ARITY), (hmirror_idx, (0,) * actions.MAX_ARITY)]
    run_dir = _write_demo_run(tmp_path, TASK_ID, program)

    demonstration = load_demonstration(run_dir)

    assert len(demonstration) == 2
    for step in demonstration:
        assert step["grid"][1].sum() == 0


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
        ("move_selected", {"direction": 2}),  # ADR-0012's "direction" kind
    ]
    for action_name, decoded_args in cases:
        primitive_index, raw_args = _encode_action(action_name, decoded_args)
        action = actions.ACTIONS[primitive_index]
        for i, spec in enumerate(action.args):
            assert spec.decode(raw_args[i]) == decoded_args[spec.name]


def test_pretrain_from_demonstration_drives_the_policy_toward_the_demonstrated_action():
    network = ActorCritic()
    grid = np.zeros((2, actions.MAX_GRID_DIM, actions.MAX_GRID_DIM), dtype=np.int8)
    grid[0, 5:8, 5:8] = 3
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
