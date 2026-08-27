"""Unit/contract tests for `arc_env.env.ArcEnv` (reset/step, Q7 invalid-action
no-op + penalty, episode termination) and the curated task loader."""

import numpy as np
import pytest

from arc_env import actions
from arc_env.env import ArcEnv, PAD_VALUE
from arc_env.task_loader import CURATED_TASK_IDS, load_task, load_curated_tasks


def action(primitive_name: str, *raw_args: int) -> dict:
    idx = actions.ACTION_BY_NAME[primitive_name]
    a = {"primitive": idx}
    for i in range(actions.MAX_ARITY):
        a[f"arg{i + 1}"] = raw_args[i] if i < len(raw_args) else 0
    return a


def test_reset_returns_padded_observation_matching_actual_grid():
    env = ArcEnv()
    obs, info = env.reset(task_id="67a3c6ac", pair_index=0)
    grid = env.get_grid()
    h, w = len(grid), len(grid[0])
    assert obs.shape == (30, 30)
    assert np.array_equal(obs[:h, :w], np.array(grid))
    assert (obs[h:, :] == PAD_VALUE).all()
    assert info["task_id"] == "67a3c6ac"


def test_valid_action_applies_the_dsl_primitive_exactly():
    from arc_env._dsl import dsl

    env = ArcEnv()
    env.reset(task_id="67a3c6ac", pair_index=0)
    grid_before = env.get_grid()
    _, reward, terminated, truncated, info = env.step(action("rot180"))
    assert info["valid_action"] is True
    assert env.get_grid() == dsl.rot180(grid_before)
    assert isinstance(reward, float)


def test_out_of_range_primitive_index_is_noop_with_penalty():
    env = ArcEnv()
    env.reset(task_id="67a3c6ac", pair_index=0)
    grid_before = env.get_grid()
    a = {"primitive": len(actions.ACTIONS)}
    for i in range(actions.MAX_ARITY):
        a[f"arg{i + 1}"] = 0
    _, reward, terminated, truncated, info = env.step(a)
    assert info["valid_action"] is False
    assert env.get_grid() == grid_before
    assert reward < 0


def test_out_of_bounds_fill_cell_coordinate_is_noop_with_penalty():
    env = ArcEnv()
    env.reset(task_id="67a3c6ac", pair_index=0)
    grid_before = env.get_grid()
    h, w = len(grid_before), len(grid_before[0])
    _, reward, terminated, truncated, info = env.step(action("fill_cell", 5, h + 10, 0))
    assert info["valid_action"] is False
    assert env.get_grid() == grid_before


def test_episode_terminates_on_exact_match():
    env = ArcEnv()
    task = load_task("67a3c6ac")  # solved by a single vmirror
    env.reset(task_id="67a3c6ac", pair_index=0, task=task)
    _, reward, terminated, truncated, info = env.step(action("vmirror"))
    assert terminated is True
    assert truncated is False
    # ADR-0005 dense reward: full similarity delta (0 -> 1) - step_cost + terminal_bonus.
    from arc_env.reward import STEP_COST, TERMINAL_BONUS

    assert reward == pytest.approx(1.0 - STEP_COST + TERMINAL_BONUS)
    assert env.get_grid() == task.train[0].output


def test_episode_truncates_at_max_steps_without_match():
    env = ArcEnv(max_steps=2)
    env.reset(task_id="67a3c6ac", pair_index=0)
    env.step(action("rot90"))
    _, reward, terminated, truncated, info = env.step(action("rot90"))
    assert truncated is True or terminated is True  # rot90 x2 could coincidentally match


def test_curated_tasks_are_all_loadable_and_same_shape():
    tasks = load_curated_tasks()
    assert set(tasks) == set(CURATED_TASK_IDS)
    for task in tasks.values():
        for pair in (*task.train, *task.test):
            assert (len(pair.input), len(pair.input[0])) == (len(pair.output), len(pair.output[0]))
