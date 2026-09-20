"""Tests for `trainers.gp.replay` - a GP-found program replayed through
`ArcEnv` produces the same step-trace shape `evaluate_episode` (PPO) does,
so it logs through the same `EpisodeWriter` path with no special-casing."""

import pytest

from arc_env import actions
from arc_env.env import ORACLE_TERMINATION, ArcEnv
from arc_env.task_loader import Pair, load_task
from trainers.gp.replay import program_to_episode_trace


def test_program_to_episode_trace_matches_a_known_solving_program():
    env = ArcEnv()
    task = load_task("67a3c6ac")  # solved by a single vmirror
    vmirror_idx = actions.ACTION_BY_NAME["vmirror"]
    program = [(vmirror_idx, (0,) * actions.MAX_ARITY)]

    result = program_to_episode_trace(env, program, "67a3c6ac", task.train[0])

    assert result["success"] is True
    assert len(result["steps"]) == 1
    assert result["steps"][0]["action_name"] == "vmirror"
    assert result["steps"][0]["exact_match"] is True
    assert result["steps"][0]["grid_after"] == task.train[0].output


def test_program_to_episode_trace_rejects_a_horizon_before_the_static_endpoint():
    env = ArcEnv(max_steps=2)
    task = load_task("67a3c6ac")
    identity_idx = actions.ACTION_BY_NAME["identity"]
    program = [(identity_idx, (0,) * actions.MAX_ARITY)] * 5  # longer than max_steps

    with pytest.raises(ValueError, match="static endpoint"):
        program_to_episode_trace(env, program, "67a3c6ac", task.train[0])


def test_empty_program_scores_its_static_identity_endpoint():
    grid = ((1, 2), (3, 4))
    pair = Pair(input=grid, output=grid)

    result = program_to_episode_trace(ArcEnv(), [], "fixture", pair)

    assert result["steps"] == []
    assert result["success"] is True


def test_program_trace_does_not_stop_at_an_intermediate_target_match():
    env = ArcEnv(max_steps=2)
    task = load_task("67a3c6ac")
    vmirror_idx = actions.ACTION_BY_NAME["vmirror"]
    program = [
        (vmirror_idx, (0,) * actions.MAX_ARITY),
        (vmirror_idx, (0,) * actions.MAX_ARITY),
    ]

    result = program_to_episode_trace(env, program, "67a3c6ac", task.train[0])

    assert len(result["steps"]) == 2
    assert result["steps"][0]["exact_match"] is True
    assert result["steps"][0]["terminated"] is False
    assert result["steps"][1]["truncated"] is True
    assert result["steps"][1]["exact_match"] is False
    assert result["success"] is False


def test_program_trace_rejects_oracle_termination_mode():
    env = ArcEnv(termination_mode=ORACLE_TERMINATION)
    task = load_task("67a3c6ac")
    vmirror_idx = actions.ACTION_BY_NAME["vmirror"]

    with pytest.raises(ValueError, match="target-independent"):
        program_to_episode_trace(
            env,
            [(vmirror_idx, (0,) * actions.MAX_ARITY)],
            "67a3c6ac",
            task.train[0],
        )
