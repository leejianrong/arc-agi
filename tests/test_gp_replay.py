"""Tests for `trainers.gp.replay` - a GP-found program replayed through
`ArcEnv` produces the same step-trace shape `evaluate_episode` (PPO) does,
so it logs through the same `EpisodeWriter` path with no special-casing."""

from arc_env import actions
from arc_env.env import ArcEnv
from arc_env.task_loader import load_task
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


def test_program_to_episode_trace_stops_at_max_steps():
    env = ArcEnv(max_steps=2)
    task = load_task("67a3c6ac")
    identity_idx = actions.ACTION_BY_NAME["identity"]
    program = [(identity_idx, (0,) * actions.MAX_ARITY)] * 5  # longer than max_steps

    result = program_to_episode_trace(env, program, "67a3c6ac", task.train[0])

    assert len(result["steps"]) == 2
    assert result["success"] is False
