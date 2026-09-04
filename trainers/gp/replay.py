"""Replays a GP-found program through `ArcEnv` step by step, producing the
same step-trace shape `trainers/ppo/rollout.py`'s `evaluate_episode`
does - so `train.py` can log it via the exact same `EpisodeWriter` path
PPO's eval episodes use (ADR-0006), and the visualizer needs no GP-specific
code to replay it (SLICES.md V4's integration-test requirement).

Deliberately goes through `ArcEnv.step` (not `trainers.gp.fitness.
run_program`, which is a fast path for fitness evaluation during search)
so the logged reward/`exact_match` per step come from the exact same
reward/termination logic PPO's episodes do - one canonical definition of
"what happened at this step" shared by both trainers.
"""

from arc_env.env import ArcEnv
from arc_env.task_loader import Pair
from trainers.gp.genome import Program


def program_to_episode_trace(env: ArcEnv, program: Program, task_id: str, pair: Pair) -> dict:
    env.reset(task_id=task_id, pair=pair)
    steps = []
    terminated = truncated = False
    exact_match = False
    total_reward = 0.0

    for primitive_index, raw_args in program:
        if terminated or truncated:
            break
        grid_before = env.get_grid()
        action = {"primitive": primitive_index, **{f"arg{i + 1}": raw_args[i] for i in range(len(raw_args))}}
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        exact_match = info["exact_match"]
        steps.append({
            "grid_before": grid_before,
            "action_name": info["action_name"],
            "action_args": info["action_args"],
            "grid_after": env.get_grid(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "valid_action": info["valid_action"],
            "exact_match": exact_match,
            "selected": info["selected"],
        })

    return {"steps": steps, "success": exact_match, "total_reward": total_reward}
