"""Replays a GP-found object-grammar genome step by step, producing the same
step-trace shape `trainers/gp/replay.py`'s `program_to_episode_trace` and
`trainers/ppo/rollout.py`'s `evaluate_episode` do - so `train.py` can log it
via the exact same `EpisodeWriter` path (ADR-0006), and the visualizer needs
no object-track-specific code to replay it (SLICES.md V6's own
integration-test requirement).

`object_env` has no `ArcEnv`-equivalent stepper, so this steps by hand
through `Program.steps` (`step.action.fn(state, **step.args)` in turn,
exactly what `Program.run` itself does) and computes reward/exact_match per
step via `arc_env.reward.compute_reward` directly - the same reward
definition PPO/flat-GP episodes use, just not routed through a Gym env
since there isn't one here.

`"selected"` is always `None`: there's no dual-slot selection-overlay data
for the object track yet (`ObjState`'s slots are a different shape from
`arc_env.env.ArcEnv.get_selected()`'s `{"a": ..., "b": ...}`) - carrying the
object set per step is V8's job (SLICES.md), not V6's.

No target-oracle stop (KAN-1546, `trainers/gp/fitness.py`'s `run_program`
docstring): the object grammar has no `commit`-equivalent action that can
voluntarily end a program early, so every step's `terminated` is always
`False`, and this never breaks out of the loop on an intermediate
`exact_match` - a program always runs to its full, static length, exactly
like `object_env.grammar.Program.run` itself (and like `to_program(genome).
run(...)`, what `fitness.evaluate_fitness` scores) already does. Only the
final step is marked `truncated` (the program ran out of genes - the
closest analogue to a flat-space episode exhausting its step budget, since
object-track genomes have no shorter-than-budget "finished early" case at
all).
"""

from arc_env import reward as reward_mod
from arc_env.tasks import Pair
from object_env.state import ObjState
from trainers.gp_object.genome import Genome, to_program


def genome_to_episode_trace(genome: Genome, pair: Pair) -> dict:
    program = to_program(genome)
    diff_mask = reward_mod.compute_diff_mask(pair.input, pair.output)

    state = ObjState(grid=pair.input)
    steps = []
    total_reward = 0.0

    for step in program.steps:
        grid_before = state.grid
        new_state = step.action.fn(state, **step.args)
        valid_action = new_state is not None
        if valid_action:
            state = new_state
        grid_after = state.grid
        exact_match = grid_after == pair.output
        result = reward_mod.compute_reward(
            grid_before, grid_after, pair.output, diff_mask, valid_action, exact_match,
        )
        total_reward += result.reward
        steps.append({
            "grid_before": grid_before,
            "action_name": step.action.name,
            "action_args": step.args,
            "grid_after": grid_after,
            "reward": result.reward,
            "terminated": False,
            "truncated": False,
            "valid_action": valid_action,
            "exact_match": exact_match,
            "selected": None,
        })

    if steps:
        steps[-1]["truncated"] = True

    success = state.grid == pair.output
    return {"steps": steps, "success": success, "total_reward": total_reward}
