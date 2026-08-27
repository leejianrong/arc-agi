"""The Gymnasium-style ARC environment (ADR-0004).

One episode = one (task_id, grid pair) instance - either one of the task's
native train pairs, or an on-the-fly `arc_env.re_arc`-generated instance of
the same task concept (`docs/SLICES.md` V2 build plan step 3). `reset`
starts from the pair's input grid; the agent edits it one curated action
(`arc_env.actions`) at a time. The episode ends (`terminated=True`) on an
exact match with the pair's output grid, OR (V3, ADR-0002) when the agent
plays a valid `commit` action - which crops the grid to a chosen sub-region
and ends the episode there whether or not that crop happens to match. Use
`info["exact_match"]`, not the `terminated` return value, to tell success
from "the agent gave up and committed something wrong" - both terminate the
episode, only one is a success. `truncated=True` on a max-step budget (Q7),
same as before.

Reward is ADR-0005's dense delta-shaped reward (`arc_env.reward`).

One deviation from ADR-0002's literal wording, worth flagging: there is no
persistent 30x30 canvas held separately from "the grid" - `canvas` (an
action, `arc_env.actions`) replaces the single working grid outright, and
`commit` crops that same working grid in place, rather than the agent
maintaining a fixed 30x30 surface that a smaller committed output is shown
"cut out of" alongside. ADR-0002's consequences describe the visualizer
needing to render "a canvas larger than the committed output, showing the
scratch area, then the cropped final grid" - here that shows up as an
ordinary before/after step (the grid shrinking at the `commit` step),
reusing the replay mechanism V1 already has, not a second persistent
canvas view. Simpler, and every same-shape V1 task still works unmodified
(their grid was never forced into a 30x30 field to begin with).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from arc_env import actions, reward as reward_mod
from arc_env.task_loader import Pair, Task, load_task

PAD_VALUE = 10  # beyond ARC's 10 colors (0-9); marks padding in the fixed-size observation
DEFAULT_MAX_STEPS = 25


def _pad_grid(grid: tuple) -> np.ndarray:
    obs = np.full((actions.MAX_GRID_DIM, actions.MAX_GRID_DIM), PAD_VALUE, dtype=np.int8)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            obs[i, j] = v
    return obs


class ArcEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = DEFAULT_MAX_STEPS):
        super().__init__()
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=0, high=PAD_VALUE, shape=(actions.MAX_GRID_DIM, actions.MAX_GRID_DIM), dtype=np.int8
        )
        self.action_space = spaces.Dict(
            {
                "primitive": spaces.Discrete(len(actions.ACTIONS)),
                **{
                    f"arg{i + 1}": spaces.Discrete(actions.RAW_ARG_RANGE)
                    for i in range(actions.MAX_ARITY)
                },
            }
        )

        self._grid = None
        self._target = None
        self._task_id = None
        self._pair_index = None
        self._step_count = 0
        self._diff_mask = None

    def reset(
        self,
        *,
        task_id: str,
        pair_index: int = 0,
        task: Task = None,
        pair: Pair = None,
        seed=None,
        options=None,
    ):
        """`pair`, if given, overrides `task`/`pair_index` entirely (e.g. a
        `arc_env.re_arc.generate_pair` instance) - `pair_index` is then just
        a label for logging, not a lookup key."""

        super().reset(seed=seed)
        if pair is None:
            task = task if task is not None else load_task(task_id)
            pair = task.train[pair_index]

        self._grid = pair.input
        self._target = pair.output
        self._task_id = task_id
        self._pair_index = pair_index
        self._step_count = 0
        self._diff_mask = reward_mod.compute_diff_mask(pair.input, pair.output)

        return _pad_grid(self._grid), self._info()

    def step(self, action: dict):
        if self._grid is None:
            raise RuntimeError("call reset() before step()")

        primitive_index = int(action["primitive"])
        if 0 <= primitive_index < len(actions.ACTIONS):
            arity = actions.ACTIONS[primitive_index].arity
            action_name = actions.ACTIONS[primitive_index].name
        else:
            arity = 0
            action_name = None
        raw_args = tuple(int(action[f"arg{i + 1}"]) for i in range(arity))

        prev_grid = self._grid
        new_grid, decoded_args, valid = actions.execute(primitive_index, raw_args, self._grid)
        self._grid = new_grid
        self._step_count += 1

        exact_match = self._grid == self._target
        is_commit = valid and action_name == "commit"
        terminated = bool(exact_match or is_commit)
        truncated = self._step_count >= self.max_steps and not terminated

        result = reward_mod.compute_reward(
            prev_grid, self._grid, self._target, self._diff_mask, valid, exact_match
        )

        info = self._info()
        info["action_name"] = action_name
        info["action_args"] = decoded_args
        info["valid_action"] = valid
        info["similarity"] = result.similarity
        info["exact_match"] = exact_match

        return _pad_grid(self._grid), result.reward, terminated, truncated, info

    def get_grid(self) -> tuple:
        """The actual (unpadded) current grid - for episode logging/replay."""
        return self._grid

    def _info(self) -> dict:
        return {
            "task_id": self._task_id,
            "pair_index": self._pair_index,
            "step": self._step_count,
            "grid_shape": (len(self._grid), len(self._grid[0])) if self._grid else (0, 0),
            "target_shape": (len(self._target), len(self._target[0])) if self._target else (0, 0),
        }
