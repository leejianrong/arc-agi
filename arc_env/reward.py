"""ADR-0005's dense, delta-based, non-background-normalized reward.

    reward_t = (similarity(grid_t, target) - similarity(grid_{t-1}, target))
               - step_cost
               + (terminal_bonus if exact_match else 0)

`similarity` is measured only over cells that actually need to change
between the episode's starting grid and its target (`diff_mask`) - not raw
whole-grid pixel match - so a mostly-unchanged/background grid doesn't score
well for doing nothing (the background-bias pitfall ADR-0005 documents).
`step_cost` discourages no-op/oscillating behavior (the "vibrating in
place" pitfall).

This is V1's env.py's placeholder reward, promoted to the real thing per
`docs/SLICES.md` V2 build plan step 2. One function, so genetic programming
(ADR-0003, V4) can reuse the exact same "how close is this grid to correct"
definition later.
"""

from dataclasses import dataclass

Grid = tuple

STEP_COST = 0.01
INVALID_ACTION_PENALTY = 0.02
TERMINAL_BONUS = 1.0

DiffMask = frozenset  # frozenset[tuple[int, int]]


def compute_diff_mask(input_grid: Grid, target_grid: Grid) -> DiffMask:
    """The (fixed, per-episode) set of cell coordinates where the episode's
    starting grid and its target disagree - the only cells `similarity`
    scores. Assumes `input_grid` and `target_grid` are the same shape (true
    for every V1/V2 curated task)."""

    return frozenset(
        (i, j)
        for i, row in enumerate(input_grid)
        for j, v in enumerate(row)
        if v != target_grid[i][j]
    )


def _grid_shape(grid: Grid) -> tuple:
    return (len(grid), len(grid[0]) if grid else 0)


def similarity(grid: Grid, target_grid: Grid, diff_mask: DiffMask) -> float:
    """Fraction of `diff_mask` cells where `grid` now matches `target_grid`.

    1.0 if `diff_mask` is empty (nothing needed to change - trivially
    already correct). 0.0 if `grid`'s shape doesn't match `target_grid`'s
    (a shape-changing action can never be judged cell-for-cell against it).
    """

    if not diff_mask:
        return 1.0
    if _grid_shape(grid) != _grid_shape(target_grid):
        return 0.0
    matched = sum(1 for (i, j) in diff_mask if grid[i][j] == target_grid[i][j])
    return matched / len(diff_mask)


@dataclass(frozen=True)
class RewardResult:
    reward: float
    prev_similarity: float
    similarity: float


def compute_reward(
    prev_grid: Grid,
    grid: Grid,
    target_grid: Grid,
    diff_mask: DiffMask,
    valid_action: bool,
    terminated: bool,
) -> RewardResult:
    prev_sim = similarity(prev_grid, target_grid, diff_mask)
    cur_sim = similarity(grid, target_grid, diff_mask)
    reward = (cur_sim - prev_sim) - STEP_COST
    if not valid_action:
        reward -= INVALID_ACTION_PENALTY
    if terminated:
        reward += TERMINAL_BONUS
    return RewardResult(reward=reward, prev_similarity=prev_sim, similarity=cur_sim)
