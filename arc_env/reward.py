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

V3 note: `diff_mask` only makes sense when the episode's starting grid and
its target are the same shape (there's no natural cell-to-cell
correspondence otherwise). For a variable-shape pair, `compute_diff_mask`
returns `None` and `similarity` matches against every target cell once the
current grid reaches the target's shape - a coarser signal (no credit for
"getting the shape right" itself), but still dense once shape is achieved,
and exact everywhere else.

Shape gradient (2026-08-29 fix, see ADR-0005's amendment): before that
shape is achieved, `similarity` used to return a flat 0.0 regardless of how
close the current grid's shape was to the target's - a dead zone with no
gradient at all until an agent (RL) or a GP individual (fitness.py) stumbled
onto the exact right dimensions by chance, which is disproportionately hard
for a wrong-shape-producing action like `commit` (ADR-0002) with its
4-argument (row, col, height, width) combination. `similarity` now blends in
a `SHAPE_MATCH_CREDIT`-weighted score for how close the shape is (Manhattan
distance between the two shapes, normalized by the largest distance two
ARC-legal shapes can have), continuous with the content-match score exactly
at the shape boundary. This only changes behavior when `diff_mask` is
`None`; the same-shape (`diff_mask` given) path is untouched; a shape
mismatch there is always an unintended/wrong action for an already-same-shape
task, not the scenario this fixes.
"""

from dataclasses import dataclass

from arc_env.actions import MAX_GRID_DIM

Grid = tuple

STEP_COST = 0.01
INVALID_ACTION_PENALTY = 0.02
TERMINAL_BONUS = 1.0

# Similarity awarded, before any content match, purely for a variable-shape
# episode's grid landing on the target's exact shape - the gradient that
# replaces the old "0.0 until shape matches" dead zone (see module docstring).
SHAPE_MATCH_CREDIT = 0.3

# Largest possible Manhattan distance between two ARC-legal grid shapes
# (1x1 .. MAX_GRID_DIM x MAX_GRID_DIM), used to normalize shape distance to [0, 1].
_MAX_SHAPE_DISTANCE = 2 * (MAX_GRID_DIM - 1)

DiffMask = frozenset  # frozenset[tuple[int, int]]


def compute_diff_mask(input_grid: Grid, target_grid: Grid) -> DiffMask:
    """The (fixed, per-episode) set of cell coordinates where the episode's
    starting grid and its target disagree - the only cells `similarity`
    scores. `None` if `input_grid`/`target_grid` are different shapes (a
    variable-shape pair - see module docstring); `similarity` handles that
    case separately."""

    if _grid_shape(input_grid) != _grid_shape(target_grid):
        return None
    return frozenset(
        (i, j)
        for i, row in enumerate(input_grid)
        for j, v in enumerate(row)
        if v != target_grid[i][j]
    )


def _grid_shape(grid: Grid) -> tuple:
    return (len(grid), len(grid[0]) if grid else 0)


def _shape_similarity(shape: tuple, target_shape: tuple) -> float:
    """1.0 when the two shapes are identical, decaying linearly with their
    normalized Manhattan distance otherwise - always in [0, 1]."""

    distance = abs(shape[0] - target_shape[0]) + abs(shape[1] - target_shape[1])
    return max(0.0, 1.0 - distance / _MAX_SHAPE_DISTANCE)


def similarity(grid: Grid, target_grid: Grid, diff_mask: DiffMask) -> float:
    """Fraction of `diff_mask` cells where `grid` now matches `target_grid`.

    1.0 if `diff_mask` is empty (nothing needed to change - trivially
    already correct). 0.0 if `grid`'s shape doesn't match `target_grid`'s
    (a shape-changing action can never be judged cell-for-cell against it).

    If `diff_mask` is `None` (a variable-shape pair - `compute_diff_mask`),
    matches against every target cell once `grid` reaches the target's
    shape; before that, returns `SHAPE_MATCH_CREDIT` scaled by how close
    `grid`'s shape is to `target_grid`'s (module docstring's shape
    gradient) rather than a flat 0.0.
    """

    if diff_mask is None:
        grid_shape = _grid_shape(grid)
        target_shape = _grid_shape(target_grid)
        if grid_shape != target_shape:
            return SHAPE_MATCH_CREDIT * _shape_similarity(grid_shape, target_shape)
        total = sum(len(row) for row in target_grid)
        if total == 0:
            return 1.0
        matched = sum(1 for i, row in enumerate(grid) for j, v in enumerate(row) if v == target_grid[i][j])
        content_sim = matched / total
        return SHAPE_MATCH_CREDIT + (1 - SHAPE_MATCH_CREDIT) * content_sim

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
    exact_match: bool,
) -> RewardResult:
    """`exact_match` (not the env's broader `terminated`, which V3's `commit`
    action can also trigger without a match) is what earns `TERMINAL_BONUS` -
    a voluntary but wrong `commit` ends the episode without it."""

    prev_sim = similarity(prev_grid, target_grid, diff_mask)
    cur_sim = similarity(grid, target_grid, diff_mask)
    reward = (cur_sim - prev_sim) - STEP_COST
    if not valid_action:
        reward -= INVALID_ACTION_PENALTY
    if exact_match:
        reward += TERMINAL_BONUS
    return RewardResult(reward=reward, prev_similarity=prev_sim, similarity=cur_sim)
