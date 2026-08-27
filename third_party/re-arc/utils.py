from random import choice, randint, sample, shuffle, uniform

from dsl import *


global rng
rng = []


def unifint(
    diff_lb: float,
    diff_ub: float,
    bounds: Tuple[int, int]
) -> int:
    """
    diff_lb: lower bound for difficulty, must be in range [0, diff_ub]
    diff_ub: upper bound for difficulty, must be in range [diff_lb, 1]
    bounds: interval [a, b] determining the integer values that can be sampled
    """
    a, b = bounds
    d = uniform(diff_lb, diff_ub)
    global rng
    rng.append(d)
    return min(max(a, round(a + (b - a) * d)), b)


def is_grid(
    grid: Any
) -> bool:
    """
    returns True if and only if argument is a valid grid
    """
    if not isinstance(grid, tuple):
        return False
    if not 0 < len(grid) <= 30:
        return False
    if not all(isinstance(r, tuple) for r in grid):
        return False
    if not all(0 < len(r) <= 30 for r in grid):
        return False
    if not len(set(len(r) for r in grid)) == 1:
        return False
    if not all(all(isinstance(x, int) for x in r) for r in grid):
        return False
    if not all(all(0 <= x <= 9 for x in r) for r in grid):
        return False
    return True


def strip_prefix(
    string: str,
    prefix: str
) -> str:
    """
    removes prefix
    """
    return string[len(prefix):]


def format_grid(
    grid: List[List[int]]
) -> Grid:
    """
    grid type casting
    """
    return tuple(tuple(row) for row in grid)


def format_example(
    example: dict
) -> dict:
    """
    example data type
    """
    return {
        'input': format_grid(example['input']),
        'output': format_grid(example['output'])
    }


def format_task(
    task: dict
) -> dict:
    """
    task data type
    """
    return {
        'train': [format_example(example) for example in task['train']],
        'test': [format_example(example) for example in task['test']]
    }
