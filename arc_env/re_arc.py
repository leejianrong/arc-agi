"""Generates extra practice instances per task via the vendored `re-arc`
(`third_party/re-arc/`), per `docs/SLICES.md` V2 build plan step 3 - PPO's
rollouts see more than ARC's native ~3-5 train pairs per task.

Import shim mirrors `arc_env/_dsl.py`: `third_party/re-arc/generators.py`
does bare top-level imports (`from dsl import *`, `from utils import *`)
against its own vendored `dsl.py`/`utils.py`, so it needs that directory on
`sys.path`.
"""

import sys
from pathlib import Path

from arc_env.task_loader import CURATED_TASK_IDS, Pair

_RE_ARC_DIR = Path(__file__).resolve().parent.parent / "third_party" / "re-arc"
if str(_RE_ARC_DIR) not in sys.path:
    sys.path.insert(0, str(_RE_ARC_DIR))

import generators  # noqa: E402

MAX_ATTEMPTS = 10  # a generator can occasionally raise or produce a degenerate pair; retry a few times


class GenerationError(RuntimeError):
    pass


def generate_pair(task_id: str, diff_lb: float = 0.0, diff_ub: float = 1.0) -> Pair:
    """One fresh synthetic (input, output) instance of `task_id`'s concept.

    `diff_lb`/`diff_ub` are re-arc's own difficulty dial (0 = easiest,
    1 = hardest, per-task-defined) - both 0 and 1 are always valid.
    """

    if task_id not in CURATED_TASK_IDS:
        raise ValueError(f"{task_id!r} is not in the V1/V2 curated task subset")

    generator = getattr(generators, f"generate_{task_id}")
    for _ in range(MAX_ATTEMPTS):
        try:
            example = generator(diff_lb, diff_ub)
        except Exception:
            continue
        input_grid, output_grid = example["input"], example["output"]
        if input_grid == output_grid:
            continue  # degenerate: nothing for the agent to do
        return Pair(input=input_grid, output=output_grid)

    raise GenerationError(f"re-arc generator for {task_id!r} failed {MAX_ATTEMPTS} times in a row")
