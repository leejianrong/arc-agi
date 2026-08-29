"""Generates extra practice instances per task via the vendored `re-arc`
(`third_party/re-arc/`), per `docs/SLICES.md` V2 build plan step 3 - PPO's
rollouts see more than ARC's native ~3-5 train pairs per task.

`third_party/re-arc/generators.py` does bare top-level imports (`from dsl
import *`, `from utils import *`) against its own vendored `dsl.py`/
`utils.py` - a *different* module than `arc_env/_dsl.py`'s (see
`third_party/re-arc/README.md`'s "two deliberate deviations" section on why
they're kept separate). Both are literally named `dsl`, though, so a naive
`sys.path` + `import generators` shim (V3's first attempt) would silently
resolve `generators.py`'s `from dsl import *` to whichever of the two
`dsl` modules happened to be imported first and cached in `sys.modules`
under that bare name - not necessarily re-arc's own, breaking exactly the
isolation that README section is about. `_import_generators` below loads
re-arc's `dsl`/`utils`/`generators` via `importlib` under private names,
registering them as `sys.modules["dsl"]`/`["utils"]` only for the instant
`generators.py`'s own module-level `from ... import *` statements need to
resolve them, then restores whatever was there before.
"""

import importlib.util
import sys
from pathlib import Path

from arc_env.task_loader import CURATED_TASK_IDS, Pair

_RE_ARC_DIR = Path(__file__).resolve().parent.parent / "third_party" / "re-arc"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_generators():
    saved = {name: sys.modules.get(name) for name in ("dsl", "utils", "generators")}
    try:
        sys.modules["dsl"] = _load_module("_re_arc_dsl", _RE_ARC_DIR / "dsl.py")
        sys.modules["utils"] = _load_module("_re_arc_utils", _RE_ARC_DIR / "utils.py")
        return _load_module("_re_arc_generators", _RE_ARC_DIR / "generators.py")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


generators = _import_generators()

MAX_ATTEMPTS = 10  # a generator can occasionally raise or produce a degenerate pair; retry a few times


class GenerationError(RuntimeError):
    pass


def generate_pair(task_id: str, diff_lb: float = 0.0, diff_ub: float = 1.0) -> Pair:
    """One fresh synthetic (input, output) instance of `task_id`'s concept.

    `diff_lb`/`diff_ub` are re-arc's own difficulty dial (0 = easiest,
    1 = hardest, per-task-defined) - both 0 and 1 are always valid.
    """

    if task_id not in CURATED_TASK_IDS:
        raise ValueError(f"{task_id!r} is not in the curated task subset")

    generator = getattr(generators, f"generate_{task_id}")
    for _ in range(MAX_ATTEMPTS):
        try:
            example = generator(diff_lb, diff_ub)
        except Exception:  # noqa: BLE001, S112 - a vendored re-arc generator can raise
            # anything for an unlucky diff bound; retry rather than propagate.
            continue
        input_grid, output_grid = example["input"], example["output"]
        if input_grid == output_grid:
            continue  # degenerate: nothing for the agent to do
        return Pair(input=input_grid, output=output_grid)

    raise GenerationError(f"re-arc generator for {task_id!r} failed {MAX_ATTEMPTS} times in a row")
