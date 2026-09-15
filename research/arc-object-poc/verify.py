"""The reusable re-arc generalization verifier — the concrete "N fresh
instances -> pass rate" check ADRs 0023-0026 did with throwaway scripts.

`verify(program, task_id)` runs an object-space program against (a) the task's
real train/test pairs and (b) N freshly-generated `re-arc` instances, and
reports pass rates. Reuses `arc_env.re_arc.generate_pair` (same tuple-of-tuples
Grid type, no conversion) and `arc_env.task_loader.load_task`.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from arc_env.re_arc import GenerationError, generate_pair  # noqa: E402
from arc_env.task_loader import load_task  # noqa: E402

from object_actions import ObjState, execute  # noqa: E402


def run_program(program, grid) -> tuple:
    """Run an object-space program from a starting grid; return the final grid."""
    state = ObjState(grid=grid)
    for index, raw_args in program:
        state, _decoded, _valid = execute(index, raw_args, state)
    return state.grid


@dataclass(frozen=True)
class VerifyResult:
    task_id: str
    real_pass: int
    real_total: int
    rearc_pass: int
    rearc_total: int
    errors: int

    @property
    def real_ok(self) -> bool:
        return self.real_total > 0 and self.real_pass == self.real_total

    @property
    def rearc_rate(self) -> float:
        return self.rearc_pass / self.rearc_total if self.rearc_total else 0.0

    def line(self) -> str:
        flag = "PASS" if (self.real_ok and self.rearc_pass == self.rearc_total) else "----"
        return (
            f"{flag}  {self.task_id}  real {self.real_pass}/{self.real_total}  "
            f"re-arc {self.rearc_pass}/{self.rearc_total}"
            + (f"  ({self.errors} gen-errors)" if self.errors else "")
        )


def _run_ok(program, pair) -> bool:
    try:
        return run_program(program, pair.input) == pair.output
    except Exception:
        return False


def verify(program, task_id: str, n: int = 30) -> VerifyResult:
    task = load_task(task_id)
    real_pairs = list(task.train) + list(task.test)
    real_pass = sum(_run_ok(program, p) for p in real_pairs)

    rearc_pass = 0
    rearc_total = 0
    errors = 0
    for _ in range(n):
        try:
            pair = generate_pair(task_id)
        except GenerationError:
            errors += 1
            continue
        rearc_total += 1
        if _run_ok(program, pair):
            rearc_pass += 1

    return VerifyResult(
        task_id=task_id,
        real_pass=real_pass,
        real_total=len(real_pairs),
        rearc_pass=rearc_pass,
        rearc_total=rearc_total,
        errors=errors,
    )
