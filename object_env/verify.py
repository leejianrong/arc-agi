"""The re-arc generalization verifier, promoted from the POC (SLICES.md V5 #4).

`verify(program, task_id)` runs an object-space `Program` against (a) the task's
real train/test pairs and (b) N freshly-generated `re-arc` instances, reporting
pass rates — the "N fresh instances -> pass rate" check ADRs 0023-0026 did with
throwaway scripts, here reusing `arc_env.re_arc.generate_pair` and
`arc_env.task_loader.load_task` (same tuple-of-tuples Grid type, no conversion).
"""

from dataclasses import dataclass

from arc_env.re_arc import GenerationError, generate_pair
from arc_env.tasks import load_task
from object_env.grammar import Program


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
        flag = "PASS" if self.real_ok else "----"
        extra = f"  ({self.errors} gen-errors)" if self.errors else ""
        return (
            f"{flag}  {self.task_id}  real {self.real_pass}/{self.real_total}  "
            f"re-arc {self.rearc_pass}/{self.rearc_total}{extra}"
        )


def _run_ok(program: Program, pair) -> bool:
    try:
        return program.run(pair.input) == pair.output
    except Exception:  # noqa: BLE001 - a malformed program can raise any DSL error
        return False


def verify(program: Program, task_id: str, n: int = 30) -> VerifyResult:
    task = load_task(task_id)
    real_pairs = list(task.train) + list(task.test)
    real_pass = sum(_run_ok(program, p) for p in real_pairs)

    rearc_pass = rearc_total = errors = 0
    for _ in range(n):
        try:
            pair = generate_pair(task_id)
        except GenerationError:
            errors += 1
            continue
        rearc_total += 1
        rearc_pass += _run_ok(program, pair)

    return VerifyResult(task_id, real_pass, len(real_pairs), rearc_pass, rearc_total, errors)
