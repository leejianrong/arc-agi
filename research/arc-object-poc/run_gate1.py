"""Gate 1 — expressibility + generality. Run every hand-written object-space
program against its real pairs and 30 fresh re-arc instances.

    uv run python research/arc-object-poc/run_gate1.py [--n 30]

Gate 1 passes when all 8 tasks are real-exact and re-arc-general (30/30, or a
documented degenerate-edge tolerance per ADR-0025/0026 precedent).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from programs import PROGRAMS  # noqa: E402
from verify import verify  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="re-arc instances per task")
    args = ap.parse_args()

    print(f"Gate 1 — object-space expressibility + generality (n={args.n} re-arc/task)\n")
    results = []
    for task_id, program in PROGRAMS.items():
        r = verify(program, task_id, n=args.n)
        results.append(r)
        print("  " + r.line())

    n_real = sum(r.real_ok for r in results)
    n_full = sum(r.real_ok and r.rearc_pass == r.rearc_total for r in results)
    print(f"\n  real-exact: {n_real}/{len(results)}   fully-general: {n_full}/{len(results)}")
    print("\nGATE 1: " + ("PASS" if n_full == len(results) else "REVIEW (see per-task above)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
