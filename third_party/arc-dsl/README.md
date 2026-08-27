# arc-dsl (vendored)

Source: https://github.com/michaelhodel/arc-dsl
Commit: `635de4902a5fb4e376f27333feaa396d3f5dfdcb` (2024-10-11)
License: MIT (see `LICENSE`)

Vendored per `docs/adr/0001-arc-dsl-as-action-space.md`. `dsl.py`,
`arc_types.py`, and `constants.py` are the action-space executor (ADR-0001's
decision). `solvers.py` is also vendored, beyond ADR-0001's literal file
list, because it is required by the testing approach ADR-0001 itself
specifies as a consequence ("arc-dsl's 400 solver programs become a free
regression-test fixture") and by `docs/PLAN.md`/`docs/SLICES.md` V1's
integration test, which replays known-correct solver programs through the
env's executor.

Read-only, not modified in place. Not a git submodule — plain tracked files,
consistent with how `third_party/ARC-AGI/` is vendored.
