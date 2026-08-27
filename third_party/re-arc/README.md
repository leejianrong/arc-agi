# re-arc (vendored)

Source: https://github.com/michaelhodel/re-arc
Commit: `e5b7f1d06362a76f9d3b8c25154ff1fafca897ce` (2025-02-24)
License: MIT (see `LICENSE`)

Vendored per `docs/PLAN.md` Scope / `docs/SLICES.md` V2 build plan step 3:
`generate_<task_id>(diff_lb, diff_ub) -> {"input": Grid, "output": Grid}`
gives PPO additional practice instances per task, beyond ARC's native ~3-5
train pairs.

## Files, and two deliberate deviations from a verbatim vendor

- `dsl.py` - re-arc's **own** copy of the DSL, kept separate from
  `third_party/arc-dsl/dsl.py` (our action-space executor) rather than
  merged. The two have diverged in minor ways upstream (e.g. `argmax`/
  `argmin` default handling) since re-arc pins its own snapshot; using
  arc-dsl's copy here would risk `generators.py` behaving differently than
  upstream tested it against. This project never calls re-arc's `dsl.py`
  directly - it's purely an implementation detail of `generators.py`.
- `generators.py` - unmodified. `generate_<task_id>` for every one of V1's
  11 curated task IDs exists here (verified before vendoring).
- `utils.py` - **trimmed**, not verbatim: dropped `plot_task` and its
  `matplotlib`/`ListedColormap`/`Normalize` imports (visualization-only,
  and we have our own visualizer), and `fix_bugs` (a dataset-repair
  routine for `main.py`'s dataset-generation CLI, which isn't vendored).
  This keeps `matplotlib` off the dependency list - `docs/PLAN.md`
  Implementation decisions is explicit that `numpy`/`torch`/`gymnasium`
  are the only Python runtime deps beyond vendored code, and pulling in a
  plotting library for an import nothing calls would be exactly the kind
  of avoidable dependency bloat that section calls out `arc-ngps`'s CUDA
  wheel mistake for. `unifint`/`is_grid`/`format_*` (what `generators.py`
  actually needs) are kept verbatim.
- `verifiers.py` and `main.py` are **not vendored**. `verifiers.py`'s
  `verify_<task_id>` programs are the same solve logic as
  `third_party/arc-dsl/solvers.py`'s `solve_<task_id>` (checked before
  deciding not to vendor) - redundant for our purposes, since
  `arc_env/task_loader.py`'s `CURATED_TASK_IDS` already carries that same
  ground truth for V1's regression test. `main.py`'s dataset-generation/
  plotting CLI isn't needed - `arc_env` calls `generators.generate_<id>`
  directly, on demand, during rollout collection rather than
  pre-generating and writing a static dataset to disk.

Read-only, not modified further in place. Not a git submodule - plain
tracked files, consistent with `third_party/arc-dsl/` and
`third_party/ARC-AGI/`.
