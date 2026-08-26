# arc-agi

Revamping this project to tackle ARC-AGI-1 with a reinforcement-learning and/or
evolutionary-algorithm agent, plus a local visualizer to watch training runs and
watch a trained agent "play" a task step-by-step like a game.

Planning artifacts (read these before making architectural changes):
- `docs/QUESTIONS.md` — decision register: every open question, its status, and
  its answer.
- `docs/PLAN.md`, `docs/SLICES.md`, `docs/adr/` — written once the pending
  research below lands.

## Repo layout

- `third_party/ARC-AGI/` — vendored official ARC-AGI-1 dataset
  (`data/training`, `data/evaluation`) and the human testing interface. Plain
  tracked files (not a submodule) — treat as read-only upstream content.
- `legacy/` — the original geometric-transform + color-bijection baseline
  (`baseline.py`, `evaluate.py`, `arc_io.py`). Kept as a reference/sanity-check
  baseline, not part of the new agent.
- `research/arc-ngps/` — a prior, half-built *supervised program-synthesis*
  scaffold (Perceiver encoder, DSL AST + executor, beam search). A different
  paradigm from RL/evolutionary; parked pending the DSL/action-space decision
  (see `docs/QUESTIONS.md` F1). Its `grid_ops.py` executor primitives are a
  candidate action-space library.
- `docs/` — planning artifacts (PLAN, ADRs, SLICES, QUESTIONS).

## Research subagent policy

For ARC-AGI technical research tasks — DSL/action-space survey, RL or
evolutionary-algorithm literature and prior-art review, or similar deep-dive
research needed to settle an open decision in `docs/QUESTIONS.md` — spin up
**at most 2 subagents at a time**. This is a deliberate cap: keep research
focused and reviewable rather than fanned out into results nobody reads.
