# ADR-0007: Custom local web app with a TypeScript + Canvas frontend

- Status: Accepted
- Date: 2026-08-27
- Deciders: repo owner

## Context

The visualizer's core requirement is a scrubbable step-by-step replay of an
agent "playing" an ARC grid, plus a training-metrics dashboard, reading from
the flat-file `runs/` layout (ADR-0006). Candidates were a rapid dashboard
framework (Streamlit/Gradio) or a custom small web app.

## Decision

Build a custom local web app: a small backend serving `runs/` contents as
JSON, and a frontend written in **TypeScript** (user's explicit preference
over JavaScript) rendering the grid on an HTML Canvas with play/pause/step/
speed controls, following the palette and grid-drawing conventions of
`third_party/ARC-AGI/apps/testing_interface.html` (the official ARC-AGI human
testing interface) for visual consistency with the source dataset's own
tooling.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Streamlit / Gradio | Fast to stand up, but their rerun-per-interaction model handles a scrubbable step timeline and custom canvas rendering clumsily — "watch it play like a game" wants smooth, responsive step controls, not a form that reruns a script per click. |
| Plain JavaScript frontend | User's explicit preference is TypeScript over JavaScript for the type safety on grid/trajectory data structures shared with the backend's JSON schema. |

## Consequences

- Requires standing up a small frontend build/type-check pipeline (TypeScript
  compiler at minimum) — more upfront setup than Streamlit, in exchange for a
  much better game-like replay experience.
- The grid-rendering component (palette, gridlines, canvas sizing) can
  directly reference `third_party/ARC-AGI/apps/js` for the established ARC
  color palette and conventions, rather than inventing a new one.
- Backend and frontend need an agreed JSON schema for run/episode/metrics
  data (derived directly from the JSONL formats in ADR-0006) — this schema
  should be written down once (PLAN.md Implementation decisions) rather than
  inferred independently by each side.
