# F16: a human-friendly editor, decoupled from the visualizer

**Status:** decided (user, 2026-09-15) — design approved via mockups.
Implementation sequenced *after* the F15 object-representation POC, since the
editor's object/layers model tracks F15's outcome. See ADR-0028.

## Why

F13 shipped an interactive "Play" editor (`viz/frontend/src/play.ts` +
`viz/backend/play.py`, ADR-0017) so a human can solve tasks by hand and log
the result as a warm-start demonstration. It works, but it is a **form over
the action API**: a dropdown of action names, integer arg boxes, a "Step"
button, and a static target grid to eyeball. The user tried it and found it
not human-friendly. F16 rebuilds it as a genuine direct-manipulation editor —
"a minimal Photoshop, specialized for ARC grids" — and decouples it from the
training visualizer, which is a different tool for a different job.

## The interview (2026-09-15)

- **Editor tools:** *free-form image editing* — real direct manipulation, not
  a constrained action-picker. (Resolved against warm-start below.)
- **Decoupling:** a *separate frontend app, shared repo + backend* — its own
  build/dev server, sharing palette/grid/types libs and the existing Python
  backend (extend `play.py`, not a second server process).
- **Must-haves (all selected):** layers/objects panel; direct drag-move +
  marquee/lasso select; undo/redo + the full ARC 10-color palette; load any of
  the 400 tasks, show the target, save as a run.
- **User addition — signallers:** many ARC tasks have objects that *direct*
  behavior ("go right if blue, left if green"). Wanted the ability to define
  conditional branching/logic, **or** at least record a free-text
  annotation/explanation, to feed a future text-embedding / LLM step.

## The free-form vs. warm-start tension, resolved

The editor's original reason to exist is generating warm-start demonstrations
**in the agent's action space** (ADR-0009/ADR-0017). A pure free-paint tool
produces solves that map to nothing the agent can do. Resolution, confirmed
with the user:

**Because the representation is going object-centric (F15), "free-form" and
"action-space-faithful" converge.** Direct object manipulation — move object,
recolor object, duplicate/merge layer, combine two object-sets — maps
naturally onto the object-level action space F15 defines. So the editor is
free-feeling *and* still records a valid demonstration trace. Where a gesture
has no clean object-action analog, it is recorded as a state edit rather than
forced into a fake action; the trace stays honest.

## The mockups (reviewed and approved in principle, 2026-09-15)

Two interaction directions were built as an interactive HTML mockup (loaded
with the real set-op task `6430c8c4`):

- **A · Studio** — a familiar image editor: left tool rail, center canvas
  stage, a **Layers/Objects** panel, the ARC palette, undo/redo, a target
  reference with a match indicator, and a **Notes** panel with role tags
  (`signaller: yellow divider`, `op: intersect`).
- **B · Object Board** — leans into the signaller idea: each detected object
  is a **card with attributes**, plus a **rule builder**
  (`SPLIT at obj → TAKE blanks of A ∩ B → PAINT green`).

**Decision:** ship **A's layout as the frame, with B's object-cards available
as an optional panel** — the intuitive image-editor feel, without forcing
every solve through a rule builder, while still exposing objects-as-entities.

**Conditional/rule authoring in v1:** free-text annotation + role tags only
(the cheap, high-value, LLM-ready hook). The full rule/condition builder (B)
is designed but deferred to a later stage — it sets up F12 without committing
to it.

## Architecture

Full decision: ADR-0028. Headline: a new standalone editor frontend
(`apps/editor/` or similar) with its own Vite build, sharing extracted UI libs
(palette, grid canvas, types) with `viz/frontend/`; driven by the existing
Python backend's `play.py` session store, extended with an `annotation` field
in the saved run. `viz/` stays the read-only training visualizer.

## Relationship to other threads

- **F13/ADR-0017** is the thing being revamped and decoupled; its `play.py`
  backend session model is reused, not thrown away.
- **F15/ADR-0027** is the sequencing dependency: the editor's object/layers
  model is F15's object model rendered for a human, so the editor build waits
  on the F15 POC. The free-form pixel-edit and annotation parts are
  independent of F15 and could ship earlier if wanted.
- **F12** (LLM-driven approaches) stays deferred; annotations + role tags are
  the cheap option kept open on it.

**Landed by:** ADR-0028 (design + architecture). Implementation sequenced
after the F15 POC.
