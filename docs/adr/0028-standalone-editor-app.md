# ADR-0028: A standalone, free-form, object-aware editor app (F16)

- Status: Accepted (design + architecture). Implementation sequenced after the
  F15 object-representation POC (ADR-0027), since the editor's object/layers
  model tracks F15's outcome.
- Date: 2026-09-15
- Deciders: repo owner, via conversation + approved mockups, 2026-09-15

## Context

`docs/QUESTIONS.md` F16 (`docs/questions/f16-editor-revamp.md`) revamps the
F13 "Play" editor. Today that editor (`viz/frontend/src/play.ts` +
`viz/backend/play.py`, ADR-0017) is a **form over the action API**: a dropdown
of action names, integer arg boxes, a "Step" button, a static target grid. It
writes real `runs/<run_id>/` dirs through the unmodified `EpisodeWriter`
(`algo="human"`), so human solves are warm-start-compatible (ADR-0009). The
user tried it and found it not human-friendly, and wants it rebuilt as "a
minimal Photoshop, specialized for ARC grids," and **decoupled** from the
training visualizer.

## Decision

Build a **new standalone editor frontend app**, sharing the repo and the
existing Python backend, not embedded in `viz/frontend/`.

### Architecture — separate app, shared repo + backend

- A new frontend under `apps/editor/` (or equivalent) with its own Vite
  build/dev server, so the editor and the training visualizer are distinct
  tools with distinct entry points. `viz/` stays the read-only training
  dashboard/replay tool (ADR-0006/0007).
- **Shared UI extracted into a common lib** the two apps both import: the ARC
  palette (`palette.ts`), grid canvas rendering (`grid.ts`), and shared types.
  This is the "extract shared-core" refactor, scoped to just what both apps
  genuinely share — not a full rewrite of `viz/`.
- **One backend, not two.** The editor drives the existing `play.py` in-memory
  session store (ADR-0017), extended, rather than a second server process —
  `make viz` already runs one process; a second port/lifecycle is operational
  surface a shared backend avoids. (Same rejection ADR-0017 gave the
  two-process option.)

### Interaction — free-form, object-aware, Studio layout

Direct manipulation on a canvas, not an action-picker. From the approved
mockup: a left tool rail (select/marquee, lasso, move, paint, fill,
eyedropper, stamp, region, canvas), a center canvas stage, the ARC 10-color
palette, undo/redo, a target reference with a live match indicator, and a
**Layers/Objects panel**. B's **object-cards** (per-object attribute cards)
ship as an *optional* panel toggled alongside the layers view — the intuitive
image-editor frame, with objects-as-entities exposed without forcing every
solve through a rule builder.

Must-haves, all from the interview: layers/objects panel; direct drag-move +
marquee/lasso select; undo/redo + full ARC palette; load any of the 400 tasks,
show the target, save as a run.

### Annotations and the signaller hook

Each solve carries a **free-text annotation** plus **role tags** (e.g.
`signaller: yellow divider`, `op: intersect`). This is the cheap, high-value,
LLM-ready hook the user asked for: today it's documentation; it's also the
exact supervision a future text-embedding / LLM step (F12) would consume.
**The full rule/condition builder (mockup variant B) is designed but deferred
to a later stage** — v1 is annotation + tags only, so the editor doesn't wait
on conditional-logic authoring to become useful.

### Free-form vs. warm-start, resolved

The editor's original purpose is generating warm-start demonstrations in the
agent's action space. A pure free-paint tool produces solves mapping to
nothing the agent can do. Resolution: **because the representation is going
object-centric (F15/ADR-0027), free-form object manipulation maps naturally
onto the object-level action space** — move object, recolor object,
duplicate/merge layer, combine two object-sets. So the editor stays
free-feeling *and* records a valid demonstration trace. Where a gesture has no
clean object-action analog, it is logged as a state edit rather than forced
into a fake action; the trace stays honest. This is the dependency that
sequences the editor after the F15 POC.

### Save path

Reuse `play.py`'s `save_session` → `EpisodeWriter` path (ADR-0017), extended
with an `annotation` field on the saved run (`run_meta.json` and/or a per-step
field). `runs/` is local and gitignored, so this is a normal additive schema
evolution, not a migration problem — same ethos as ADR-0017/ADR-0020.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Keep the editor embedded in `viz/frontend/` | The user explicitly wants the two decoupled — the editor is operated (direct manipulation, a working tool), the visualizer is read (dashboards, replay). One bundle conflates two jobs and one grew unusable inside the other. |
| Fully separate app *and* backend (second server/port) | Cleanest separation but real new operational surface for no benefit — `play.py`'s session store already does exactly what the editor needs; a shared backend keeps `make` one-process. Same call ADR-0017 made. |
| Keep the constrained action-picker (faithful-only) editor | It's the thing being replaced. Faithful-only was right for F13 Stage 0's warm-start goal, but the object-centric representation lets free-form and action-faithful converge, so the UX cost is no longer necessary. |
| Full rule/condition builder in v1 | Higher build cost and the hardest part to keep faithful; annotation + role tags capture the signaller intent cheaply now and leave the builder as a clean later stage. |

## Consequences

- A new `apps/editor/` frontend, its own build; `viz/frontend/` unchanged in
  purpose (still the training visualizer). Shared UI (`palette.ts`,
  `grid.ts`, types) extracted into a common lib both import — a bounded
  refactor, not a `viz/` rewrite.
- `viz/backend/play.py` gains editor-facing capability (richer state edits,
  an `annotation` field on save) alongside its existing routes; the read-only
  run-browsing/dashboard routes stay as read-only as ADR-0006/0007 designed.
- `arc_env/episode_log.py` gains an additive `annotation` field.
- **Sequenced after ADR-0027's POC.** The editor's object/layers model is the
  F15 object model rendered for a human; building it before the representation
  is proven risks designing UI around a model that changes. The free-form
  pixel-edit + annotation scaffolding is independent and could start earlier
  if the user chooses to parallelize.
- The mockups (two interaction directions, loaded with real task `6430c8c4`)
  are the approved visual reference for the build; `docs/questions/
  f16-editor-revamp.md` records the direction chosen (A's frame + B's
  object-cards optional).
