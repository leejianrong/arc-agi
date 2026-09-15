# ADR-0027: An object-centric representation, gated on a two-gate POC (F15)

- Status: Proposed — POC-gated. The *direction* is accepted; the commitment to
  the full rewrite is contingent on both gates below passing. A follow-up ADR
  records that commitment (or the decision to stop).
- Date: 2026-09-15
- Deciders: repo owner, via conversation + approved design mockup, 2026-09-15

## Context

`docs/QUESTIONS.md` F15 (`docs/questions/f15-object-representation.md`) asks
whether to model ARC grids as objects on composable layers instead of a single
mutable pixel grid. This reconsiders **ADR-0001**, the foundational decision
that the action space is `arc-dsl` primitives over one 30×30 grid.

The 2026-09-15 training pass hit a hard ceiling: GP 70/91, PPO+warm-start
69/91, **21 tasks solved by no arm**. F11 already concluded 400/400 is
unreachable under this architecture and that breaking past it "would need a
fundamentally different representation," never designed. The policy's
observation is literally `np.stack([padded_grid, selection_mask])` — two
`(30, 30)` channels (`arc_env/env.py`). Every mechanism since (selection
ADR-0011, dual-slot ADR-0020, region-scoping ADR-0022) is a **side channel on
that one flat grid**; objects exist only transiently inside a single action's
Python body.

**The concrete evidence.** 8 of the 21 unsolved tasks are one family (F14's
set-op cluster): `6430c8c4`, `94f9d214`, `ce4f8723`, `f2829549`, `fafffa47`,
`99b1bc43`, `3428a4f5`, `dae9d2b5`. Each: split into regions, read a color's
cells from each independently, combine the two location-sets with a set
operation, paint the result. One flat grid holds one live state; this needs
two, addressed and recombined. The user also raised **signaller/director**
tasks ("go right if this object is blue") — a conditional gated on an object's
attribute, expressible only when objects are first-class entities.

## Decision

**Pursue an object-centric representation, but do not commit to the rewrite
until a two-gate POC proves it out on the 8-task set-op cluster.** This ADR
records the direction, the target design sketch, and the gates. It does not
change any shipped code.

### Target representation (the sketch approved via mockup)

- On `reset`, the env segments the raw grid into a **set of typed objects**,
  each carrying queryable attributes: color, cell set, bounding box, shape
  signature, region membership, and a role tag. The **raw grid stays ground
  truth**; the object set is a derived, editable view — `arc-dsl` primitives
  remain the executor underneath, so ADR-0001 is generalized, not discarded.
- The **action space** changes axis: from `(primitive, args)` over a grid to
  `(verb, object-ref, arg)`. Object selection becomes native rather than a
  side channel. Set operations over two named object-sets become **one
  relational action**, not a per-task bespoke derived action (the pattern
  ADR-0020 through ADR-0026 kept adding).
- The **policy encoder** changes from a CNN over 2 channels to a per-object
  embedding plus relational attention (a set encoder), so "the blue object"
  and "cells of color K in region A" are things the network can name.

### The gates

- **Gate 1 — expressibility.** Hand-write short object-space programs that
  solve all 8 target tasks offline, and verify they generalize against fresh
  `re-arc` instances (brute-force / direct replay against ~30 fresh instances
  per task, not just the real JSON — the exact bar ADR-0023/0025/0026 held
  every candidate to). **Passes at 8/8 express-and-generalize.** A partial pass
  is a signal to narrow scope, not to proceed silently.
- **Gate 2 — search finds it.** Run GP over the new object action space and
  show it *discovers* a solution for at least some of the eight without being
  handed the program. **Passes at ≥1 solved by search.** PPO is deferred — GP
  is the cheaper, faster discovery signal; a PPO arm is a later question.

**Only if both gates pass** do we write the follow-up ADR(s) committing to the
env/action/encoder/GP/log/viz rewrite. If Gate 1 fails, the object model
doesn't even express these tasks and the cost was days, not weeks.

### Where the POC lives, and on what hardware

The POC is exploratory and must not touch the shipped agent's code paths until
a follow-up ADR commits to it. It lives in its own scratch/research location
(e.g. `research/arc-object-poc/`), reusing `re-arc` and the `arc-dsl` executor
as libraries but not modifying `arc_env/` or `trainers/`. Gate 1 is pure
Python expressibility work — local. **Gate 2's GP search runs on RunPod**
(potentially GPU), via the `runpod-jobs` skill, not on this ~7.8 GiB local
machine — heavy runs moved off-box as of 2026-09-15. The serial-to-avoid-OOM
rule only governs anything that still runs locally.

## Mechanism — the blast radius, named honestly

If the gates pass and the rewrite proceeds, it touches (roughly, worst-first):

| Component | Today | After the object model | Scope |
|-----------|-------|------------------------|-------|
| Env / observation (`arc_env/env.py`) | `(2, 30, 30)` tensor | emits an object set + relations; grid kept as ground truth | big |
| Action space (`arc_env/actions.py`) | `(primitive, args)` over grid | `(verb, object-ref, arg)`; selection native | big |
| PPO encoder (`trainers/ppo/network.py`) | CNN over 2 channels | per-object MLP + relational attention (set encoder) | big |
| GP genome (`trainers/gp/`) | flat gene list | genes reference object slots; same flat-list spirit | med |
| Episode log (`arc_env/episode_log.py`) | grid + selection mask | + object set per step, + annotation field (F16) | med |
| Visualizer / editor (`viz/`, F16) | grid + amber overlay | layers/objects panel — F16 wants this anyway | low |

This is genuinely ADR-0001-scale, which is exactly why it is gated rather than
started.

## Alternatives considered

| Option | Why not (now) |
|--------|----------------|
| Commit to the full rewrite up front, no POC | ADR-0001-scale bet with no proof it clears the ceiling; a failed Gate 1 would mean weeks spent on a representation that can't even express the target tasks. Gating costs days to de-risk weeks. |
| Additive object layer (keep the flat grid, add an object channel + a few object actions) | The evolutionary path F14/ADR-0020..0026 already walks — each step bolts another side channel on and the ceiling holds. Doesn't test whether objects-as-primary breaks it; risks a third representation that's neither. Kept as the fallback if the POC underwhelms. |
| Keep adding bespoke per-task derived actions | Technically reaches many tasks (ADR-0020's own audit finding) but each hardcodes one task's constants and barely tests whether search *discovers* anything — the "single-task special case dressed up as a mechanism" pattern ADR-0013/0015/0020 already argued against. Doesn't touch the set-op cluster's real need. |
| Do nothing; accept 77% | Legitimate — but the user explicitly wants to attempt the ceiling, and the set-op cluster is a coherent, well-evidenced target. The POC is the cheap way to find out. |

## Consequences

- No shipped code changes in this ADR. `arc_env/`, `trainers/`, `viz/` are
  untouched until a follow-up ADR commits to the rewrite.
- A POC lands under a research/scratch path with its own throwaway harness;
  its results (per-task expressibility + `re-arc` generalization numbers, and
  GP discovery outcomes) get written up as the input to the follow-up
  go/no-go ADR.
- Heavy POC compute (Gate 2 GP) provisions RunPod via `runpod-jobs` with
  cost-safe teardown, not the local machine.
- **F16 (the editor revamp) is sequenced after this POC** — its object/layers
  model is this object model rendered for a human, so building it first would
  risk designing UI around an unproven representation. Its free-form pixel-edit
  and annotation parts are independent and could proceed earlier if wanted.
- ADR-0001 is not superseded by this ADR; it is put under reconsideration.
  The follow-up ADR either supersedes ADR-0001's single-grid foundation (gates
  passed) or reaffirms it (gates failed), with the POC evidence on record
  either way.
