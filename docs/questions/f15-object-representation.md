# F15: an object-centric representation to break the 77% ceiling

**Status:** decided — **GO** (user, 2026-09-15). The POC ran and passed (with a
reframing); the full rewrite is committed, headlined by a *typed action
grammar*. See ADR-0029 for the go/no-go and the re-scoped design; ADR-0027 for
the original gated proposal. Rewrite implementation planned as slices next
(`docs/SLICES.md`), not started.

## POC outcome (2026-09-15) — the gate resolved, with a twist

The POC (`research/arc-object-poc/`, `RESULTS.md`) found something more useful
than a clean yes:

- **Gate 1 (expressibility): PASS 8/8.** A 5-verb object vocabulary expresses
  all 8 set-op tasks on their real pairs, cleaner than the shipped dual-slot.
- **re-arc generality is a task property, not a representation one.** The object
  program fails the strict 30-instance bar — but so do the *literal official
  solvers*, equally (both hardcode colors re-arc randomizes). Orthogonal to
  objects-vs-pixels; fixed in either by derived-color selectors.
- **Gate 2 (searchability): the reframing.** Free-form GP over 5 object actions
  finds **0/8** (same failure as the flat 85-action space — a deceptive ~0.88
  similarity plateau), but a **typed grammar** over the *same* actions finds
  **8/8 in seconds**. The bottleneck is action-space *structure* + the fitness
  landscape, **not** pixels-vs-objects or action count.

**Decision (ADR-0029):** GO on the object rewrite, but re-centered — the *typed
compositional action grammar* is the headline lever, the object model is the
validated substrate, derived-color actions handle generality, and
demonstration/LLM seeding (F16/F12) is kept as a complementary attack on the
landscape. Note this corrects the pre-POC framing below (which said the pixel
grid "can't express" set-ops — via the dual-slot it can; the real gap was
search).

This is an ADR-0001-scale question: it reconsiders the foundational decision
that ARC grids are a single mutable 30×30 pixel grid, with `arc-dsl`
primitives as `Grid -> Grid` actions.

## Why now

The 2026-09-15 full training pass
(`docs/results/training-pass-2026-09-15-combined.md`) put the architecture at
a hard ceiling: GP 70/91, PPO+warm-start 69/91, and **21 tasks solved by no
arm**. F11 already concluded 400/400 is out of reach under this architecture
and that breaking past it "would need a fundamentally different
representation," which was never designed. This thread designs it.

The observation the policy sees today is literally
`np.stack([padded_grid, selection_mask])` — two `(30, 30)` channels
(`arc_env/env.py`). Every mechanism added since (single selection ADR-0011,
the dual-slot `a`/`b` ADR-0020, region-scoping ADR-0022) is a **side channel
bolted onto that one flat grid**. Objects only ever exist transiently inside
one action's Python body; they are never first-class state the policy can see
or the agent can compose across steps.

## The strongest evidence: the set-op cluster

8 of the 21 unsolved tasks are one coherent family, already studied under
F14: `6430c8c4`, `94f9d214`, `ce4f8723`, `f2829549`, `fafffa47`, `99b1bc43`,
`3428a4f5`, `dae9d2b5`. Each has the same shape — **split the grid into
regions, read a color's cells from each region independently, combine the two
location-sets with a set operation, paint the result.** The single mutable
grid holds one live state; this needs two, addressed and recombined. F14's
dual-slot mechanism reached a *near-miss* cluster by bolting on yet another
side channel. F15 asks whether to stop bolting channels onto a flat grid and
make objects first-class instead.

A second motivation the user raised directly: many ARC tasks have objects
that act as **signallers / directors** ("go right if this object is blue, left
if green"). A conditional gated on an object's attribute is only expressible
when objects are first-class entities with queryable attributes (color, size,
position, shape) — exactly what this representation provides, and a natural
setup for a future text/LLM step (F12, still deferred).

## The interview (2026-09-15)

Structured interview before any design. User's answers:

- **Sequencing:** representation POC *first*, before the editor revamp (F16).
- **Ambition:** full object-centric rewrite is the goal — but gated on a POC
  proving it out on real unsolved tasks before committing.
- **POC target:** the 8-task set-op cluster above (coherent, clearly
  object-relational, a meaningful ceiling break if it works).
- **POC bar:** *expressibility, then GP finds it* — (1) hand-write
  object-space programs that solve the targets and generalize against fresh
  `re-arc` instances, then (2) show GP search over the new object action
  space actually discovers a solution for at least some. PPO deferred (the
  expensive arm; prove search finds it first).

**Compute:** Gate 1 is pure expressibility work — local, no training. Gate 2's
GP search (and any later PPO) runs on **RunPod** (potentially GPU), via the
`runpod-jobs` skill, not on this local machine — heavy runs moved off-box as
of 2026-09-15. The local box stays for light dev, tests, and the editor
mockups; the "run locally-run things serially" RAM rule only governs what
still runs locally.

## The design sketch (reviewed via mockup, 2026-09-15)

A visual decision brief was produced and approved in principle. Headline of
the target representation:

- A grid decomposes (on `reset`) into a **set of typed objects**, each with
  queryable attributes (color, cells, bbox, shape signature, region, role).
  The raw grid stays ground truth on disk; the object set is a derived,
  editable view — not a new file format.
- The **action space** changes axis: from `(primitive, args)` over a grid to
  `(verb, object-ref, arg)` — object selection becomes native, not a side
  channel.
- The **policy encoder** changes from a CNN over 2 channels to a per-object
  embedding plus relational attention (a set encoder).
- Set operations over two named object-sets become **one relational action**,
  not a per-task bespoke derived action.

Full blast-radius table and the two-gate framing: ADR-0027.

## Gates

- **Gate 1 — expressibility.** Hand-write short object-space programs that
  solve all 8 set-op tasks offline, verified to generalize against fresh
  `re-arc` instances (not just the real JSON — the bar every recent ADR held
  to). Passes at 8/8 express-and-generalize.
- **Gate 2 — GP finds it.** Run GP over the new object action space, serially,
  one task at a time; show it *discovers* a solution for at least some of the
  eight without being handed the program. Passes at ≥1 solved by search.

Only if **both** gates pass do we write the follow-up ADR(s) committing to the
full env/action/encoder rewrite. If Gate 1 fails, the object model doesn't
even express these tasks and we've spent days, not weeks — the point of
gating.

## Relationship to other threads

- Reconsiders **ADR-0001** (single-grid `arc-dsl` action space). Does not
  discard it: `arc-dsl` primitives remain the executor; the object model is a
  layer of addressable state *over* them.
- **F14** (dual-slot) is the incremental precedent this generalizes: F14
  bolted a second selection channel on; F15 asks whether objects-as-entities
  is the right home for that need instead.
- **F16** (editor revamp) is sequenced *after* this POC and its object model
  tracks F15's outcome — the editor's layers/objects panel is the same object
  model, rendered for a human.
- **F12** (LLM-driven approaches) stays deferred; the signaller/annotation
  hook keeps a cheap option open on it without committing.

**Landed by:** ADR-0027 (the gated-POC decision). Follow-up ADR(s) on the
rewrite itself are contingent on both gates passing.
