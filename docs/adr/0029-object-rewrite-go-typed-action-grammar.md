# ADR-0029: GO on the object rewrite — headlined by a typed action grammar (F15)

- Status: Accepted
- Date: 2026-09-15
- Deciders: repo owner, via conversation + POC results, 2026-09-15
- Supersedes: ADR-0027's "POC-gated" hold (the gate is now resolved); begins
  superseding ADR-0001's single-mutable-grid foundation for the new track.

## Context

ADR-0027 pursued an object-centric representation but **gated the commitment**
on a two-gate POC over the 8 set-op-cluster tasks (`6430c8c4`, `94f9d214`,
`ce4f8723`, `f2829549`, `fafffa47`, `99b1bc43`, `3428a4f5`, `dae9d2b5`) — all
curated (F14's dual-slot expresses them) yet solved by no arm in the 2026-09-15
pass. The POC ran (`research/arc-object-poc/`, `RESULTS.md`, PR #71). Results:

- **Gate 1 (expressibility): PASS 8/8.** A tiny 5-verb object vocabulary
  (`split → select_color → select_color → combine → paint`) expresses all 8 on
  their real train/test pairs, differing only in args — cleaner than the
  shipped dual-slot + region-scoping bolt-ons (ADR-0020/0022).
- **Gate 1 (re-arc generality): a task property, not a representation one.**
  The object program fails the strict 30-fresh-instances bar (0–3/30) — but the
  *literal official arc-dsl solvers fail just as hard* (0–2/30), and the object
  program equals or beats them on every task. re-arc randomizes the
  background/blank colors and sizes both hardcode (the ADR-0025/0026 pattern).
  Orthogonal to objects-vs-pixels; fixable in either representation by
  derived-color selectors.
- **Gate 2 (searchability): the decisive, reframing finding.** Free-form GP
  over just 5 object actions still finds **0/8** (~360k evals/task; the same
  failure mode as the flat 85-action space — the dense-similarity reward
  plateaus at a deceptive ~0.88 local optimum). But a **typed grammar** over
  the *same* actions — fixing the skeleton, searching only typed args — finds
  **8/8 in seconds**. The exact object programs score a perfect `(1.0, 1.0)`;
  GP simply can't *find* them without structure.

**The lesson:** the bottleneck is action-space **structure** (a typed /
compositional grammar) plus the **fitness landscape**, not pixels-vs-objects
and not action count. A flat "pick any of N actions per step" genome fails
whether N=85 or N=5.

## Decision

**GO** on the object-centric rewrite — but re-centered on what the POC actually
proved. The rewrite's four design commitments, in priority order:

1. **A typed, compositional action grammar is the headline** — the proven lever
   (0/8 → 8/8). Actions carry typed argument slots, and the search (GP genome
   and PPO action head) operates over a *structured* space — an enforced or
   strongly-biased skeleton / grammar over verbs and object-refs — not a flat
   unstructured gene list. This is the core deliverable; everything else
   supports it.
2. **The object state model** (validated in Gate 1): the raw grid stays ground
   truth; the env emits a set of typed objects (color, cells, bbox, shape
   signature, region, role) as first-class, addressable state; actions
   reference objects by slot/attribute (`(verb, object-ref, arg)`).
3. **Derived-color object actions** — attribute queries like "the region's
   background / least-common color" — so solutions generalize across re-arc
   instances, which neither the current hardcoded actions nor the literal
   solvers do.
4. **Encoder + fitness support:** a per-object embedding plus relational
   attention (set encoder) for PPO; and, because even the typed search leaned
   on structure to beat the deceptive similarity plateau, keep an explicit hook
   for demonstration/LLM seeding (the F16 editor's annotations; F12) and/or a
   curriculum as a complementary attack on the landscape.

### Build approach — parallel, not in-place

Develop the new representation as a **new module set alongside** the existing
`arc_env`/`trainers`, not an in-place mutation of them, so the working 91-task
GP/PPO pipeline stays intact and benchmarkable during development. Cut over
only once the new track reaches parity on the current curated set *and* clears
some of the 21 unsolved tasks. Detailed slice breakdown is a separate planning
pass (a `docs/SLICES.md` update), not this ADR.

## Alternatives considered

| Option | Why not chosen |
|--------|----------------|
| Cheaper: add a typed grammar / set-op macro to the *existing* single-grid representation | Genuinely tempting given the POC (structure, not objects, is the lever) and would likely reach these 8 tasks — but it doesn't serve the F16 object-aware editor or the derived-color generality path as naturally, and repeats the "bolt another mechanism onto the flat grid" pattern F14 already showed has diminishing returns. The full rewrite subsumes it. **User chose the full rewrite.** |
| Pivot to the search/fitness side first (curriculum / novelty / demonstration seeding), no representation change | The deceptive-similarity plateau is real and shared — but folded in as commitment #4 (a complementary hook), not the primary track. On its own it wouldn't give the clean, general object vocabulary Gate 1 validated. |
| Park the representation; build F16 first | Rejected — the user's sequencing is representation-first, and the rewrite gives the editor a concrete object model to build against rather than the reverse. |
| No-go; accept 77% | Rejected — Gate 1 validated expressibility and Gate 2 gave a concrete, actionable design lever (the grammar). The path forward is clear enough to invest. |

## Consequences

- ADR-0027's POC gate is resolved (passed, with the reframing above). ADR-0001's
  single-grid foundation begins being superseded **for the new track**; the
  shipped agent keeps running on the current representation until the rewrite
  reaches parity.
- A `docs/SLICES.md` update (or a new slices doc) will break the rewrite into
  vertical slices — object model + grammar first, then encoder, then GP/PPO
  integration, then cutover — each planned and reviewed before coding, same bar
  as every prior slice.
- `research/arc-object-poc/` stays as the reference: `verify.py` (the packaged
  re-arc generalization check) and `grammar_search.py` (the typed-grammar
  demonstration) inform the real grammar design.
- **F16 (the editor)** now has a concrete, validated object model to build its
  layers/objects panel against; its annotation capability doubles as the
  demonstration-seeding hook commitment #4 names.
- `docs/questions/f15-object-representation.md` and `docs/QUESTIONS.md`'s F15 row
  are updated to point here; ADR-0027's framing is corrected (the pixel grid,
  via the dual-slot, *does* express these — the gap was searchability).
