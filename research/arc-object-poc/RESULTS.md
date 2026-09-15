# F15 POC results — object-centric representation on the set-op cluster

Ran 2026-09-15 against ADR-0027's two gates, on the 8 "set-op cluster" tasks
(`6430c8c4`, `94f9d214`, `ce4f8723`, `f2829549`, `fafffa47`, `99b1bc43`,
`3428a4f5`, `dae9d2b5`) — all curated (F14's dual-slot expresses them) yet
solved by no arm in the 2026-09-15 training pass.

## Headline

| Gate | Test | Result |
|------|------|--------|
| **1 — expressibility** | object programs solve real train/test pairs | **PASS 8/8** |
| 1 — re-arc generality | object program vs literal solver on 30 fresh instances | both fail equally (task property, not representation) |
| **2 — searchability (free-form)** | GP over 5 object actions, unstructured genome | **0/8** (same failure as the flat 85-action space) |
| **2 — searchability (typed grammar)** | same actions, search a fixed skeleton's args | **8/8 in seconds** |

## Gate 1 — the object model expresses all 8, cleanly

A tiny 5-verb object vocabulary — `split(axis)`, `select_color(slot, region,
color)`, `combine(op)`, `paint_canvas(bg, fill)`, `paint_onto_region(fill)` —
expresses **all 8** tasks on their real train/test pairs, differing only in
args, not in bespoke per-task actions. This is cleaner than the shipped
approach, where reaching these needed a bolted-on dual-slot mechanism
(ADR-0020) plus region-scoping (ADR-0022). Objects (two region sub-grids, two
named index-sets) are first-class, addressable state; actions reference them by
slot — the `(verb, object-ref, arg)` axis ADR-0027 describes. (`run_gate1.py`,
and `tests/test_poc.py` as a deterministic check.)

### The re-arc generality result is a task property, not a representation one

The strict "30/30 fresh re-arc instances" bar (which recent ADRs use for new
*shipped* actions) fails here — but the **literal official arc-dsl solvers fail
just as hard** (0–2/30), and the object program equals or beats them on every
task:

```
  task        literal-solver   object-program   (/30 re-arc)
  6430c8c4         0/30              2/30
  94f9d214         0/30              3/30
  ce4f8723         0/30              1/30
  f2829549         2/30              2/30
  fafffa47         1/30              1/30
  99b1bc43         2/30              2/30
  3428a4f5         0/30              2/30
  dae9d2b5         0/30              0/30
```

re-arc randomizes the background/blank colors and sizes the literal solvers
hardcode (the exact ADR-0025/0026 finding). This is **orthogonal to
objects-vs-pixels** — a derived-color selector ("the region's background/
least-common color", a natural *object attribute* query) would help both
representations equally. So the strict bar measures solver-tightness, not
representation power; the right Gate-1 bar is "expresses the real pairs, ≥ the
literal solver on re-arc" — met 8/8.

## Gate 2 — the real finding: structure, not size

The set-op tasks were unsolved by GP/PPO over the flat 85-action space. The
naive thesis was "a smaller object action space is more searchable." **That's
false, and the falsification is the most useful thing the POC produced:**

- **Free-form GP over just 5 object actions still finds 0/8**, even at pop 600 ×
  gens 200 × 3 seeds (~360k evals/task). It plateaus at ~0.88 mean similarity —
  the dense delta-similarity reward (ADR-0005) has a deceptive local optimum
  (right shape, ~84% right content), and the last mile to exact-match is a
  narrow ridge the unstructured genome doesn't climb. The exact hand programs
  score a perfect `(1.0, 1.0)` under the same fitness, so the target is
  reachable — GP just doesn't *find* it. (`gp_object.py`.)
- **Typed grammar over the *same* actions finds 8/8** in a few thousand to ~100k
  arg-samples each (<15s/task), by fixing the `split → select → select →
  combine → paint` skeleton and searching only the typed args.
  (`grammar_search.py`.)

**Conclusion: the bottleneck is action-space *structure* + the fitness
landscape, not pixels-vs-objects and not action-count.** A flat "pick any of N
actions per step" genome fails whether N=85 (pixels) or N=5 (objects). Adding
compositional/typed structure is what unlocks discovery.

## Recommendation — conditional GO, re-scoped thesis

The object representation is validated as **expressive and compact** (Gate 1),
and it's the natural home for the derived-color attribute queries re-arc
generality needs. But the POC moves the rewrite's center of gravity:

1. **The rewrite's headline should be a *typed, compositional action grammar*
   over an object model — not "objects instead of pixels" alone.** The grammar
   (typed slots, an enforced/biased skeleton, or a small macro layer) is the
   thing that turned 0/8 into 8/8. Without it, an object rewrite inherits the
   same searchability failure.
2. **Add derived-color object actions** ("select the region's background /
   least-common color") so solutions generalize across re-arc instances, which
   neither the current hardcoded actions nor the literal solvers do.
3. **The fitness landscape needs help too.** Even the typed search leaned on
   structure to beat the deceptive similarity plateau — demonstration/LLM
   seeding (the F16 editor's annotations; F12) or a curriculum would attack the
   same problem from the search side.

**A cheaper alternative worth weighing:** since the win came from *structure*,
much of it might be reachable by adding a typed grammar / set-op macro layer to
the *existing* representation (closer to what F14's dual-slot already started),
without a full env/encoder rewrite. The POC can't rule that out — it shows
structure is necessary, not that a full object rewrite is the only way to get
it.

This is a genuine go/no-go decision for the user, not an automatic proceed.

## Reproduce

```
uv run python research/arc-object-poc/run_gate1.py --n 30       # Gate 1
uv run python research/arc-object-poc/gp_object.py --seeds 3    # Gate 2 free-form (0/8)
uv run python research/arc-object-poc/grammar_search.py         # Gate 2 typed (8/8)
uv run pytest research/arc-object-poc/tests/ -q                 # deterministic self-check
```

Heavier future sweeps (per the 2026-09-15 directive) belong on RunPod via
`runpod-jobs`; everything above runs in seconds locally, so no pod was needed
for this POC.
