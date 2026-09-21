# V6 demo: GP over the object grammar vs. flat GP (`runpod-v6-20260922b`)

SLICES.md V6's demo ask: does `trainers/gp_object` (a GP trainer over `object_env`'s typed
action grammar, PR #80) solve the 8 set-op-cluster tasks the flat 85-action GP gets 0/8 on, and
hold parity on a sample of already-solved curated tasks? Run on RunPod (`gp,gp-object` arms of
`scripts/training_pass.py`) against merged `main` (PR #81, after fixing a `DerivedColor`
episode-logging crash the first sweep attempt caught), same GP hyperparameters both arms already
default to (population 200, generations 100, seed 0 - true apples-to-apples).

Tasks: the 16-task V5 fixture basket (`object_env/programs.py`) - the 8 set-op tasks plus its 8
non-set-op tasks (move/recolor/crop/canvas/transform), chosen because every one of them has a
**verified, hand-written object-grammar solution** (`tests/test_object_grammar_regression.py`),
which isolates the question this slice actually asks ("can blind search *find* a solution the
grammar can express") from "can the grammar express this task at all" (untested for tasks
outside this basket).

## Headline: the answer is no, on both halves of the ask

| | Set-op cluster (8) | Non-set-op sample (8) | Total |
|---|---|---|---|
| Flat `gp` (baseline) | 0/8 | 5/8 | 5/16 |
| `gp-object` (V6) | **0/8** | **2/8** | 2/16 |

- **Set-op cluster: 0/8 for gp-object, matching flat GP's own 0/8** - the grammar-typed genome
  does not (yet, at this budget) clear the cluster ADR-0029 named as the central test. This
  matches the local finding documented in `tests/test_gp_object_evolve.py`'s slow test: a full
  local survey (several budgets, up to 15x this sweep's) found the same reproducible similarity
  plateau (0.68-0.88 depending on task) regardless of search budget - a landscape problem, not
  (at these scales) a budget-starved one.
- **Non-set-op sample: gp-object is currently *below* parity, not at it** - 2/8 vs. flat GP's
  5/8. The 3 tasks flat GP solves that gp-object doesn't (`25ff71a9`, `1f85a75f`, `23b5c85d`) all
  have a **known 2-step object-grammar solution** (`select_* -> move_object` /
  `select_* -> crop_to_object`, per `object_env/programs.py`) - the grammar can express them
  (proven by the regression suite), but blind search from scratch didn't rediscover them within
  100 generations at this population. The 2 tasks it does solve (`5582e5ca`, `b1948b0a`) are both
  single-step (`canvas_mostcolor`, `replace_color`) - solved instantly (generation 0) by both this
  sweep and local testing.
- The flat `gp` column reproduces `docs/results/training-pass-2026-09-15-combined.md`'s numbers
  for all 16 of these tasks exactly (same seed, same hyperparameters) - a useful consistency check
  that this sweep's harness and settings match the standing baseline.

## What this means for V6's central open question

SLICES.md V6 frames this as a genuinely open, unguaranteed test ("does grammar-typed search
generalize past the 8 ... now under search rather than hand-written programs") - not a
foregone conclusion. The measured answer here is a real negative on both halves at this budget:
the position-typed genome (`trainers/gp_object/genome.py`, each gene an index into the grammar's
*current* legal-step menu) is a structural improvement over the free-form design the F15 POC
already falsified (0/8, same failure mode) - it guarantees every produced program is grammar-valid
and every gene is type-sensible at its position, so search time isn't wasted on illegal steps -
but it does not by itself fix the *fitness-landscape* half of the problem ADR-0029 also names.
Plain mutation/crossover pressure over a dense similarity reward gets stuck on deceptive local
optima even in this narrower, type-filtered space, for tasks 2+ steps deep.

ADR-0029 commitment #4 (demonstration/LLM seeding, curriculum) is the named follow-up attack on
exactly this landscape problem, explicitly out of V6's scope. Given this sweep's evidence, it -
or a comparably structural search-side change - looks like a prerequisite for V6's coverage claim
to hold, not an optional refinement.

## Solved totals

| Arm | Solved | Runs |
|-----|--------|------|
| gp | 5 | 16 |
| gp-object | 2 | 16 |

## Per-task

| Task | Arm | Solved | Ever | Wall (s) | Peak RSS (MB) | rc |
|------|-----|--------|------|----------|---------------|----|
| 6430c8c4 | gp | no | - | 16 | 288 | 0 |
| 6430c8c4 | gp-object | no | - | 19 | 226 | 0 |
| 94f9d214 | gp | no | - | 11 | 231 | 0 |
| 94f9d214 | gp-object | no | - | 21 | 226 | 0 |
| ce4f8723 | gp | no | - | 12 | 231 | 0 |
| ce4f8723 | gp-object | no | - | 22 | 226 | 0 |
| f2829549 | gp | no | - | 11 | 233 | 0 |
| f2829549 | gp-object | no | - | 20 | 226 | 0 |
| fafffa47 | gp | no | - | 8 | 231 | 0 |
| fafffa47 | gp-object | no | - | 21 | 227 | 0 |
| 99b1bc43 | gp | no | - | 10 | 231 | 0 |
| 99b1bc43 | gp-object | no | - | 22 | 226 | 0 |
| 3428a4f5 | gp | no | - | 13 | 231 | 0 |
| 3428a4f5 | gp-object | no | - | 25 | 227 | 0 |
| dae9d2b5 | gp | no | - | 13 | 234 | 0 |
| dae9d2b5 | gp-object | no | - | 21 | 226 | 0 |
| 25ff71a9 | gp | yes | - | 4 | 231 | 0 |
| 25ff71a9 | gp-object | no | - | 21 | 226 | 0 |
| a79310a0 | gp | no | - | 6 | 231 | 0 |
| a79310a0 | gp-object | no | - | 21 | 226 | 0 |
| ea32f347 | gp | no | - | 69 | 245 | 0 |
| ea32f347 | gp-object | no | - | 24 | 226 | 0 |
| 1f85a75f | gp | yes | - | 373 | 246 | 0 |
| 1f85a75f | gp-object | no | - | 23 | 226 | 0 |
| 23b5c85d | gp | yes | - | 21 | 239 | 0 |
| 23b5c85d | gp-object | no | - | 24 | 228 | 0 |
| 1c786137 | gp | no | - | 99 | 240 | 0 |
| 1c786137 | gp-object | no | - | 23 | 228 | 0 |
| 5582e5ca | gp | yes | - | 6 | 230 | 0 |
| 5582e5ca | gp-object | yes | - | 2 | 226 | 0 |
| b1948b0a | gp | yes | - | 5 | 231 | 0 |
| b1948b0a | gp-object | yes | - | 2 | 226 | 0 |
