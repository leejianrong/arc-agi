# arc-dsl / re-arc survey — action-space & data-generation foundation

Sources fetched and read directly (raw source, not summaries):
- https://github.com/michaelhodel/arc-dsl — `dsl.py` (1524 lines), `solvers.py` (6576 lines), `main.py`, `LICENSE`
- https://github.com/michaelhodel/re-arc — `main.py`, `generators.py` (14962 lines), `LICENSE`
- Local: `research/arc-ngps/src/arc_ngps/dsl/{ast,types,parse}.py`, `research/arc-ngps/src/arc_ngps/executor/{grid_ops,runtime}.py`

## 1. DSL primitive catalog

`dsl.py` defines exactly **160 primitives** (`grep -c "^def " dsl.py`). They form a typed functional toolkit over a small set of ARC-native types (`Grid` = tuple of tuples of ints, `Object`/`Patch` = frozenset of `(color, (row, col))`, `Indices` = frozenset of `(row, col)`, plus `Integer`, `IntegerTuple`, `Boolean`, `Callable`). Roughly by category:

- **Grid geometry**: `rot90/180/270`, `hmirror`, `vmirror`, `dmirror`, `cmirror`, `hconcat`, `vconcat`, `hsplit`, `vsplit`, `tophalf/bottomhalf/lefthalf/righthalf`, `trim`, `compress`, `upscale`/`downscale`/`hupscale`/`vupscale`.
- **Canvas / shape-changing**: `canvas(value, dims)` (new blank grid of arbitrary size), `crop(grid, start, dims)`, `subgrid`, `fill`, `paint`, `cover`, `underfill`, `underpaint`, `replace`, `switch`, `cellwise`.
- **Object/connected-component ops**: `objects(grid, univalued, diagonal, without_bg)` (the general connected-components extractor — our `select_color_objs` is a narrow special case of this), `partition`, `fgpartition`, `colorfilter`, `sizefilter`, `toobject`, `asobject`, `normalize`, `shift`, `move`, `occurrences`, `frontiers`.
- **Geometry queries**: `ulcorner/urcorner/llcorner/lrcorner`, `uppermost/lowermost/leftmost/rightmost`, `center`, `centerofmass`, `height/width/shape`, `bordering`, `adjacent`, `manhattan`, `gravitate`, `position`, `corners`, `box/inbox/outbox`, `delta`, `backdrop`.
- **Color/palette**: `mostcolor`, `leastcolor`, `palette`, `numcolors`, `colorcount`, `color`.
- **Set/functional plumbing**: `sfilter`, `mfilter`, `extract`, `apply`, `mapply`, `rapply`, `papply`, `mpapply`, `prapply`, `compose`, `chain`, `fork`, `rbind`, `lbind`, `power`, `branch`, `matcher`.
- **Arithmetic/logic on ints and 2-vectors**: `add/subtract/multiply/divide/invert/increment/decrement/sign/crement`, `both/either/flip`, comparisons.
- **Lines/rays**: `connect`, `shoot`, `vfrontier/hfrontier`, `hperiod/vperiod` (periodicity detection).

These compose as a **typed functional pipeline**: every primitive is a pure function `(typed args) -> typed value`, and solver programs are straight-line SSA chains of primitive calls (see §2). There is no shared mutable state and no side effects — clean to sandbox/execute.

## 2. Solver program structure

`solvers.py` contains exactly **400 functions**, one per ARC-AGI-1 training task, named `solve_<task_id>(I)`. Each is a flat sequence of `variable = primitive(args)` lines ending in `return O`, e.g.:

```python
def solve_67a3c6ac(I):
    O = vmirror(I)
    return O

def solve_9172f3a0(I):
    O = upscale(I, THREE)
    return O
```

More complex tasks chain many more calls — average body length across all 400 solvers is **~15 lines** (~13 primitive calls after subtracting the `def`/`return` lines), with some running much longer. `main.py`'s `test_solvers_formatting` even enforces this shape mechanically: every line must be `var = function(args)`, no other constructs, args must be prior variables, DSL functions, named constants (`constants.py`), or the input `I`.

**This is directly steppable as an RL action space, not just a synthesis target.** Because every solver is a linear sequence of discrete primitive applications, the natural RL framing is: **one action = one primitive call** (choose a function from the 160-item catalog, choose its typed arguments), and an episode is the sequence of calls a solved task would need (~13 steps on average, more for harder tasks). This maps exactly onto "watch the agent play like a game" — each step visibly transforms the grid (or produces an intermediate typed value like an object set, which then feeds a later `paint`/`fill` step). Higher-order primitives (`compose`, `fork`, `chain`, `rbind`/`lbind`, `power`) are a complication for a step-by-step action space since they build closures rather than immediately transforming a grid — the recommendation below addresses this.

## 3. Variable-output-shape handling (resolves F2)

The DSL has first-class primitives for exactly this:
- `canvas(value, dimensions)` — allocate a new grid of **any** height/width, filled with one color.
- `crop(grid, start, dims)` / `subgrid` — cut out an arbitrary rectangular region.
- `hconcat`/`vconcat`, `hsplit`/`vsplit` — combine/divide grids, changing overall dimensions.
- `upscale`/`downscale`, `hupscale`/`vupscale` — integer-factor resize.
- `trim`, `compress` — shrink by removing border/uniform rows-cols.

Solvers freely mix these — e.g. `solve_9172f3a0` triples every dimension via `upscale`, and many others build an output via `canvas(...)` sized differently from the input, then `paint`/`fill` into it. **This means variable-output-shape is a first-class, well-trodden case in this DSL**, not an edge case bolted on. Our own scaffold (`research/arc-ngps`) has *no* such primitive at all — its `paint()` only ever writes into the input grid's existing bounds (see §6), so output shape is always frozen equal to input shape today.

Practical implication for the RL environment: give the agent a fixed-size scratch canvas (e.g. the ARC max of 30×30) plus an explicit "commit output" step that crops it to whatever the agent painted (mirroring `canvas` + `crop`/`subgrid`), so the action space stays fixed-arity per step while still allowing any final output shape. This unblocks doing variable-shape tasks in milestone 1 if wanted, rather than being forced into the same-shape-only restriction — the shape problem is a solved pattern in this DSL, not a genuine unknown.

## 4. License

Both repos are **MIT licensed** (`arc-dsl` copyright Michael Hodel 2023; `re-arc` copyright Michael Hodel 2024). Fully permissive — vendoring, modifying, and relicensing our derived work is unproblematic. One caveat found during search: the original `arc-dsl` repo stopped accepting pull requests, and there's a community-maintained fork continuing it (`arc-dsl-2/arc-dsl-2`) for ARC-AGI-2 support and quality-of-life fixes — worth a quick look before vendoring if we want fixes beyond the frozen original, but not required for ARC-AGI-1 which is our stated scope.

## 5. re-arc for training-data generation

`re-arc`'s `generators.py` has exactly 400 functions, `generate_<task_id>(diff_lb, diff_ub) -> {'input': grid, 'output': grid}`, one per training task, each procedurally constructing a *fresh* input/output pair (not sampled from the original 400) using the same DSL vocabulary, with two difficulty knobs threaded through random choices (`unifint`, `choice`, `sample` — see `dbc1a6ce`/`2281f1f4` examples read directly). `verifiers.py` (not fully read, but referenced throughout `main.py`) holds one verifier per task — effectively a cross-check that a generated example is solvable by the *known* ground-truth program, which doubles as an automatic label-correctness check.

`main.py`'s `generate_dataset()` drives this at scale: for each task, keep sampling until `n_examples` de-duplicated, verified, non-degenerate examples are collected, tracking two difficulty metrics per example (RNG-difficulty: mean of sampled floats during generation; PSO-difficulty: a pixel/symbol/object-density heuristic) and per-task generation stats (runtime, verification rate). Output is one JSON file per task (`re_arc/tasks/<task_id>.json`, a plain list of `{'input':..., 'output':...}` dicts — trivially wrapped into ARC's standard `{"train": [...], "test": [...]}` shape) plus `re_arc/metadata.json`.

**This is directly usable for our RL/evolutionary agent as an effectively unlimited, difficulty-controllable curriculum**: instead of training against a fixed 400×(a few examples) dataset, we can sample new instances of the *same 400 task concepts* at chosen difficulty, which is exactly the kind of curriculum an RL agent needs to get enough training signal (sample efficiency is the core practical obstacle to RL on ARC's tiny native dataset). It generates variations of existing training tasks, not novel task concepts, so it does not help with generalizing to the held-out evaluation set's *novel* task types — it's a data-volume lever, not a generalization guarantee.

## 6. Comparison with `research/arc-ngps`'s scaffold

Read directly: `dsl/ast.py`, `dsl/types.py`, `dsl/parse.py`, `executor/grid_ops.py`, `executor/runtime.py`.

Our scaffold has:
- **6 AST node types** (`VarGrid`, `ConstColor`, `SelectColor`, `Paint`, `Translate`, `Compose`) vs. Hodel's 160 primitives.
- **3 executor primitives** (`select_color_objs` — 4-connectivity only, no `diagonal`/`without_bg` flags; `translate_objs`; `paint`) vs. Hodel's ~30 object/geometry/canvas primitives covering the same ground plus far more (rotations, mirrors, splits, canvas creation, cropping, line-drawing, periodicity, gravity/adjacency queries, etc.).
- **No shape-changing capability whatsoever** — `paint()` always writes into the pre-existing grid bounds; there is no `canvas`, `crop`, `concat`, or `upscale` equivalent. This is strictly less capable than what F2 needs.
- A `Compose` node explicitly marked in its own code comment as "just a placeholder... composition typically needs function-typed terms" — i.e. an acknowledged half-finished mechanism, not production-ready.
- The type system (`Ty` enum: `GRID, OBJSET, OBJ, INT, COLOR, BOOL, COORD`) is a reasonable subset of Hodel's richer type vocabulary but has no `Callable`/function type, which is why `Compose` had to be special-cased instead of being a normal higher-order primitive.

In short: `arc-ngps`'s DSL is an early, ~1-week-scale sketch of the same idea Hodel spent "several months" building out to full ARC-1 coverage (per his README framing). It is not a smaller-but-solid alternative; it is an unfinished subset.

## Recommendation

**Adopt Hodel's `arc-dsl` primitive catalog and typed values as the RL action-space foundation; discard `arc-ngps`'s DSL/executor (`ast.py`, `types.py`, `parse.py`, `grid_ops.py`, `runtime.py`) rather than adapting it.** Concretely:

1. Vendor `dsl.py`, `arc_types.py` (referenced by `dsl.py`, not yet fetched — fetch alongside), and `constants.py` under a clear `third_party/arc-dsl/` (MIT license, same treatment as `third_party/ARC-AGI/`).
2. Build the RL action space as: pick-a-primitive (categorical over the ~160, or a curated subset for milestone 1) + pick-its-typed-arguments, executed directly via Hodel's pure functions — no need for our own AST/executor layer at all; his functions already *are* the executor. This also sidesteps the unfinished `Compose`/function-typed-term problem: don't expose higher-order primitives (`compose`, `chain`, `fork`, `rbind`, `lbind`, `power`) as direct RL actions at all in milestone 1 — they build closures, not grid transforms, and add real complexity to a step-by-step action space for comparatively little payoff early on.
3. Use Hodel's 400 solver programs as ground truth for two things beyond curiosity: (a) validating the RL/evolutionary environment's executor against known-correct programs, and (b) potentially as expert-trajectory data for imitation-learning warm-starts or reward-shaping baselines later (not required for milestone 1, but a natural fast-follow).
4. Vendor `re-arc` too (`third_party/re-arc/`) as the training-data generator — this resolves the "RL needs way more signal than 400 fixed tasks" problem directly, and its difficulty knob is a ready-made curriculum axis.
5. **F2 resolution**: since `canvas`/`crop`/`concat`/`upscale` are first-class, well-exercised primitives in this DSL (used across many of the 400 real solvers), variable-output-shape is *not* the hard, novel problem I originally assumed when recommending same-shape-only scope — it reduces to giving the RL agent a fixed-size scratch canvas plus an explicit "commit/crop" action, a well-defined and boundable addition. Recommend revisiting F2: variable-shape support can likely be included in milestone 1 without the blast-radius I originally worried about, though a same-shape-first slice is still reasonable as the very first smoke test before adding the canvas/crop actions.
