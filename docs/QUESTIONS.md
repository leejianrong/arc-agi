# Questions — ARC-AGI RL/Evolutionary revamp

Statuses: `DECIDED` (user or delegated-research answered) · `ASSUMED` (default
taken, correct it if wrong) · `FORK` (waiting on the user) · `DEFERRED` (not
needed this milestone).

## Open forks

None — all forks from the 2026-08-27 round are resolved (see Register).

## Register

| ID | Question | Status | Answer or default | Landed |
|----|----------|--------|--------------------|--------|
| F1 | DSL / action-space foundation | DECIDED (delegated research: `docs/research/arc-dsl-survey.md`) | Adopt Hodel's `arc-dsl` (160 primitives, MIT) as the action space/executor; discard `research/arc-ngps`'s DSL/executor; vendor `re-arc` for training-data generation | ADR-0001 |
| F2 | Variable output-grid shape handling / milestone-1 task scope | DECIDED (delegated research, reverses original same-shape-only recommendation) | Fixed 30×30 scratch canvas + explicit commit/crop action, mirroring `arc-dsl`'s own `canvas`/`crop` primitives; Slice 1 still starts same-shape-only as a smoke test, canvas/commit added in Slice 3 | ADR-0002, SLICES.md V1 & V3 |
| F3 | Build order: RL vs. evolutionary vs. both | DECIDED (user) | RL (PPO) first; genetic programming over the same DSL as the evolutionary fast-follow; both share one env/DSL/trajectory-log format | ADR-0003 |
| F4 | RL framework: Stable-Baselines3 vs. custom loop | DECIDED (user) | Gymnasium-style env + hand-rolled PPO (no SB3) | ADR-0004 |
| F5 | Visualizer live-streaming vs. log-and-replay | DECIDED (user) | Log-and-replay via JSONL/CSV on local disk, no trainer↔visualizer IPC | ADR-0006 |
| F6 | Visualizer tech stack | DECIDED (user) | Custom local web app; TypeScript (not JavaScript) + Canvas frontend | ADR-0007 |
| F7 | Reward shaping design | DECIDED (delegated research: `docs/research/rl-evolutionary-survey.md`) | Dense, delta-based, non-background-normalized similarity reward + no-op step penalty + terminal exact-match bonus | ADR-0005 |
| F8 | Evolutionary method: genetic programming vs. neuroevolution | DECIDED (delegated research) | Genetic programming over the DSL (CPU-cheap, zero prior benchmark to beat); neuroevolution documented as a future option only | ADR-0003 |
| F9 | PPO training lifecycle (per-task vs. shared/cross-task) and whether a pretrained autoencoder/VAE embedding is needed | DECIDED (user, via conversation 2026-08-27) | Per-task, solve-time training — one fresh PPO policy per `task_id`, trained only on that task's own train pairs + `re-arc` variations, never shared across tasks. No representation-pretraining this milestone: the grid encoder (color-embedding + conv/attention) is learned end-to-end from the PPO reward signal; a reconstruction-based AE/VAE would target a different problem (general ARC perception) and is a weak proxy for the relational structure ARC needs. Cross-task generalization + shared pretraining documented as an explicit future direction, not this milestone | ADR-0008 |
| F10 | GP-to-PPO behavior-cloning warm-start: always-on pipeline vs. opt-in flag vs. continuous auxiliary loss | DECIDED (user, via conversation 2026-08-29) | Opt-in only: `train.py --algo ppo --warm_start_from <gp_run_dir>` loads a same-task GP run's best-program trajectory and runs a one-time supervised pretrain phase before normal PPO proceeds unmodified. Design decision only — implementation not yet built | ADR-0009 |
| F11 | Task-coverage scaling past the curated 16: broaden primitives vs. add object selection vs. hold at 16 vs. deepen via re-arc | DECIDED (user, via conversation 2026-08-29) | Both broadening curated primitives and object selection, sequenced: Phase 1 (near-term) extends the curated action space with more scalar-arg-only arc-dsl primitives, no new mechanism, ceiling ~61/400 tasks per a 2026-08-29 repo audit; Phase 2 (fast-follow) adds an object-selection mechanism to reach the larger 79/400 object-manipulation bucket. Phase 1 landed 2026-08-29: 4 new self-concatenation actions, 16→24 curated tasks. Phase 2 Slice 1 landed 2026-08-31: a "currently selected patch" side-channel (`select_largest`/`select_smallest`/`commit_selection`), 24→26 curated tasks. Phase 2's deferred full menu landed 2026-09-04 (ADR-0012): `select_by_color`/`select_unique_color`, `delete_selected`/`recolor_selected`/`move_selected`/`paint_selected_at`, 26→29 curated tasks. ADR-0013 (2026-09-05) added two more `objects(...)` connectivity variants (`select_largest_no_diag`, `select_tallest`), 29→30 curated tasks - `select_largest_no_diag` has a verified fixture (`be94b721`), but `select_tallest`'s fixture (`1c786137`) needs a `trim` step after `commit_selection` already ends the episode, so it stays out of reach without a new fused act-on-selection primitive (a well-scoped follow-up, not landed this pass). **2026-09-05 refresh, user asked to make this more urgent and push toward 400/400 training tasks**: re-ran the coverage audit against the current (post-ADR-0013) action space and all 370 still-uncurated tasks (`scripts/audit_action_coverage.py`, committed this pass) - 269/370 (73%) call a higher-order combinator (`mapply`/`compose`/`fork`/`apply`/`lbind`/`rbind`/`chain`/`sfilter`/... - structurally excluded per ADR-0001, unchanged), 10 are reachable today by primitive name alone (re-check candidates - name overlap doesn't guarantee the actual parameterization/composition is curated, see ADR-0013's `select_tallest`/`1c786137` caveat), 21 need exactly one new primitive by name (top: `ofcolor` unlocks 5, `first` unlocks 5, `sizefilter` unlocks 4), 18 need exactly two, and 52 need three or more. Revised ceiling: **400/400 is very unlikely under this architecture** - the 73% figure is *higher* than the 2026-08-29 audit's 65%, meaning growth so far has disproportionately drained the "neither" bucket, leaving a remaining pool skewed even harder toward structurally-excluded tasks. Continuing incrementally (highest-leverage near-miss clusters first, e.g. `ofcolor`+`first` next) remains the right near-term move, but reaching all 400 training tasks - let alone the harder, solver-free evaluation set - would need a fundamentally different representation (see F12). **ADR-0015 (2026-09-06) is that `ofcolor`+`first` pass, with a mixed result worth recording precisely**: `first`'s 5 flagged tasks turned out to share one root cause - the same `commit_selection`-ends-the-episode-too-early wall ADR-0013 already hit for `1c786137` - fixed generically with a new non-terminal `crop_to_selection` act-on-selection action (identical crop, different action name, so the termination check by literal name doesn't fire) plus one new `objects(...)` connectivity variant (`select_largest_multicolor`, `univalued=False`); this landed 6 tasks at once (`1c786137` plus 5 more: `a79310a0` needed **zero** new primitives - the audit's name-level check missed that its solver already maps onto curated actions - `28bf18c6`, `f25fbde4`, `2013d3e2`, `7468f01a` needed the new mechanism), taking curated tasks 30 → 36. `ofcolor`'s 5 flagged tasks, by contrast, each need a *different* additional capability the audit's name-level check couldn't see (two tasks need multiple simultaneous selections across different coordinate spaces or grids; two need a selection reused across several "stamp a shifted copy" operations, which needs a new directional act-on-selection primitive that doesn't exist yet; one needs a selection checked against an unrelated grid) - none is a bare "add one primitive" fix, so none was landed this pass. Next concrete candidate named in ADR-0015: a directional "stamp" act-on-selection primitive, which would unlock 2 of those 5 (`a9f96cdd`, `d364b489`) on its own. **ADR-0016 (2026-09-06) lands that candidate**: `stamp_selected` (`dsl.fill(grid, color, dsl.shift(selected, DIRECTION))`) plus a new, separate 8-direction menu (4 cardinal + 4 diagonal - `a9f96cdd` needs diagonals `move_selected`'s existing cardinal-only menu doesn't have, so a fresh menu was added rather than widening that one and risking its own `25ff71a9` fixture) - unlocks `a9f96cdd` and `d364b489` exactly as scoped, both verified by direct replay against every train/test pair, taking curated tasks 36 → 38. The remaining 3 `ofcolor`-flagged tasks (`7c008303`, `9ecd008a`, `a68b268e`) still need their own separate capabilities (multi-selection across coordinate spaces/grids, or a selection checked against an unrelated grid) not addressed by this pass. **ADR-0019 (2026-09-06) lands two more one-new-primitive tasks from the same 2026-09-05 audit refresh's 21-task "needs exactly one new primitive" bucket**: `5582e5ca` (`mostcolor`) and `aabf363d` (`leastcolor`), landed as two more derived actions (`canvas_mostcolor` - `canvas` fed a grid-derived color instead of an agent-chosen one; `swap_two_least_colors` - a fully self-contained zero-arg composition of `leastcolor`/`replace`), no new mechanism - taking curated tasks 38 → 40 (19 same-shape + 21 variable-shape; both new tasks turned out to be same-shape on computational re-check, correcting an initial "almost certainly variable-shape" guess for `5582e5ca`). **2026-09-06 follow-up pass: independent re-verification of the remaining 3 `ofcolor`-flagged tasks (`7c008303`, `9ecd008a`, `a68b268e`) plus an explicit go/no-go call on `9ecd008a`, prompted by F11's own coverage audit still flagging all 3 as open.** All 3 known-correct solvers were independently replayed (bare `dsl` calls, not `actions.execute`, since none is curated) against every train+test pair - all reproduce the expected output exactly, confirming ADR-0015's characterization rather than just restating it: `7c008303` computes one `Indices` set (`ofcolor(I,3)` → `subgrid`) then a *second*, unrelated `Indices` set in that sub-grid's own coordinate space (`ofcolor` on the crop), and fills a *third* grid built via a wholly different pipeline (`replace`×2 → `compress` → `upscale(3)`) - confirmed computationally that the crop's shape (6×6 in the checked fixture) and the compress+upscale result's shape (also 6×6) only coincide because of this task's specific geometry, not any general relationship; `a68b268e` computes three separate `ofcolor` calls against three different quadrant sub-grids (`lefthalf(tophalf(I))`, `righthalf(tophalf(I))`, `lefthalf(bottomhalf(I))`, each independently confirmed 4×4 in the fixture) and fills their indices onto a fourth quadrant - a genuine simultaneous-multi-selection-across-grids requirement. Both reconfirmed as needing the multi-selection/cross-grid-index mechanism the original scope boundary named (PPO's factored action head, GP's flat gene list, episode logging, and the visualizer's selection overlay would all need to represent more than one live named selection) - a redesign bigger than anything ADR-0011 through ADR-0019 made, correctly out of scope for an incremental pass and not attempted. `9ecd008a` (`vmirror(I)` computed once; `ofcolor(I,0)` computed against the *original*, pre-mirror `I`; `subgrid` crops the *mirrored* grid using those stale, pre-transform indices) was traced further this pass: `ofcolor(I,0)` selects a compact 9-cell/3×3 hole (color 0 is the least-common color in the checked fixture, occupying a tight bounding box) in an otherwise-dense 16×16 grid, and the output is exactly that 3×3 patch - i.e. this is the classic ARC "symmetry repair" motif (a masked region's true content is recovered from the grid's own mirror-symmetric counterpart), not an arbitrary coincidence. That makes the *task* well-motivated, but the mechanism it needs is still the specific one ADR-0015 flagged as tension with ADR-0011: a selection computed before a transform, deliberately kept alive and reused *against the post-transform grid*. Explicit go/no-go: **no-go, not implemented.** The only way to reach this inside the current one-selection-slot model is an `act_on_selection`-kind action that performs an ordinary whole-grid transform (e.g. `vmirror`) while requiring - but never actually using - the current selection, purely to exploit `execute()`'s existing rule that only the generic/`"transform"` kind clears the selection (`arc_env/actions.py`, ADR-0011). Unlike every actual `act_on_selection` action shipped so far (`crop_to_selection`, `delete_selected`, `stamp_selected`, ...), which all *consume* `selected` as a real input to their computation, this hypothetical action's function body would ignore `selected` entirely - a selection would be a precondition for no reason connected to what the action does, a shape no existing action has and a confusing precedent to set. More importantly, landing it as a general menu entry (not a fixture-only special case) would put a "transform the whole grid, keep whatever was selected before, no matter how the transform changed the grid" action into the same policy/genome action space every other task's search draws from - reviving, generally and permanently, exactly the "silently wrong stale selection surviving an unrelated edit" failure mode ADR-0011 was written to rule out, on the strength of one task where the staleness happens to be exactly correct. Conclusion: this is a single-task special case dressed up as a general mechanism, not a generally useful addition - consistent with ADR-0013's `select_tallest`/`1c786137` precedent and ADR-0015's own `ofcolor`-cluster precedent, both cases where a well-reasoned "don't land this" was the right call. All 3 tasks remain uncurated; no code changed this pass | ADR-0010, ADR-0011, ADR-0012, ADR-0013, ADR-0015, ADR-0016, ADR-0019 |
| F12 | Whether/how to pursue LLM-driven approaches (program synthesis, hypothesis-testing, on-the-fly reward/primitive design) as a track toward higher ARC-AGI-1 coverage and, eventually, ARC-AGI-2/3 | DEFERRED (not needed this milestone; revisits Q2's "no LLM-in-the-loop" scope boundary for a later one) | User's 2026-09-05 framing, recorded for later: (1) an LLM designs reward functions or primitives on the fly; (2) an LLM writes arbitrary Python/DSL-composition scripts per task, verified against train pairs; (3) an LLM models a single task's rules explicitly and iterates via an exploring agent that tests hypotheses. Assessment: all three are a genuinely different architecture family from this project's GP/PPO-over-curated-flat-DSL substrate, not an increment on it - in particular, (2)/(3) sidestep the exact restriction (no `Callable`/higher-order primitives, ADR-0001) that F11's 2026-09-05 refresh found blocks 73% of ARC-AGI-1's remaining uncurated tasks, and ARC-AGI-2 was deliberately designed to defeat compositional search over a small fixed DSL menu (the current architecture's whole category), so (2)/(3) look like the more credible path to either goal than more curated-DSL growth. (1) is valid but in tension with ADR-0005's one-reward-shared-by-both-trainers invariant, which is this project's own comparative-research value - if pursued, treat as its own experiment track with its own comparability story, not a bolt-on to the existing GP/PPO harness. Recommend keeping this a clearly separate initiative (its own docs/ADRs, not folded into F11's incremental-coverage loop) if/when picked up. **Refinement (2026-09-06): "LLM-seeded search, not LLM-designed reward."** A narrower, lower-risk cut of option (2) that doesn't touch ADR-0005's shared-reward invariant at all: an LLM proposes a candidate *action sequence drawn from the existing curated DSL* (`arc_env.actions`, never arbitrary code) for a given task, and that sequence feeds GP's initial population and/or PPO's ADR-0009 warm-start as a *second demonstration source* alongside (not instead of) GP's own search - applied symmetrically to both trainers, so the GP-vs-PPO comparison this project is built around stays intact. Small experiment shape: for a handful of curated tasks GP/PPO currently struggle with (or a few from F11's near-miss buckets), have an LLM propose a `CURATED_TASK_IDS`-style action sequence from the task's train pairs + `arc_env.actions`'s menu, verify it against train/test pairs the same way `tests/test_dsl_regression.py` does, and if valid, seed it into GP's population / PPO's warm-start and compare against the no-seed baseline. Not yet scoped as a slice or started. **2026-09-06 pass (ADR-0018,
`docs/research/llm-seeded-search-experiment.md`)**: built the two code
mechanisms this refinement needs (`trainers/gp/evolve.py::run_gp(...,
seed_programs=...)`, seeding generation 0's population rather than pure
random init; `train.py`'s `check_warm_start_compatible` widened from a
hardcoded `algo == "gp"` check to an explicit `WARM_START_COMPATIBLE_ALGOS
= {"gp", "llm-seed"}` allowlist) plus a harness (`scripts/
llm_seed_search.py`: `describe_task`/`verify`/`sequence_to_program`/
`write_seed_episode`), then ran the actual small experiment on 3 curated
tasks GP/PPO currently struggle with: `ea32f347`, `5bd6f4ac` (GP's only two
standard-budget zero-success tasks), `5614dbcf` (GP solves, PPO doesn't,
and its existing GP-based warm-start was separately documented as unstable
- see `docs/PLAN.md`'s KAN-1239 section). Reasoned through each task's
train pairs directly (pixel-level diffs) and proposed a `CURATED_TASK_IDS`-
style sequence for each; all 3 verified against every train+test pair and,
notably, all 3 matched `CURATED_TASK_IDS`'s existing entry exactly (with an
honesty caveat recorded in the research doc: the same pass's code-change
work required reading `task_loader.py`, so this isn't a fully blinded
derivation). **GP-seeding result (standard budget, 3 seeds each)**:
unseeded 0/6 exact matches across both tasks (reproducing the
already-documented standard-budget failure), seeded 6/6, every seeded run
solving at generation 0 - strictly better than `docs/PLAN.md`'s
previously-tried 25x-larger-budget mitigation (which itself only reached
2/3 seeds on `ea32f347`, 0/3 on `5bd6f4ac`), though this is the
least-surprising possible outcome of seeding with an already-perfect
demonstration, not a test of noisier/partial seeds. **PPO warm-start
result on `5614dbcf` (single seed, 3 arms)**: plain PPO never reaches
`eval_success=True` at any checkpoint (ends `eval_reward=0.12`); both a
fresh GP-warm-start and the LLM-seed warm-start *do* reach
`eval_success=True` and hold it through the final checkpoint - the GP arm
immediately (its randomly-found demonstration this run happened to have a
low 0.34 behavior-cloning loss), the LLM-seed arm only from update 20
onward (its cleaner 2-step demonstration had a harder-to-imitate 1.10 bc
loss, closer to KAN-1239's own reported 1.21 for this task). Read plainly:
the LLM-seed mechanism is a viable *alternative* demonstration source that
also turns this task from a plain-PPO failure into a stable final-checkpoint
solve, not proven better or worse than GP's own demonstration from this one
run alone (GP's demonstration difficulty varies run-to-run; a proper
comparison needs multiple GP seeds, an explicit, acknowledged limitation of
this single-seed-per-arm pass). All new/existing tests pass (`uv run pytest
-m "not slow" -q` and the full `uv run pytest`, `uv run ruff check .`) -
see the research doc for exact counts. Overall: seeding helped in every
comparison run this pass, with the PPO side's caveat stated above rather
than oversold | ADR-0018, `docs/research/llm-seeded-search-experiment.md` |
| F13 | Whether to build an interactive "Photoshop-like" editor - a human solves ARC-AGI-1 tasks by hand with a constrained toolset mirroring the curated action space, logged as a demonstration trajectory for warm-start/imitation learning | DECIDED (user, via conversation 2026-09-06) - staged; **Stage 0 built 2026-09-06 (ADR-0017); Stages 1/2 still not started** | Assessed as viable, staged to de-risk before committing: **Stage 0** (cheap) - a minimal UI over the *existing* curated actions (ADR-0016: 41 as of this pass) + a live `ArcEnv` instance, logging through the unmodified `EpisodeWriter` schema (`arc_env/episode_log.py`) so human solves are automatically warm-start-compatible (ADR-0009), no schema change. This needs a real write path into `viz/backend/server.py`, previously read-only by design (ADR-0006/ADR-0007) - a deliberate, explicit architectural reversal, not an incidental one, scoped as its own small addition (`viz/backend/play.py` plus `/api/actions` + `/api/play/*` routes driving a per-session `ArcEnv`), not a rewrite of the read-only run-browsing path the rest of the visualizer depends on. **Landed 2026-09-06 (ADR-0017)**: `viz/backend/play.py` (in-memory `session_id -> PlaySession` store, guarded by a `threading.Lock()`, no persistence until an explicit save - accepted local-single-user tradeoff), path-traversal validation on both `task_id` and (client-supplied) `run_id` before any file access, and any of the 400 training tasks playable (not restricted to the curated subset, per Stage 1's own plan below); a new `viz/frontend/src/play.ts` panel wired into the existing single-page app, with its action picker built generically from `GET /api/actions` rather than a hardcoded menu. **Stage 1** - run Stage 0 on the excluded-269 tasks from F11's coverage audit to see what tools a human naturally reaches for, as direct design input for new ADRs (a cheap way to *observe* which mechanism gaps matter most, complementing F11's solver-text-audit approach with actual usage data). **Stage 2** - turn whatever Stage 1 reveals (layers, multi-select, relational predicates) into real ADR'd mechanisms, same review bar as ADR-0011/0012/0013/0015/0016. Stages 1/2 remain explicitly contingent on Stage 0's own findings, not pre-committed | ADR-0017 |
| Q1 | Primary user and actors | ASSUMED | Solo user (repo owner), personal research; no multi-actor conflicts | PLAN.md Users and actors |
| Q2 | Scope boundary | ASSUMED + F2 | ARC-AGI-1 only; no Kaggle/private test; no distributed training; no LLM-in-the-loop; see full in/out list | PLAN.md Scope |
| Q3 | Core data model and identity | ASSUMED | `task_id` (ARC filename stem) + timestamped `run_id`; JSONL trajectory schema per step | PLAN.md Implementation decisions, ADR-0006 |
| Q4 | State and storage | ASSUMED | Local disk only, `runs/<run_id>/`, no database | ADR-0006 |
| Q5 | Concurrency and conflict | ASSUMED | No locking needed; independent `run_id` directories, no shared mutable state | PLAN.md Assumed defaults |
| Q6 | Interfaces and contracts (CLI) | ASSUMED | `train.py --algo ppo\|gp --task_id <id> [hyperparameter flags]` (flat per-hyperparameter flags, no single `--config`) | PLAN.md Affordances |
| Q7 | Failure behaviour | ASSUMED | Invalid actions → no-op + small penalty; hard max-step termination; checkpoint-based recovery | PLAN.md Assumed defaults |
| Q8 | External dependencies (license, offline) | ASSUMED + F1 | `arc-dsl`/`re-arc` MIT; `gymnasium` MIT; numpy/torch (BSD-style), CPU-only wheel installed explicitly (avoids repeating `arc-ngps`'s 6.9GB CUDA-wheel mistake); all offline/local | ADR-0001, PLAN.md Implementation decisions |
| Q9 | Runtime and deployment | ASSUMED | Single local machine, CPU-only (confirmed: no CUDA, 16 cores), `localhost`-only visualizer | PLAN.md Assumed defaults |
| Q10 | Measurable success | ASSUMED + research-calibrated | Task-exact-match accuracy on scoped subset; calibrated against ~20% brute-force-search floor (icecuber) as context, not a target | PLAN.md Assumed defaults, `docs/research/rl-evolutionary-survey.md` |
| Q11 | Security and secrets | ASSUMED | None applicable — no credentials/PII, public dataset only | PLAN.md Assumed defaults |
| Q12 | Versioning and migration | ASSUMED | `schema_version` field in `run_meta.json` | PLAN.md Assumed defaults |

## Coverage

| Category | Covered by |
|----------|-----------|
| Primary user and actors | Q1 |
| Scope boundary | Q2, F2 |
| Data model and identity | Q3 |
| State and storage | Q4 |
| Concurrency and conflict | Q5 |
| Interfaces and contracts | Q6, F5, F6 |
| Failure behaviour | Q7 |
| External dependencies | Q8, F1, F8 |
| Runtime and deployment | Q9 |
| Measurable success | Q10 |
| Security and secrets | Q11 |
| Versioning and migration | Q12 |
| Reward design (domain-specific) | F7 |
| Build order / paradigm sequencing (domain-specific) | F3, F4 |
| Policy architecture / training lifecycle (domain-specific) | F9 |
| RL/GP interoperation (domain-specific) | F10 |
| Task-coverage scope past the curated 16 (domain-specific) | F11 |
| Cross-benchmark generalization strategy past the curated-DSL approach (domain-specific) | F12 |
| Human-in-the-loop demonstration collection (domain-specific) | F13 |

## Repo-audit facts used throughout

- No GPU: `torch` not installed at root env at time of audit; no `nvidia-smi`;
  16 CPU cores available. Confirmed directly, not assumed.
- `third_party/ARC-AGI/` is the official ARC-AGI-1 clone (de-gitted, plain
  tracked files): `data/training`, `data/evaluation`, plus the human-facing
  `apps/testing_interface.html`. No private/Kaggle test set present or
  obtainable.
- `research/arc-ngps/` was a half-built supervised program-synthesis scaffold;
  superseded per ADR-0001, not deleted, disposition deferred.
- `legacy/` holds the original non-learned geometric-transform baseline;
  kept as a reference, not deleted.
- 2026-08-29 solver-corpus audit (parsing `third_party/arc-dsl/solvers.py`'s 400
  `solve_<task_id>` bodies for calls to higher-order-combinator vs.
  object-manipulation function names): 260/400 (65%) need a higher-order combinator
  (excluded per ADR-0001), 79/400 (20%) need object selection/manipulation (no
  representation yet), 61/400 (15%) need neither — of those 61, only 16 are
  currently curated (`arc_env/task_loader.py`'s `CURATED_TASK_IDS`). Basis for
  ADR-0010's task-coverage-scaling decision (F11).
- 2026-09-05 refresh (`scripts/audit_action_coverage.py`, checks `dsl.py` type
  hints for a `Callable`-typed parameter rather than a fixed name list, so it
  stays accurate as new higher-order primitives are/aren't added): of the 370
  tasks still uncurated after ADR-0013, 269 (73%) are structurally excluded by
  a higher-order primitive, 10 look reachable by primitive name alone (not
  verified — name overlap isn't proof of matching parameterization, see
  ADR-0013's `select_tallest`/`1c786137` caveat), 21 need exactly one new
  primitive (top clusters: `ofcolor` unlocks 5 tasks, `first` unlocks 5,
  `sizefilter` unlocks 4), 18 need exactly two, 52 need three or more. Basis
  for F11's 2026-09-05 refresh and F12.
