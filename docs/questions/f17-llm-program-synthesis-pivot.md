# F17: pivot from curated-DSL RL/GP (+ object-grammar rewrite) to LLM-driven program synthesis

**Status:** DECIDED (user, 2026-09-22, via a `software-product-director`
skill review). Full rationale, the disqualifying facts the review found, and
the exact decision boundaries live in
[`docs/adr/0030-llm-program-synthesis-pivot.md`](../adr/0030-llm-program-synthesis-pivot.md).
This file tracks the decision's own history as it develops — same pattern as
F11–F16 — not a restatement of the ADR.

## Why this is a fork, not just an amendment to F11/F12/F15

The project's goal changed (2026-09-22): no longer personal
research/exploration, now (1) best achievable ARC-AGI performance via
actually state-of-the-art methods, and (2) a real, competitive ARC Prize 2026
entry, including the Paper Track. That goal change is what forced this
decision — the prior architecture wasn't merely improvable, it was
structurally disqualified from the entry vehicle the new goal requires (no
ARC-AGI-2/Kaggle path existed at all; see the ADR's Context for the three
specific disqualifying facts).

## The three prior forks this decision resolves or reverses

- **F12** (whether/how to pursue LLM-driven approaches) was `DEFERRED` — "not
  needed this milestone." This decision ends that deferral: LLM-driven
  program synthesis is now the project's primary method, not a
  demonstration-seeding side channel. F12's own history (the ADR-0018
  LLM-seeded-search experiment) stays on record as a smaller, earlier
  data point — it found LLM-seeded search "helped in every comparison run,"
  a mild positive signal in hindsight, though it was seeding the old curated
  action space, not synthesizing free-form Python.
- **F15** (GO on the object-centric representation + typed grammar rewrite)
  is superseded as an active track. It isn't wrong in retrospect — the POC's
  Gate 2 finding (search needs structure, not just more primitives) was real
  and well-evidenced — but V6's negative result (0/8 on its own target
  cluster, below parity elsewhere) closed the question of whether that
  structure was sufficient, and the user decided not to fund a further
  attempt at fixing its fitness-landscape problem.
- **Q2** (scope boundary) is reversed on three of its four original
  exclusions: ARC-AGI-2 is now in scope (vendoring is the first concrete
  roadmap step), Kaggle submission tooling is now in scope (required for the
  stated goal), and LLM-in-the-loop is now the primary method rather than
  excluded. Distributed/multi-GPU training stays out — moot under an
  inference-only, API-bound design, not reversed on its own merits.

## Deliberately out of scope for this decision

**Test-time training / fine-tuning.** Raised as an option during the review
(the dominant approach among the highest-scoring public ARC-AGI entries) and
explicitly declined by the user: "test time training is a bag of worms and i
want to stay away from that for now." Read as a deferral, not a permanent
rejection — if the inference-only propose→execute→diff→refine loop plateaus
well below a competitive score, TTT is the obvious next fork to reopen, not a
closed door. Whoever reopens it should treat it as a new, separate decision
with its own evidence, not something this ADR already settled either way.

**ARC-AGI-3.** Structurally a different kind of benchmark (interactive game
environments, not static input/output grid pairs) — a program-synthesis loop
built for ARC-AGI-1/2's task shape doesn't obviously transfer. Not addressed
by this decision at all; a future fork if pursued.

## Open items this decision hands to implementation (not yet resolved here)

- **A1 from the ADR** (plain Python/numpy vs. routing through vendored
  `arc-dsl` primitives): assumed default is plain Python, `arc-dsl` available
  but optional. Revisit once real synthesis attempts show whether unconstrained
  Python's search space is harder for the LLM to hit reliably than a narrower,
  pre-verified primitive library would be.
- **Provider/model choice for the LLM calls**, cost-per-task budget, and the
  attempt/token budget per task (ADR-0030's A2/A3) — none chosen yet; the
  first roadmap step (vendor ARC-AGI-2 + build the submission pipeline) will
  need at least a provisional answer before the baseline run and the first
  real synthesis attempts.
- **The frozen-stack ARC-AGI-2 public-eval baseline run** (ADR-0030's
  Consequences) hasn't happened yet — pending ARC-AGI-2 being vendored.
  Expected to score near zero, given ARC-AGI-2 is explicitly designed to
  defeat brute-force/DSL search, but that's an expectation to confirm with a
  real number, not skip.
- **F16's disposition** (the standalone editor / Play write path) is
  explicitly *not* resolved by this decision — ADR-0030 freezes the
  training-visualizer half of `viz/` but leaves the human-demonstration
  editor's fate to F16 itself, since it could still be useful (e.g. collecting
  few-shot human-solve demonstrations to seed LLM prompts) under the new
  architecture.
