# MCTS / AlphaZero-style search survey for ARC-AGI-1

Scope: whether Monte Carlo Tree Search guided by a learned value/policy network
(AlphaZero-style self-play) is a fit for this repo's per-task, solve-time
training model (ADR-0008) and curated DSL action space (ADR-0001/ADR-0002).
Findings from arXiv, the ARC Prize technical reports/blog, and GitHub prior
art, gathered 2026-08-29. Companion document to
`docs/research/rl-evolutionary-survey.md`, which this doesn't repeat —
read that first for the PPO/GP baseline this is being compared against.

## 1. Prior art — has anyone applied MCTS/AlphaZero to ARC-AGI, or a close analog?

**No ARC-AGI-specific result surfaced anywhere in this search.** Neither the
[ARC Prize 2024 Technical Report](https://arxiv.org/pdf/2412.04604) nor the
[ARC Prize 2025 Technical Report](https://arxiv.org/abs/2601.10904) mentions
MCTS or AlphaZero-style tree search as an attempted or competitive approach
for ARC-AGI-1. Both describe the same lineage as the existing
`rl-evolutionary-survey.md`: brute-force/greedy DSL search (icecuber, ~20%),
then LLM-driven program synthesis and, in 2025, "the refinement loop" — a
per-task iterative program-optimization loop guided by feedback, which
includes evolutionary program synthesis but not tree search with a learned
value network. Other named symbolic ARC-AGI-1 solvers checked here —
[ARGA](https://arxiv.org/abs/2210.09880) (Xu, Khalil & Sanner, AAAI 2023;
graph-abstraction DSL + constraint acquisition + **Tabu search**, not MCTS)
and Hodel's own greedy/stochastic DSL search behind `arc-dsl`'s solver
corpus — both use non-MCTS discrete search. Say this plainly, the way the
sibling survey does for neuroevolution: **this is an empty cell in the
literature, not a case where MCTS was tried and failed.**

The closest thing to "MCTS + ARC" that turned up is
[Executable World Models for ARC-AGI-3 in the Era of Coding Agents](https://arxiv.org/abs/2605.05138)
(2026) — but ARC-AGI-3 is a different benchmark (interactive, multi-step
*games*, not this repo's static input→output grid pairs), and the method
isn't AlphaZero-style self-play either: an LLM coding agent builds and
verifies an executable Python "world model" of the game, then plans through
it. There's a tree-search-shaped "plan before acting" step, but no learned
value/policy network trained via self-play — it's closer to model-based
planning with an LLM-authored simulator than to AlphaZero. Not a
transferable precedent for this repo's per-task DSL-search setting either
way.

What does exist, and is the right reference class to reason from, is
**MCTS applied to DSL-based program synthesis in general** (not ARC-specific):
- [Improved Tree Search for Automatic Program Synthesis](https://arxiv.org/abs/2303.07166)
  (Carmon & Wolf, 2023) proposes an MCTS variant (modified visit counts,
  dataset preprocessing, encoding the partially-executed program) for
  synthesizing DSL programs from input/output examples, reporting
  state-of-the-art results on two DSLs versus a prior MCTS baseline (CAB).
  This is the general-purpose analog of what an "MCTS for ARC" system would
  need to look like — search over program *construction* choices, not over
  grid-cell edits directly.
- [Program Synthesis Through Reinforcement Learning Guided Tree Search](https://arxiv.org/abs/1806.02932)
  and the [Field Report on Applying MCTS for Program Synthesis](https://coinse.kaist.ac.kr/projects/mctsps/)
  (Lim & Yoo, SSBSE 2016) are earlier entries in the same lineage — MCTS
  and RL-guided tree search over program space are established techniques
  for program synthesis broadly, just never pointed at ARC-AGI specifically.
- [DreamCoder](https://arxiv.org/pdf/2310.04327)-adjacent bottom-up program
  synthesis (BUSTLE, LambdaBeam, Eco Search) generally uses learned cost
  functions to guide best-first or bottom-up enumeration rather than MCTS —
  a different search paradigm again, worth noting only because it shows the
  broader program-synthesis field mostly reaches for enumerative-plus-
  learned-heuristic search, not tree search with rollouts, when the
  simulator (executing a candidate program) is this cheap. That's a
  meaningful signal for §3 below.

## 2. What would a value/policy network need to represent here, and does per-task training give it enough signal?

AlphaZero's value proposition rests on **amortization**: one network is
trained across millions of self-play games so it generalizes across
positions it has never exactly seen, and the network gets *better* precisely
because the corpus of games it learns from is enormous and diverse. ADR-0008
is the structural opposite of that setup by design: "one fresh PPO policy
per `task_id`, trained only on that task's own train pairs plus `re-arc`
variations... never shared across tasks." A value network here would have to
be trained from scratch, per task, on a training set of a handful of grid
pairs (a task's ~2-5 train examples plus however many `re-arc` variations of
the *same underlying transformation* are generated) — not millions of
diverse self-play trajectories, but many near-duplicate views of one fixed
concept.

Concretely, what would the network need to represent? Given this repo's
actual action space (ADR-0001's ~23 curated primitives plus `canvas`/`commit`,
ADR-0002), a value function at a tree-search node would need to answer
"how close is this partially-edited scratch-canvas state to solving *this
one task's* transformation" — which is exactly what ADR-0005's
`similarity()` function in `arc_env/reward.py` already computes,
non-learned, in closed form (delta pixel-match against the task's own
diff mask, plus the 2026-08-29 shape-distance gradient for variable-shape
`commit` targets). A learned value network would be trying to approximate a
function this project already has exactly and for free. The only thing a
learned value net could add over the closed-form `similarity()` is
generalizing to states *unlike* anything in the small train set — but with
2-5 train pairs (plus near-duplicate `re-arc` instances of the same
concept, not novel concepts), there is essentially no distributional breadth
to generalize *from*. This is the single biggest reason AlphaZero's core
mechanism doesn't transfer: **its whole bet is that scale turns a value net
into a better heuristic than anything hand-designed; at N≈dozens of training
instances per task, there's no scale to bet on, and the exact
same-task-restricted heuristic already exists as an interpretable formula.**
A learned policy prior (which action to try first) is a more plausible thing
to gain from training — but that's exactly what PPO's policy head or GP's
mutation/crossover operators already are, without needing tree search or a
value net alongside them.

## 3. Does MCTS's core advantage — needing far fewer rollouts than brute enumeration — matter when GP's fitness eval is already near-instant?

MCTS earns its keep when the simulator is expensive relative to network
inference (e.g. a chess/Go rollout, or a physics simulation) — you want to
spend an evaluation budget on the *most promising* branches because you
can't afford to try everything. This repo's situation is close to the
opposite. `trainers/gp/fitness.py`'s `evaluate_fitness` runs a candidate
program (≤6 genes, per `GPConfig.max_program_length`) against a task's train
pairs by direct DSL execution (`actions.execute`) — no forward pass, no
simulation, just interpreting a short list of typed function calls against
grids ≤30×30. `trainers/gp/evolve.py`'s default config evaluates 100
candidates/generation for up to 50 generations, i.e. up to 5,000 full
program evaluations per run, each one microseconds-to-low-milliseconds. On
any task GP can solve at all, this reaches a perfect-fitness program in a
small fraction of that budget (the loop breaks the moment `best_fitness[0]
>= 1.0`). There is no "too expensive to enumerate" problem here for the
kind of short, near-compositional programs (≤6 primitives) this repo's
action space and task subset are scoped around — which matches the general
program-synthesis literature's own revealed preference in §1: when the
executor is this cheap, the field mostly reaches for enumerative/best-first
search with a learned cost heuristic (BUSTLE, Eco Search), not MCTS with
rollout simulation, because there's no expensive-simulator problem to solve.

Where MCTS's simulation budget *would* actually go, if built anyway, is
**deeper, non-compositional or hard-to-reach states** — tasks needing a
program longer than GP's `max_program_length` ceiling, or where the fitness
landscape is deceptive enough that tournament selection/mutation gets stuck
in a local optimum an exploration-weighted tree search might escape. That's
a real, narrow niche — but it's a niche about search-*depth* and
deceptive-landscape robustness, not about MCTS's headline advantage (cheap
proxy for an expensive simulator), which doesn't apply here at all. Any
prototype would need to target that niche specifically rather than being
framed as a general replacement for GP.

## 4. Does self-play make sense in a single-agent, no-adversary setting?

ARC-AGI grid-editing has no two-player structure: there's one agent editing
a grid against a fixed, known-in-advance target (the train pair's output).
AlphaZero's self-play is a bootstrapping trick for domains **without ground
truth** — the network's own past version stands in for an opponent so a win/
loss signal exists to learn from at all. This repo doesn't have that
problem: `arc_env/reward.py`'s `similarity()`/`compute_reward()` already
*is* ground truth, computed directly from the train pair, every step. There
is nothing for a self-play opponent to bootstrap that isn't already given
for free.

The literature does have a real non-adversarial analog worth naming
precisely so as not to build a straw man: **single-player MCTS + value/policy
learning without an opponent**, exemplified by
[Solving the Rubik's Cube Without Human Knowledge](https://arxiv.org/abs/1805.07470)
(McAleer et al., 2018, "Autodidactic Iteration"/DeepCube) and
[Single-Agent Optimization Through Policy Iteration Using Monte-Carlo Tree Search](https://arxiv.org/abs/2005.11335).
Autodidactic Iteration's trick for generating a value-network training
signal *without* an adversary is to sample backward from the **known goal
state** (start solved, scramble outward, so every sampled state has a
computable ground-truth distance-to-goal via lookahead), then train the
value/policy net on those samples and refine with MCTS — genuinely the
correct analog for "self-play in a single-player setting," not a category
error in the abstract. But look at what it cost to make that work: per the
paper, training used **~2,000,000 ADI iterations, ~8 billion cube states
(including repeats), over 44 hours on a 32-core server with three Titan XP
GPUs** — to solve *one general combinatorial puzzle* (any scrambled cube),
not one fixed instance. That scale is the entire mechanism by which
backward-from-goal sampling substitutes for adversarial self-play: it works
*because* it can generate an effectively unlimited, diverse corpus of
graded-difficulty training states from the one goal state. ADR-0008's
setting inverts every one of those preconditions: a fixed task-specific
target (fine, matches ADI's "goal state" idea) but a tiny, non-diverse
instance corpus (`re-arc` variations of one task's concept, not "any
scramble of the same underlying puzzle" the way Rubik's Cube states are
homogeneous), a CPU-only 16-core box with no GPU, and a per-task budget
nowhere near 8 billion samples. **The mechanism that makes non-adversarial
"self-play" work in the literature is available in principle, but its
enabling condition — cheap, unlimited, structurally homogeneous sample
generation from a known goal — is exactly what a single ARC task's few
train pairs plus a handful of `re-arc` variations don't provide.** This is
a genuine, not superficial, mismatch: not "there's no two-player game here"
(true but a strawman version of the objection), but "the specific technique
that gets around needing a two-player game needs a scale of self-generated
data this project's per-task setting structurally cannot produce."

A curriculum-via-`re-arc`-difficulty idea (using generated instances of
increasing difficulty as a training curriculum) is the one part of "self-play"
spirit that could transfer without the scale requirement — but that's a
curriculum-ordering idea for PPO/GP's existing training loop, not an
argument for adding MCTS or a value network; nothing about it requires tree
search.

## 5. Compute cost: MCTS's rollout budget vs. PPO's per-update cost vs. GP's near-free fitness evaluations

On the "how expensive would this actually be" question, the numbers are
lopsided. [How much did AlphaGo Zero cost?](https://www.yuzeh.com/data/agz-cost.html)
estimates the original AlphaGo Zero self-play/training run at roughly
**$3M in TPU compute** at 2018 pricing (~6,380 TPUs across self-play
machines) — the canonical case-study of what it costs to make AlphaZero's
mechanism pay off at the scale it needs to. That's an extreme upper bound,
obviously not what a from-scratch, small-scale MCTS prototype would spend,
but it establishes the shape of the cost curve: AlphaZero-style methods are
cheap *per simulation* (a single forward pass) but expensive in aggregate
because the whole approach is a bet on doing an enormous number of them. A
scaled-down version for one ARC task would still need, at minimum: (a) a
value/policy network — meaning `torch` forward passes at every simulated
tree node, on top of (b) a game/DSL-execution step per simulation (the same
cost GP already pays per candidate), for (c) however many simulations per
move MCTS needs to meaningfully outperform GP's already-adequate search —
i.e., strictly more compute per decision than either existing trainer, for
a benefit (§2, §3) that's structurally capped by the tiny per-task training
corpus. On this project's CPU-only 16-core hardware (no GPU, per
`docs/QUESTIONS.md` Q9), a forward pass through even ADR-0008's small
conv/attention network at every simulated node, times a simulation budget
per move, is a strictly worse trade than GP's fitness evaluation (one DSL
program execution, no forward pass at all — see §3) on exactly the tasks
this repo's task subset and action-space depth (`max_program_length` ≤ 6)
are already scoped around. There's no realistic way this reads as cheaper
than what's already working; the honest framing is "more expensive per
decision, for a value signal this setting is too data-poor to make worth
the price" — not a close call.

## Net assessment

MCTS/AlphaZero-style search has zero ARC-AGI-specific track record — a
genuine gap, not a tried-and-failed result — but the two structural facts
this repo is built around (ADR-0008's per-task, solve-time-only training,
and ADR-0005's reward already being an exact, non-learned ground-truth
similarity function) remove almost all of AlphaZero's reason for existing
in the first place: there's no need to *learn* a value function when the
correct one is already computed in closed form, and no self-play bootstrap
problem when ground truth is available every step. The one legitimate
non-adversarial analog in the literature (Autodidactic Iteration) confirms
rather than undermines this: it works by manufacturing a training corpus
at a scale (billions of samples, tens of GPU-hours) this project's per-task
budget cannot approach, and MCTS's headline efficiency argument (fewer
rollouts than brute force, because the simulator is expensive) doesn't
apply when the "simulator" here — GP's DSL program execution — is already
near-free and already reaches solvable tasks well inside its search budget.

## Recommendation for this project

**Not worth building, including as a narrow prototype, given this repo's
current shape** — three independent reasons, any one of which would be
enough on its own:

1. **Per-task training removes AlphaZero's entire value proposition.**
   AlphaZero's bet is that a value/policy network gets better than any
   hand-designed heuristic *because* it trains on a massive, diverse
   self-play corpus. ADR-0008 trains from scratch per task on a handful of
   train pairs plus near-duplicate `re-arc` variations of the same concept
   — there is no scale to make that bet on, and (§2) the closed-form
   `similarity()` reward this repo already has *is* the value function a
   network would be trying to learn, computed exactly rather than
   approximately. A learned value net here would be strictly worse
   (approximate, needs training data it doesn't have) at a job already
   done exactly and for free.
2. **The self-play mechanism has a real non-adversarial analog (ADI/
   DeepCube), but its enabling condition is scale this project's per-task
   setting cannot produce.** This isn't "no two-player game exists, so
   category error" (too easy a dismissal) — it's "the technique that
   substitutes for a two-player game needs billions of self-generated
   samples and tens of GPU-hours (§4), and a single ARC task's train pairs
   plus `re-arc` variations of one fixed concept are neither numerous nor
   diverse enough to play that role." State this as the sharper, correct
   version of the mismatch, not the vaguer "single-agent domains can't
   self-play" claim.
3. **MCTS's core efficiency argument doesn't apply against GP's already-
   near-instant fitness evaluation.** MCTS pays for itself when simulation
   is expensive; GP's DSL program execution (§3) is microseconds-to-low-
   milliseconds and already reaches perfect fitness well inside its
   5,000-evaluation budget on tasks it can solve at all. Adding a
   value-network forward pass per simulated tree node (§5) is strictly
   more expensive per decision than what's already working, on CPU-only
   hardware, for a benefit capped by reason (1).

**If this is ever worth reopening**, it would be narrowly scoped to the one
place §3 identifies real headroom: tasks whose solving program exceeds GP's
`max_program_length` ceiling or whose fitness landscape is deceptive enough
that GP's mutation/tournament-selection gets stuck — i.e., a search-*depth*
or *deceptiveness*-robustness problem, not a "we need a smarter heuristic
than GP already has" problem. Even then, the right first move is to try
raising GP's program-length ceiling and diversity (larger population,
different selection pressure) before reaching for tree search, since that's
a much smaller change to a trainer that's already working, and only look at
MCTS if that fails on a task GP + PPO both provably can't solve. Do not
build a value/policy network or self-play loop as a general-purpose third
trainer track — the two structural reasons above (no scale to learn from,
no bootstrap problem to solve) apply to the whole task subset, not just the
hard tail.
