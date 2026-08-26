# RL & evolutionary-algorithm survey for ARC-AGI-1

Scope: non-LLM approaches only (pure RL, evolutionary/genetic program search, neuroevolution).
Findings from arXiv papers, the ARC Prize technical reports/blog, and GitHub prior art, gathered 2026-08-27.

## 1. Prior attempts — what exists, what worked, what didn't

**Pure RL is the least-explored path and the one paper doing it head-on reports it's hard.**
[ARCLE](https://arxiv.org/abs/2407.20806) (Lee et al., CoLLAs 2024, code on GitHub) is the one
purpose-built Gymnasium RL environment for ARC. It frames the grid as the RL state and symbolic
grid-editing operations (select region, recolor, move, etc.) as a discrete action space — the
same shape of environment we're planning. Their own framing of the problem: RL on ARC has "a vast
action space, a hard-to-reach goal, and a variety of tasks." They report PPO **can** learn
individual, single tasks, but:
- Sparse terminal-only reward (correct final grid) starves most rollouts of any signal — "most
  trajectories yield no reward," which stalls PPO training on its own.
- To make it tractable at all, they needed behavior cloning from a companion human-demonstration
  dataset ([ARCTraj](https://arxiv.org/pdf/2511.11079), human action-trajectories on ARC tasks) to
  bootstrap the policy toward promising regions of the huge action/state space, then fine-tuned
  with PPO on top.
- No result in this line of work claims cross-task generalization at meaningful accuracy — the
  demonstrated wins are per-task learning, not a general ARC-AGI-1 solver. This is consistent with
  ARC-AGI-1 being explicitly designed to defeat memorization/single-task overfitting.

**Genetic/evolutionary program search over a DSL is essentially unexplored as its own category** —
what exists under the "evolutionary" label for ARC is closer to brute-force/greedy DSL search than
true genetic programming (population, crossover, mutation, selection):
- The dominant classical-search lineage is [icecuber's 2020 Kaggle-winning solver](https://arxiv.org/pdf/2412.07322):
  142 hand-crafted unary grid functions, greedily composed and memoized into a DAG of "pieces,"
  then a greedy stacker assembles pieces to **minimize pixel distance** to the training outputs
  when no exact match exists. This is brute-force/greedy search with a shaped objective, not
  population-based evolution — but it's the closest thing to a proven, non-learned baseline and
  the number to beat: **~20% task accuracy on the private eval set**, per the [ARC Prize 2024
  Technical Report](https://arxiv.org/pdf/2412.04604) / [ARC Prize blog](https://arcprize.org/blog/beat-arc-agi-deep-learning-and-program-synthesis).
  Refined variants of this style later reached ~36%, and François Chollet has stated **up to ~50%
  is plausible from brute-force DSL search alone**, without any learned component.
- True genetic-programming ARC solvers (actual crossover/mutation of program ASTs, a population,
  fitness = train-pair match or pixel distance) do not show up as a named, benchmarked entry in
  the literature or the ARC Prize resources page — a gist survey of "approaches not tried"
  ([bigsnarfdude](https://gist.github.com/bigsnarfdude/7697459c610975989720aaf953bd15ff)) and the
  ARC Prize retrospectives both describe "discrete search over a DSL with program selection" as
  **"extremely underexplored,"** and note the DSL itself typically has to be hand-built rather than
  learned. This is a genuine gap, not a dead end — it means there's real room here, but also no
  existing benchmark number to calibrate expectations against for a from-scratch GP implementation.
- Related but distinct: [Neural Cellular Automata for ARC-AGI](https://arxiv.org/abs/2506.15746)
  (Xu & Miikkulainen, 2025) learns a local per-cell update rule (gradient-trained, not evolved,
  though NCA rule search is a classic evolutionary-search target) applied iteratively to transform
  the grid. It does well on tasks with local, structured patterns (e.g. spiral drawing) and
  explicitly struggles on tasks needing **global context propagation** — i.e. it's weak exactly
  where ARC-AGI-1 tasks require reasoning about the whole grid/object relationships at once. Useful
  as a cautionary data point for any local-update-rule-style evolutionary encoding.
- Neuroevolution (CMA-ES / ES evolving a policy network's weights, as in the general RL literature,
  e.g. ["Playing Atari with Six Neurons"](https://arxiv.org/pdf/1806.01363)) has no ARC-AGI-specific
  result at all in what surfaced here. It's a well-proven technique for small/compact policies on
  classic RL benchmarks, but nobody has published applying it to ARC's grid-editing action space.

**Net assessment**: don't expect either an RL or an evolutionary approach, built from scratch and
without an LLM in the loop, to be competitive with the ~20-50% brute-force-search or ~40-50%
LLM+search state of the art in this milestone. The realistic bar for a from-scratch RL/evolutionary
system is demonstrating non-trivial learning signal on a bounded task subset — which matches what
was already scoped as this milestone's success criterion, not "solve ARC-AGI-1."

## 2. Reward shaping for grid-similarity RL

- General RL literature is unambiguous that **sparse terminal-only reward** (exact match at
  episode end) is the harder-to-learn but "correct" objective, while **dense shaped reward**
  (continuous progress signal every step) trades some risk of policy distortion for much better
  sample efficiency. ARCLE's own finding — that sparse ARC reward alone stalls PPO and they needed
  human-demonstration behavior cloning to compensate — is a direct, ARC-specific data point in
  favor of shaping the reward rather than relying on sparse-only signal.
- The concrete shaping icecuber's solver effectively optimizes against (even though it's search, not
  RL) is **pixel-distance-to-target** as a tiebreaker/objective when no exact program match exists.
  That's a validated, ARC-appropriate dense signal: e.g. per-step reduction in Hamming distance (or
  a smoother measure like IoU per color, or normalized pixel match %) between the current grid and
  the target output.
- **Known pitfall — reward hacking via "vibrating in place."** The generic RL literature on
  similarity-shaped rewards (e.g. frame-similarity-based imitation rewards) documents a specific
  failure mode: an agent can find a state that scores well on the similarity metric and then
  oscillate around it without making real progress, because the metric rewards *closeness* without
  distinguishing genuine progress from stagnation or lucky partial overlap. For grid-editing this
  maps directly onto: repainting the same few cells back and forth, or reaching a partially-correct
  grid that already accounts for most of the pixel-similarity reward and having no gradient left to
  push toward full correctness. Mitigations documented in that literature: split the reward into a
  progress term (delta similarity vs. previous step, not absolute similarity) plus a separate
  penalty for redundant/no-op actions, rather than a single absolute-similarity reward.
- A second pitfall specific to grid tasks: absolute pixel-match reward is background-color-biased —
  a mostly-background (color 0) grid gets a high similarity score for doing nothing, which can trap
  a policy at a "paint nothing" local optimum. Suggests normalizing the similarity signal against
  non-background cells, or against the specific cells that need to change from input to output,
  rather than raw whole-grid pixel match.

## 3. Evolutionary program search vs. neuroevolution — CPU-only tradeoffs

Our hardware: 16 CPU cores, no GPU/CUDA. This materially favors one branch of "evolutionary" over
the other:

- **Genetic programming over a DSL (evolving program ASTs)** is naturally CPU-friendly and
  embarrassingly parallel: fitness evaluation is just executing a short program against a handful
  of train pairs (microseconds to low milliseconds per candidate on a grid ≤30×30), and a
  population of thousands of candidates can be evaluated in parallel across 16 cores with plain
  multiprocessing — no GPU, no autodiff needed. This is the more tractable branch to prototype fast
  on this machine, and it's also the one with essentially zero prior benchmarked art (see §1) —
  meaning it's simultaneously the cheaper build and the more novel contribution.
- **Neuroevolution (CMA-ES/ES on a policy network's weights)** is CPU-viable too — CMA-ES was
  specifically noted in the literature as effective for **small-size networks**, which fits a
  CPU-only budget — but every fitness evaluation requires running a full episode (many forward
  passes through the policy, one per action) rather than one program execution, so wall-clock cost
  per generation is higher for an equivalent population size. It's the better fit if we specifically
  want a neural policy without gradient-based RL (e.g. as a gradient-free alternative/ablation to
  PPO on the same Gymnasium env), but it's a less direct match to the DSL/action-space design than
  genetic programming is.
- Given the DSL/action-space decision already in flight (this repo's F1, being researched in
  parallel), genetic programming over that DSL is the natural "evolutionary" counterpart to test
  once the DSL exists — the same executor that runs a policy's chosen action can run a
  genetic-programming candidate's full program, so building the DSL executor once serves both
  tracks.

## 4. RL/evolutionary interoperation

Two genuinely complementary patterns show up across the literature, both compatible with "RL
first, evolutionary as fast-follow" while keeping one shared environment:

- **Search-guided-by-policy**: use the RL policy (or a simpler learned value/heuristic function) to
  bias which branches a program search or genetic algorithm explores first — analogous to how
  neural-guided program induction work (e.g. the "neurally-guided program induction" line cited in
  §1's search results) uses a network to score candidate program pieces rather than search
  uniformly. Concretely: the PPO policy's action logits or a learned per-action value estimate can
  seed the initial population / mutation proposal distribution for the genetic search, rather than
  the two tracks being fully separate.
- **Demonstration/self-play bootstrapping**: ARCLE's own finding — RL alone stalls on sparse
  reward, but behavior cloning from existing solved trajectories unstalls it — generalizes to using
  genetic-programming-discovered solutions (once found for some tasks) as demonstration
  trajectories to warm-start or fine-tune the RL policy on those and structurally similar tasks.
  This gives the two tracks a data-sharing loop: GP-found programs → replay as expert trajectories
  → behavior-clone into the policy → PPO fine-tunes further.

Both patterns only require that (a) the action space / DSL used by the RL policy and by the
genetic-programming search are the **same** primitive set, and (b) episode trajectories are logged
in a common format regardless of which trainer produced them — which is already the plan (single
Gymnasium-style env, JSONL trajectory logs, F4/F5 in `docs/QUESTIONS.md`).

## Recommendation for our environment & reward design

1. **Reward**: dense, delta-based shaping — reward = (similarity(grid_t, target) −
   similarity(grid_{t-1}, target)) using a non-background-normalized similarity measure (e.g. %
   of non-matching cells among cells that actually need to change, not raw whole-grid pixel match),
   plus a small per-step step-cost/no-op penalty to discourage vibrating in place, plus a terminal
   bonus for exact match. This directly addresses both documented pitfalls (sparse-reward stall,
   background-bias local optimum) while staying faithful to what the one existing ARC RL paper
   (ARCLE) found necessary.
2. **Don't rely on reward shaping alone to solve the sparse-reward problem** — ARCLE needed
   behavior cloning from human demonstrations to make PPO tractable at all. We don't have a human
   demonstration dataset, but we do have a cheaper substitute: once genetic-programming search
   (or brute-force DSL search) finds a solving program for a task, log its execution trace as a
   trajectory and use it for behavior-cloning warm-start of the RL policy on that task and
   near-neighbors. This is the concrete interoperation hook from §4 and argues for building the
   trajectory-logging format (already planned for the visualizer, F5) to be reusable as BC training
   data from day one, regardless of which trainer produced the trajectory.
3. **Same action space for both tracks**: whatever DSL/primitive set F1's research lands on should
   be the action space for the RL policy's discrete actions *and* the terminal set for
   genetic-programming's program ASTs. Don't build two separate primitive libraries.
4. **Evolutionary track choice**: prototype genetic programming over the DSL before neuroevolution
   — it's cheaper to evaluate on CPU-only hardware (no forward passes, just program execution),
   has zero existing benchmarked competition to worry about matching/beating, and reuses the same
   executor the RL env needs anyway. Keep CMA-ES/neuroevolution as a documented option for later,
   not part of this milestone.
5. **Calibrate ambition honestly**: ~20% task accuracy (icecuber, pure brute-force search, no
   learning) is the realistic floor for "search-based methods work at all" on ARC-AGI-1; the
   ~40-50% state of the art is LLM+search and out of scope by design. Milestone success here should
   be measured as "reward/success rate improves measurably over training on the scoped task subset
   and is visually inspectable in the replay tool" — not as approaching either of those numbers.
