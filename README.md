# arc-agi

An agent that learns to solve [ARC-AGI-1](https://arcprize.org/arc-agi/1/) puzzles by editing a grid one action at a time, plus a local visualizer for watching it happen: a training dashboard, and a step-by-step replay of the agent "playing" a puzzle the way a human would in ARC-AGI's own testing interface.

Two trainers share one environment. A hand-rolled PPO implementation learns a policy from a dense reward signal. A genetic-programming search evolves a population of short DSL programs instead. Both train one dedicated model per task rather than a single model shared across tasks, an approach every prior solver in this repo's history has taken, and the one thing the existing ARC-specific RL literature has actually demonstrated working.

The starting point was a non-learned geometric-transform baseline (still in `legacy/`, kept as a sanity check) and a half-built supervised program-synthesis scaffold that never got object-level manipulation working (`research/arc-ngps/`, superseded but not deleted). Neither of those learns from experience, and neither lets you watch a policy actually improve. This revamp does both.

## Quick start

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/), and Node 20+.

```bash
make            # no target: lists every available command
make install    # uv sync (Python) + npm ci (frontend)
make rollout    # random-policy episodes across all 24 curated tasks
make train      # trains a PPO policy on one task (~90s)
make viz        # builds the frontend, serves the dashboard + replay UI at :8000 (override with PORT=)
make demo       # train + viz in one command - trains, then opens the visualizer on that run
```

Or run things by hand:

```bash
uv run python scripts/rollout_random.py --task_id 67a3c6ac
uv run python train.py --algo ppo --task_id 67a3c6ac --run_id demo
uv run python train.py --algo gp --task_id 67a3c6ac --run_id demo-gp
uv run python -m viz.backend.server   # http://127.0.0.1:8000, reads runs/
```

`train.py` always trains one policy (or evolves one population) for a single `--task_id`. There's no shared model across tasks, so solving N tasks means N separate runs, each writing its own `runs/<run_id>/` directory that the visualizer can open.

## How it works

**The environment.** `arc_env/` wraps a curated slice of Michael Hodel's [arc-dsl](https://github.com/michaelhodel/arc-dsl) as a discrete, factored action space: 27 actions in total, from simple zero-argument transforms like `rot90` and `vmirror` up to `commit`, a four-argument action that crops the working grid to a chosen region and ends the episode there. Higher-order primitives (`compose`, `chain`, `fork`, and friends) are excluded on purpose. They build closures rather than transforming a grid directly, and a flat "pick one action per step" space has no way to represent that. Picking an action that needs an argument the DSL can't express as a plain color, coordinate, or size (an arbitrary object, say) is out of scope for the same reason; see [`docs/adr/0001`](docs/adr/0001-arc-dsl-as-action-space.md) for the full reasoning. ADR-0010's Phase 1 (2026-08-29) added 4 self-concatenation actions (`hconcat_self` and friends) on the same basis, growing the curated task subset from 16 to 24.

Reward is dense, not sparse: at each step it's the change in how many of the "should differ" cells now match the target, minus a small per-step cost, plus a bonus on an exact match. A purely terminal reward starves both trainers of signal on anything but the most trivial task, which is exactly what happened in the one prior ARC RL paper we could find.

**The task set.** Not every ARC-1 task is solvable with this action space, and rather than guess which ones are, we checked. Of the 400 training tasks with a known-correct `arc-dsl` solver, 24 can be solved using only the curated actions: 12 where the output is the same shape as the input, 12 where it isn't. That list lives in `arc_env/task_loader.py`, derived by parsing every solver program and testing it against the action whitelist, not picked by hand.

**PPO** (`trainers/ppo/`). A small network, about 178K parameters: a learned embedding per color over the padded 30x30 grid, a couple of residual conv blocks, one self-attention layer so it can notice relations a local convolution would miss (symmetry, "the other object like this one"), and a factored action head that picks a primitive first and its arguments second. No pretraining. The encoder learns end-to-end from the PPO signal on that one task's handful of instances, which is a narrow enough job that a general-purpose pretrained representation wouldn't obviously help.

**Genetic programming** (`trainers/gp/`). A program here is just a flat list of (action, arguments) pairs, not a tree. Since the DSL subset has no composition to represent, that list is already the natural unit, the same one PPO's per-step action already uses. Fitness is the fraction of training pairs solved exactly, with average pixel similarity as a tiebreaker. Crossover splices two programs at a random point; mutation resamples a single argument, swaps a whole gene, or inserts/deletes one. On anything the action space can express in a handful of steps, this finds a perfect solution in single-digit milliseconds, no gradient descent involved.

**re-arc.** Beyond each task's native 3-5 training pairs, `arc_env/re_arc.py` generates fresh synthetic instances of the same concept on demand, using Michael Hodel's [re-arc](https://github.com/michaelhodel/re-arc) generators, so PPO's rollouts see more variety than the raw ARC dataset provides.

**The visualizer** (`viz/`). A small Python backend serves a `runs/` directory as JSON. The frontend is TypeScript and Canvas: a training dashboard (reward and success-rate curves, whether the run came from PPO or GP) and two side-by-side replay panels, so you can step through an early checkpoint and a late one at the same time and watch the difference. It follows ARC-AGI's own color palette, so a replayed episode looks like the same puzzle you'd see in the official testing interface.

## What actually works right now

PPO and GP both solve the easy end of the task set reliably and fast: single-action tasks converge in well under two minutes of wall-clock training, GP faster still since there's no network to optimize. The reward used to score a flat zero for any output of the wrong shape, so there was no gradient at all toward `commit`'s exact four-argument combination until an agent stumbled onto the right dimensions by chance. Fixed 2026-08-29 (`docs/adr/0005-dense-delta-shaped-reward.md`'s amendment): `similarity` now gives partial credit for how close the shape is before falling back to content-matching once it's exact. PPO's success rate on the `commit`-requiring fixture task went from 0% to 20-25% within a single ~3.5-minute training run after the fix.

Scaling past the original 16-task set needed a real decision first, not just more compute (`docs/adr/0010-task-coverage-scaling.md`): over half of the 400 training tasks need the higher-order primitives this action space deliberately excludes, and most of the rest need selecting or manipulating a specific object, which has no clean way to be a small categorical action yet (a scoped-but-not-yet-designed fast-follow). What's already landed (2026-08-29): broadening the curated actions with 4 self-concatenation primitives, taking the task set from 16 to 24 with no new representational mechanism.

## Repo layout

```
third_party/ARC-AGI/       official ARC-AGI-1 dataset + testing interface (vendored, read-only)
third_party/arc-dsl/       Michael Hodel's arc-dsl - the action-space DSL and 400 known-correct solvers
third_party/re-arc/        Michael Hodel's re-arc - per-task synthetic instance generators

arc_env/                   the Gymnasium-style environment: actions, task loader, reward, env, JSONL logging
trainers/ppo/              the policy/value network, rollout collection + GAE, the PPO update
trainers/gp/               genome representation, fitness, the evolutionary loop, replay for logging
train.py                   train.py --algo ppo|gp --task_id <id>

viz/backend/               read-only HTTP server exposing runs/ as JSON
viz/frontend/              TypeScript + Canvas dashboard and replay UI

scripts/rollout_random.py  random-policy baseline rollout (no training)
tests/                     pytest suite - see Testing below
docs/                      the plan, ADRs, and vertical slices this was built against

legacy/                    the original non-learned baseline, kept as a sanity check
research/arc-ngps/         a superseded supervised program-synthesis scaffold, not on the current path
```

`runs/` is created locally when you train or roll out an episode and is gitignored; nothing there is meant to be committed.

## Testing

```bash
make test           # the fast layer: 264 tests, ~4 seconds, no GPU or training runs involved
make test-py-slow   # adds the PPO convergence check (~90s): does reward actually improve over training?
```

CI runs five jobs in parallel - `python-tests`, `python-tests-slow`, `frontend`, `lint` (`ruff check .`), and `security` (`gitleaks` secret scan + `pip-audit`, plus `npm audit` in the `frontend` job) - and `main` is protected on all five passing before merge. Dependabot opens weekly update PRs for the `uv`, `npm`, and GitHub Actions dependencies.

`git config core.hooksPath .githooks` installs a pre-push hook that mirrors the fast checks (`ruff`, the fast test layer, frontend typecheck/tests, and a local `gitleaks` scan if installed) so most red CI runs never happen.

The regression backbone is `tests/test_dsl_regression.py`: every curated task's known-correct `arc-dsl` solver gets replayed through this project's own action executor and has to reproduce the exact expected output. Since the ARC dataset and Hodel's solvers already establish ground truth, this is free, strong coverage for the part of the system where a subtle bug would be easiest to miss.

## Further reading

- [`docs/PLAN.md`](docs/PLAN.md) - the agreed scope and requirements
- [`docs/SLICES.md`](docs/SLICES.md) - the four vertical slices this was built in, each with its own demo and test plan
- [`docs/adr/`](docs/adr/) - the design decisions and the alternatives considered for each
- [`CLAUDE.md`](CLAUDE.md) - the terse version, kept up to date as an agent-facing map of the repo

## License

Apache 2.0 (see [`LICENSE`](LICENSE)) for this project's own code. Vendored third-party code under `third_party/` keeps its own upstream license: MIT for `arc-dsl` and `re-arc`, ARC-AGI's own license for the dataset itself.
