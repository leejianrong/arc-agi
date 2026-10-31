# F20 - What the Haiku + semula runs tell us (2026-11-01)

Data: five semula projects exported to `runs/semula-exports/` (gitignored; `semula export <project> --dir ...`), about 133
recorded model calls. They cover 14 agent conversations (python + submit tools) on six public ARC-AGI-2 eval tasks across
three runs, two verified-program synthesis runs on the same six tasks, a 20-task scheduler test, and four easy ARC-AGI-1
tasks. Every submitted program was re-run on the held-out test input and compared with the public answer.

**Read this with its limits.** Haiku is not Qwen: its habits (few tool calls, short reasoning) differ, six tasks is a tiny
sample, and nothing here measures Qwen's accuracy. What the runs do show reliably is how the *harness* behaves and where
its verification is weak.

## Going well

- **Mechanics hold up.** About 130 tool-call turns through the real loop with no crash, no malformed state, tags and
  diagnostics as designed. Contract bugs found by the players (tool text vs REPL namespace, traceback leaks, untagged
  replies, vague duplicate warning) were fixed.
- **Richer diagnostics help.** After the colour-pair summary and expected/actual grids were added, partial scores moved:
  `13e47133` went 43% -> 86% -> 98% of cells right, `142ca369` 9% -> 56% -> 77%, and `135a2760` reached a verified program
  after the pairs exposed a period-search overfit.
- **Replay observability works.** The viewer's totals matched numbers computed independently from the trace.
- Agent mode and synthesis mode verify the same two tasks (`0934a4d8`, `135a2760`) with Haiku, so the agent loop is no
  worse than the single-shot loop on this sample.

## Not going well

1. **Verified on train is not correct on test, and the failures were detectable.** Both programs that verified were wrong
   on the test input:
   - `0934a4d8`: a general symmetry-completion program with the right output shape that **leaves 8 masked cells unfilled**
     (8 wrong). Colour 8 is in every train input and in no train output, so an output still containing it is incomplete.
     The existing plausibility guard missed it because 8 is in the test input.
   - `135a2760`: a "repair stray cells" task. Train examples change 1 and 9 cells (1.5% and 1.9% of the grid); the program
     changed **47 test cells (5.6%) when 9 needed fixing**, ending 52 wrong. A change-fraction well outside the train
     envelope is detectable.
2. **Plateaus and looping.** Of 7 runs with 3 or more scored submits, the best result was among the first two submits in
   5; the others improved 17-29% at best. Seven submits repeated an earlier wrong-cell count (programs differing in text,
   not in behaviour). On `13e47133` (about 600 wrong cells per example) nothing improved over four tries.
3. **Little exploration.** Median 2 python calls per conversation (max 5), and 4 of 14 conversations made their first
   submit before running any python. Programs are guesses from one or two measurements; the REPL is used for printing,
   not for testing hypotheses. (Qwen at xhigh may behave very differently; the bench traces will show.)
4. **Most hard tasks stay far away.** Four of six tasks never get close, which is expected for Haiku and says nothing
   about Qwen.

## A free measurement: what would two new guards cost?

Both checks only need the demos and the program's test output, no answers. Cost = how often a *correct* answer would be
flagged, measured on the true outputs of all public tasks:

| Rule | False alarms, 1,076 training test inputs | False alarms, 167 eval test inputs |
|---|---|---|
| Leaked colour: a colour in every train input and in no train output appears in the test output | 1.7% | 4.8% |
| Change envelope: for same-shape tasks, test changed-cell fraction > max(3x the largest train fraction, that + 0.02) | 0.3% | 0.6% |

On our own data the two rules flag both verified-but-wrong programs (one each). That is two cases, so it is encouraging,
not proof. Used only to **re-rank** candidates and to **keep refining instead of stopping**, a few percent of false
alarms is cheap.

## Ideas, ranked

| # | Idea | Why | How to test |
|---|---|---|---|
| A | **Verified-but-suspect keeps going.** When a program reproduces all demos but trips a guard on the test input, do not end the attempt: reply "reproduces all examples, but on the test input the output still contains colour 8 / changes N% of cells; fix the generalization", and show the test output grid. Add the two rules above to `plausibility`. | Both silent failures above become refinement rounds; costs a few tokens | semula + Haiku replay of the two tasks; then Qwen bench |
| B | **Progress-aware submit replies.** Show best-so-far, trend ("worse than submission 3: 117 wrong cells"), and a list of behaviours already tried; expose `submissions` (programs and results) in the REPL. | Addresses plateau and looping | Haiku on the same six tasks; compare repeats and best-submit position |
| C | **Plateau policy.** After K non-improving submits (try 3), restart with fresh context plus a summary of what is ruled out, or run diverse independent attempts and vote. | Stops sinking tokens into a stuck line | Needs Qwen tokens: a bench config |
| D | **Exploration nudge.** Require or nudge N python calls before the first submit. | Haiku rarely explores | Check Qwen's real python-per-task from bench traces first; do not tune to Haiku |
| E | **Equivariance as ranking evidence.** Run a candidate on colour-permuted or transposed demos and prefer programs that stay correct when the task is colour-agnostic. | A general rule should survive relabelling | Needs false-positive measurement (not all tasks are colour-agnostic) |

Recommended order: **A, then B** (both testable for free with semula and Haiku), then use the Qwen bench traces to decide
C and D.

## What this cannot tell us

How Qwen behaves (token use, exploration, plateaus), whether the guards help it, and anything about accuracy. The
bench kernel's `BENCH` lines and the next traced Kaggle run are what answer those.
