# F22 - Why the time-aware agent lost (Qwen bench v2 traces, 2026-11-01)

Data: `bench2` on Kaggle (L4, 12 tasks x 2 configs, `trace_agent_v2_cap12k.jsonl` and `..._cap8k_early.jsonl`). Result 1/16 and 0/16
against direct 6/16. All numbers below are counted from the traces.

| | cap 12k | cap 8k, submit-frac .35 |
|---|---|---|
| model turns | 56 | 59 |
| turns ended by the per-turn cap (`finish=length`) | 21 | 24 |
| cut turns that contained a tool call | 0 | 0 |
| median completion tokens per turn | 6,290 | 6,492 |
| tasks with zero python calls | 8 of 12 | 9 of 12 |
| tasks that submitted at least once | 11 of 12 | 12 of 12 |

## What happened

1. On most tasks the first turn spent the whole cap thinking and never reached a tool call. All 45 cut turns, in both configs, ended
   without one.
2. The harness then fired the forced submit (`rescue:cut`), which asks for an answer from the same short reasoning. So an "agent"
   run on 8-9 of 12 tasks was one capped thinking turn plus one forced answer, at 17-25k tokens and 11-16 minutes, much less
   thinking than direct's 40k-token answers.
3. Only 3-4 tasks per config got real tool use (2-15 python calls), and those used 28-63k tokens. One verified program, zero correct on test.

## Conclusions

- The cap trades the model's thinking for a worse answer. Per-turn caps are the wrong lever when thinking comes before the first
  action. Direct mode wins because it lets the thinking finish.
- Agent mode needs a budget big enough for one full think plus a few tool rounds (>= ~60k per task), which at ~22 tok/s per stream
  and ~200 tok/s aggregate is about 20 tasks per hour. That is not 240 tasks in 11 hours.
- Hybrid keeps direct's answers as the floor, so the agent phase can only add. Its best use is on tasks where direct failed to
  produce a verifying answer, with a large per-task budget and few tasks, not a thin budget on all of them.
- ARC-166: plateau data is too thin here (1-2 tasks with more than 3 submits). ARC-169: only one verified program exists in these traces,
  too few to calibrate guards.

## Actions

- For the agent phase of hybrid, order tasks by "direct produced no train-verified answer" first (already implied by
  `hybrid_order`; check it) and lower concurrency so each stream is faster.
- Do not set a per-turn cap below the model's typical think length; prefer the time-share cut.
- Record in ARC-136/166/169/170.
