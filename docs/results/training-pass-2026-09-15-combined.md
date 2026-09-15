# Training pass — 91 curated tasks (2026-09-14/15)

Generated 2026-09-15 05:20:10Z from `fullpass-20260914-222030` (GP + PPO-cold, all 91) + `warmpass-20260915-100457` (PPO-warm on the 29 GP-solved∩cold-failed tasks).

Settings: PPO n_updates=30, rollout_steps=256, eval_every=5; GP generations=100, population=200; max_steps=25, seed=0. Serial (one training process at a time, RAM-constrained machine). 'Solved' = GP best_fitness==1.0 / PPO final eval_success (fixed held-out pair's greedy exact_match).

## Headline

| Arm | Solved | Rate |
|-----|--------|------|
| GP | 70/91 | 77% |
| PPO cold-start | 41/91 | 45% |
| PPO + GP warm-start (cold-or-warm) | 69/91 | 76% |

- **Warm-start rescued 28/29** of the tasks PPO failed cold but GP solved — near-complete transfer of GP's solution into a learned PPO policy (ADR-0009 validated). The one it didn't rescue: `42a50994`.
- **21 tasks unsolved by any arm** — GP's short-program search can't find them and PPO can't learn them; the practical ceiling of this architecture at these settings.
- vs the 2026-08-31 pass (26 tasks: GP 96%, PPO 54%): GP's rate fell to 77% as harder tasks entered the curated set, and PPO-cold held ~comparable (45%) despite the action space tripling (30→85) — the exploration cost of more actions is real but modest, and warm-start more than recovers it.

## Unsolved by any arm (21)

`0d3d703e`, `1b2d62fb`, `1c786137`, `3428a4f5`, `46f33fce`, `5bd6f4ac`, `6430c8c4`, `80af3007`, `94f9d214`, `99b1bc43`, `a740d043`, `a79310a0`, `a9f96cdd`, `ce4f8723`, `cf98881b`, `d10ecb37`, `d364b489`, `dae9d2b5`, `ea32f347`, `f2829549`, `fafffa47`

## Per-task

| Task | GP | PPO-cold | PPO-warm |
|------|----|----------|----------|
| 007bbfb7 | yes | no | yes |
| 0d3d703e | no | no | — |
| 1b2d62fb | no | no | — |
| 1bfc4729 | yes | yes | — |
| 1c786137 | no | no | — |
| 1cf80156 | yes | yes | — |
| 1f85a75f | yes | yes | — |
| 1fad071e | yes | yes | — |
| 2013d3e2 | yes | no | yes |
| 23b5c85d | yes | yes | — |
| 25ff71a9 | yes | no | yes |
| 28bf18c6 | yes | no | yes |
| 2dee498d | yes | yes | — |
| 32597951 | yes | no | yes |
| 3428a4f5 | no | no | — |
| 3af2c5a8 | yes | yes | — |
| 3c9b0459 | yes | yes | — |
| 42a50994 | yes | no | — |
| 44f52bb0 | yes | yes | — |
| 46442a0e | yes | yes | — |
| 46f33fce | no | no | — |
| 4c4377d9 | yes | no | yes |
| 5582e5ca | yes | no | yes |
| 5614dbcf | yes | no | yes |
| 5bd6f4ac | no | no | — |
| 5c0a986e | yes | yes | — |
| 6150a2bd | yes | yes | — |
| 62c24649 | yes | yes | — |
| 6430c8c4 | no | no | — |
| 67385a82 | yes | yes | — |
| 67a3c6ac | yes | yes | — |
| 67a423a3 | yes | yes | — |
| 67e8384a | yes | yes | — |
| 68b16354 | yes | yes | — |
| 6d0aefbc | yes | yes | — |
| 6d75e8bb | yes | yes | — |
| 6fa7a44f | yes | yes | — |
| 7468f01a | yes | no | yes |
| 74dd1130 | yes | no | yes |
| 7b7f7511 | yes | yes | — |
| 7fe24cdd | yes | yes | — |
| 80af3007 | no | no | — |
| 88a10436 | yes | yes | — |
| 88a62173 | yes | no | yes |
| 8be77c9e | yes | yes | — |
| 8d5021e8 | yes | yes | — |
| 8f2ea7aa | yes | no | yes |
| 90c28cc7 | yes | no | yes |
| 9172f3a0 | yes | no | yes |
| 928ad970 | yes | no | yes |
| 94f9d214 | no | no | — |
| 99b1bc43 | no | no | — |
| 9dfd6313 | yes | no | yes |
| a416b8f3 | yes | no | yes |
| a68b268e | yes | no | yes |
| a740d043 | no | no | — |
| a79310a0 | no | no | — |
| a9f96cdd | no | no | — |
| aabf363d | yes | no | yes |
| ac0a08a4 | yes | yes | — |
| b1948b0a | yes | no | yes |
| b548a754 | yes | yes | — |
| b91ae062 | yes | yes | — |
| b94a9452 | yes | no | yes |
| be94b721 | yes | no | yes |
| c1d99e64 | yes | yes | — |
| c3e719e8 | yes | yes | — |
| c59eb873 | yes | yes | — |
| c8f0f002 | yes | no | yes |
| c909285e | yes | yes | — |
| c9e6f938 | yes | no | yes |
| ce4f8723 | no | no | — |
| cf98881b | no | no | — |
| d0f5fe59 | yes | yes | — |
| d10ecb37 | no | no | — |
| d364b489 | no | no | — |
| d511f180 | yes | no | yes |
| d631b094 | yes | yes | — |
| dae9d2b5 | no | no | — |
| e3497940 | yes | yes | — |
| e8593010 | yes | yes | — |
| e9afcf9a | yes | no | yes |
| ea32f347 | no | no | — |
| eb281b96 | yes | yes | — |
| ed36ccf7 | yes | yes | — |
| f25fbde4 | yes | no | yes |
| f25ffba3 | yes | no | yes |
| f2829549 | no | no | — |
| f76d97a5 | yes | yes | — |
| fafffa47 | no | no | — |
| ff805c23 | yes | yes | — |
