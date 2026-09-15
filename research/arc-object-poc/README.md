# arc-object-poc

Throwaway proof-of-concept for **F15 / ADR-0027** — does an object-centric
representation break the 77% ceiling? Gated *before* any rewrite of `arc_env/`,
`trainers/`, or `viz/`, which this dir does **not** touch. It reuses the
vendored `arc-dsl` executor and `re-arc` generators as libraries only.

Target: the 8-task "set-op cluster" (split the grid into two regions, combine a
color's cells from each via a set operation, paint the result).

## Layout

| File | What |
|------|------|
| `object_actions.py` | the object state model + a 5-verb object action space, mirroring the shipped executor's surface (`ACTIONS` / `execute`) |
| `programs.py` | the 8 hand-written object-space programs |
| `verify.py` | reusable re-arc generalization verifier (`generate_pair` × run × pass-rate) — the check ADRs 0023–0026 did ad-hoc |
| `run_gate1.py` | Gate 1: expressibility + generality |
| `gp_object.py` | Gate 2: free-form GP over the object actions |
| `grammar_search.py` | Gate 2: the *typed grammar* search (the headline contrast) |
| `tests/test_poc.py` | deterministic self-check (not in `make test`) |
| `RESULTS.md` | findings + go/no-go recommendation |

## Run

```
uv run python research/arc-object-poc/run_gate1.py --n 30
uv run python research/arc-object-poc/gp_object.py --seeds 3
uv run python research/arc-object-poc/grammar_search.py
uv run pytest research/arc-object-poc/tests/ -q
```

See `RESULTS.md` for what it found — short version: the object model expresses
all 8 cleanly (Gate 1), but discovery depends on a **typed action grammar**
(0/8 free-form vs 8/8 typed), so *structure*, not pixels-vs-objects, is the
real lever.
