"""Generate sample_trace.jsonl: a clearly synthetic trace following docs/arc2-trace-schema.md.

Run: python3 tools/agent_watch/sample/make_sample.py
Three tasks: 135a2760 (failing submit first, then verified), 136b0064 (never verified),
142ca369 (cut-off turn, still running). Two more ids are listed in run_start only (queued).
"""

import json
from pathlib import Path

OUT = Path(__file__).with_name("sample_trace.jsonl")
CFG = "default"


def pattern(h, w):
    return [[(1, 2)[c % 2] if r % 2 == 0 else (3, 4)[c % 2] for c in range(w)] for r in range(h)]


def with_cells(grid, changes):
    g = [row[:] for row in grid]
    for (r, c), v in changes.items():
        g[r][c] = v
    return g


def wrong_of(expected, got, ex):
    cells = [[r, c, got[r][c], expected[r][c]] for r in range(len(expected)) for c in range(len(expected[0])) if got[r][c] != expected[r][c]]
    return {"ex": ex, "total": len(cells), "cells": cells[:10]}


events = []


def ev(t, kind, task=None, **kw):
    e = {"ev": kind, "t": round(t, 3), "cfg": CFG}
    if task:
        e["task"] = task
    e.update(kw)
    events.append(e)


def turn(t, task, n, completion, reasoning, calls, finish="tool_calls", content="", dur=20.0, att=0):
    ev(t, "turn", task, att=att, turn=n, prompt_tokens=3000 + 900 * n, completion_tokens=completion,
       reasoning_tokens=max(0, completion - 120), finish=finish, dur_s=dur, reasoning=reasoning, content=content,
       tool_calls=[{"id": f"call_{task}_{n}_{i}", "name": nm, "args": {"code": code}} for i, (nm, code) in enumerate(calls)])


def py_result(t, task, n, text, att=0, dur=0.4):
    ev(t, "tool_result", task, att=att, turn=n, name="python", text=text, dur_s=dur)


def submit(t, task, n, program, train, outputs, flags=(), tests=(), att=0):
    wrong = [wrong_of(train[i]["output"], o, i) for i, o in enumerate(outputs) if o is not None and o != train[i]["output"]]
    repro = sum(1 for i, o in enumerate(outputs) if o == train[i]["output"])
    verified = repro == len(train)
    ev(t, "submit", task, att=att, turn=n, program=program, reproduced=repro, n_train=len(train), verified=verified,
       outputs=outputs, wrong=wrong, flags=list(flags), test_outputs=list(tests))
    msg = f"VERIFIED on all {len(train)} examples." if verified else f"{repro} of {len(train)} examples reproduced."
    ev(t + 0.05, "tool_result", task, att=att, turn=n, name="submit", text=msg, dur_s=0.05)
    return verified


# ---------------------------------------------------------------- task A: fail first, then verified
A = "135a2760"
a_train = []
for h, w, strays in [(4, 8, {(2, 4): 5}), (4, 10, {(0, 9): 5, (2, 6): 6})]:
    exp = pattern(h, w)
    a_train.append({"input": with_cells(exp, strays), "output": exp})
a_test_in = with_cells(pattern(6, 12), {(1, 3): 5, (4, 8): 7})
a_test_out = pattern(6, 12)
ev(0.0, "run_start", run="sample-001", model="synthetic-model", effort="xhigh", mode="agent", persist=1,
   tasks=[A, "136b0064", "142ca369", "16b78196", "1818057f"], started_at="2026-11-01T09:00:00Z")
ev(1.0, "task_start", A, train=a_train, test=[{"input": a_test_in}], answers=[a_test_out])
prog1 = "def solve(g):\n    out = [row[:] for row in g]\n    for r, row in enumerate(g):\n        for c in range(len(row)):\n            if c >= 2 and row[c] != row[c - 2]:\n                out[r][c] = row[c - 2]\n    return out\n"
prog2 = "def solve(g):\n    out = [row[:] for row in g]\n    for r, row in enumerate(g):\n        for c in range(len(row)):\n            period = [row[k] for k in range(c % 2, len(row), 2)]\n            out[r][c] = max(set(period), key=period.count)\n    return out\n"
turn(30, A, 1, 6400, "Rows look periodic with period 2, and one or two cells per grid break the period. Print the diff between input and output to see which cells change. The stray colors differ per example, so the rule should not depend on a color.\n" + "I will compare each row against its period. " * 6,
     [("python", 'for ex in train:\n    print(diff(ex["input"], ex["output"]))')], dur=29.0)
py_result(31, A, 1, "1 of 32 cells differ: (2,4):5->1\n2 of 40 cells differ: (0,9):5->2 (2,6):6->1")
turn(48, A, 2, 900, "Each stray cell is replaced by the value two columns to its left. Try that as a first program.",
     [("submit", prog1)], dur=17.0)
o1 = [a_train[0]["output"], with_cells(a_train[1]["input"], {(0, 9): 2})]
submit(48.4, A, 2, prog1, a_train, o1, tests=[a_test_out])
turn(95, A, 3, 5200, "The first program fails when the stray sits in the first two columns, and it misses a stray whose left neighbour is also wrong. Use a majority vote per parity class instead of the left neighbour, which is robust to one bad cell.\n" + "Majority over each row and parity. " * 8,
     [("submit", prog2)], dur=46.0)
submit(95.5, A, 3, prog2, a_train, [g["output"] for g in a_train], tests=[a_test_out])
turn(110, A, 4, 160, "", [], finish="stop", content="Submitted a verified program: majority vote per row and column parity.", dur=4.0)
ev(110.5, "attempt_end", A, att=0, status="verified", verified=True, tokens=12660, turns=4, python_calls=1, submits=2, wall_s=109.5)
ev(110.6, "task_end", A, rows=[{"attempt_1": a_test_out, "attempt_2": None}], verified=1, solved=[True])

# ---------------------------------------------------------------- task B: never verified
B = "136b0064"
b_exp = [[0, 3, 0, 3], [3, 3, 0, 0], [0, 3, 3, 3]]
b_exp2 = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
b_train = [{"input": [[3, 0, 0, 3], [0, 3, 3, 0], [3, 0, 3, 3]], "output": b_exp},
           {"input": [[1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 0]], "output": b_exp2}]
ev(2.0, "task_start", B, train=b_train, test=[{"input": [[3, 3, 0, 0], [0, 3, 0, 3], [3, 0, 3, 0]]}], answers=[None])
turn(40, B, 1, 3100, "The output seems to be a flipped, reordered version of the input. Hypothesis: reverse each row, then roll the rows down by one.",
     [("python", "def f(g):\n    return [row[::-1] for row in g[-1:] + g[:-1]]\nprint(f(train[0]['input']))")], dur=34.0)
py_result(41, B, 1, "[[3, 3, 0, 3], [3, 0, 0, 3], [0, 3, 3, 0]]\nexpected shape (3, 4): ok, content differs")
prog_b1 = "def solve(g):\n    return [row[::-1] for row in g[-1:] + g[:-1]]\n"
turn(75, B, 2, 2400, "Python output does not match either example. Submit it anyway to get exact wrong cells.", [("submit", prog_b1)], dur=30.0)
ob = [with_cells(b_exp, {(0, 0): 3, (1, 3): 3}), with_cells(b_exp2, {(2, 1): 1})]
submit(75.4, B, 2, prog_b1, b_train, ob, flags=["hardcodes training literal"])
prog_b2 = "def solve(g):\n    return TRAIN_LOOKUP[str(g)]\n"
turn(120, B, 3, 4100, "Nothing generalises. As a last resort, look the grid up. This is clearly overfit but exercises the flag.", [("submit", prog_b2)], dur=50.0)
submit(120.5, B, 3, prog_b2, b_train, [None, None], flags=["hardcodes training literal", "raises on unseen input"])
ev(140, "attempt_end", B, att=0, status="deadline", verified=False, tokens=9600, turns=3, python_calls=1, submits=2, wall_s=138.0)
ev(140.1, "task_end", B, rows=[{"attempt_1": None, "attempt_2": None}], verified=0, solved=[False])

# ---------------------------------------------------------------- task C: cut-off turn, still running
C = "142ca369"
c_train = [{"input": [[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            "output": [[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 2, 0]]},
           {"input": [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "output": [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 0, 8], [0, 0, 0, 0]]}]
ev(3.0, "task_start", C, train=c_train, test=[{"input": [[0, 0, 0], [0, 4, 0], [0, 0, 0]]}], answers=[None])
turn(60, C, 1, 4800, "A single pixel shoots a diagonal ray down and to the right until it leaves the grid. Check with a short script.",
     [("python", "print(shoot(train[0]['input']))")], dur=52.0)
py_result(61, C, 1, "NameError: name 'shoot' is not defined", dur=0.1)
turn(160, C, 2, 49152, "Let me define the ray carefully. The pixel at (1,1) shoots to (2,2), (3,3) and stops at the edge. In example two the pixel at (1,2) shoots to (2,3) and stops. I keep re-deriving the same rule from the objects and the borders, considering whether other colors bounce, whether rays continue off-grid... [cut 41000]",
     [], finish="length", dur=99.0)

# ---------------------------------------------------------------- heartbeats
for i, (t, tok) in enumerate([(30, 212), (90, 251), (150, 268), (165, 259)]):
    ev(t + 0.5, "heartbeat", tok_s=tok, running=3 - (1 if t > 140 else 0), waiting=0, kv_pct=40 + i * 6, gpu_mem_mib=71000 + i * 120)

events.sort(key=lambda e: e["t"])
OUT.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events))
print(f"wrote {len(events)} events to {OUT}")
