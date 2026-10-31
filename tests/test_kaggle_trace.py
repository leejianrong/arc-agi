import copy
import importlib.util
import json
import threading
from pathlib import Path

_p = Path(__file__).resolve().parent.parent / "kaggle/arc2_synth/trace.py"
_spec = importlib.util.spec_from_file_location("arc2_trace_under_test", _p)
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)

G = [[0, 1], [2, 3]]
CALL = {"id": "c1", "name": "submit", "args": {"code": "x"}}
BASE = {"t": 0.0, "cfg": "default"}
VALID = {
    "run_start": {"run": "r", "model": "m", "effort": "high", "mode": "agent", "persist": 1, "tasks": ["a"],
                  "started_at": "2026-11-01T00:00:00Z"},
    "task_start": {"task": "a", "train": [{"input": G, "output": G}], "test": [{"input": G}], "answers": [G]},
    "turn": {"task": "a", "att": 0, "turn": 1, "prompt_tokens": 1, "completion_tokens": 2, "reasoning_tokens": 1,
             "finish": "tool_calls", "dur_s": 0.5, "reasoning": "r", "content": "", "tool_calls": [CALL]},
    "tool_result": {"task": "a", "att": 0, "turn": 1, "name": "submit", "text": "ok", "dur_s": 0.1},
    "submit": {"task": "a", "att": 0, "turn": 1, "program": "x", "reproduced": 0, "n_train": 1, "verified": False,
               "outputs": [G, None], "wrong": [{"ex": 1, "total": 2, "cells": [[0, 0, 1, 2]]}], "flags": ["f"],
               "test_outputs": [None]},
    "rescue": {"task": "a", "att": 0, "turn": 2, "reason": "cut"},
    "attempt_end": {"task": "a", "att": 0, "status": "verified", "verified": True, "tokens": 5, "turns": 2,
                    "python_calls": 1, "submits": 1, "wall_s": 1.0},
    "task_end": {"task": "a", "rows": [{"attempt_1": G, "attempt_2": None}], "verified": 1, "solved": None},
    "heartbeat": {"tok_s": 10.0, "running": 1, "waiting": None, "kv_pct": None, "gpu_mem_mib": None},
    "run_end": {"summary": {"tasks": 1, "verified": 1, "solved_test_inputs": None, "tokens": 5, "tok_s": 1.0,
                            "wall_s": 2}},
}


def ev(_n, **over):
    return {"ev": _n, **BASE, **copy.deepcopy(VALID[_n]), **over}


def test_every_event_type_valid_and_extra_fields_ignored():
    for name in VALID:
        assert tr.validate_event(ev(name)) == [], name
    assert tr.validate_event(ev("turn", future_field={"x": 1})) == []
    assert tr.validate_event(ev("task_start", answers=None)) == []
    e = ev("task_start")
    del e["answers"]  # only present when solutions are known
    assert tr.validate_event(e) == []


def test_missing_wrong_type_and_unknown():
    e = ev("turn")
    del e["dur_s"]
    assert tr.validate_event(e) == ["turn: missing dur_s"]
    assert tr.validate_event(ev("turn", att="0")) == ["turn: bad att"]
    assert tr.validate_event(ev("turn", att=True)) == ["turn: bad att"]
    assert tr.validate_event(ev("turn", finish="weird")) == ["turn: bad finish"]
    assert tr.validate_event(ev("submit", outputs=[[[10]]])) == ["submit: bad outputs"]
    assert tr.validate_event(ev("submit", outputs=[[[1], [1, 2]]])) == ["submit: bad outputs"]
    assert tr.validate_event(ev("submit", outputs=[[[0] * 31]])) == ["submit: bad outputs"]
    assert tr.validate_event(ev("submit", outputs=[[[0] * 30] * 30])) == []
    assert tr.validate_event(ev("tool_result", name="bash")) == ["tool_result: bad name"]
    assert tr.validate_event({"ev": "nope", **BASE}) == ["unknown ev 'nope'"]
    assert tr.validate_event([1]) == ["event is not an object"]
    e = ev("run_end")
    del e["t"], e["cfg"]
    assert tr.validate_event(e) == ["run_end: missing t", "run_end: missing cfg"]


def test_capping_and_cut_markers(tmp_path):
    p = tmp_path / "t.jsonl"
    w = tr.TraceWriter(str(p))
    w.emit("turn", reasoning="a" * 5000 + "Z" * 5000, content="c" * 2500,
           tool_calls=[{"id": "1", "name": "python", "args": {"code": "x" * 7000}}])
    w.emit("tool_result", text="t" * 4001)
    w.emit("submit", program="p" * 6000)
    a, b, c = (json.loads(ln) for ln in p.read_text().splitlines())
    assert a["content"] == "c" * 2000 + "…[cut 500]"
    r = a["reasoning"]
    assert r.startswith("a" * 2000 + "…[cut 6000]") and r.endswith("Z" * 2000) and len(r) < 4100
    assert a["tool_calls"][0]["args"]["code"] == "x" * 6000 + "…[cut 1000]"
    assert b["text"] == "t" * 4000 + "…[cut 1]"
    assert c["program"] == "p" * 6000  # at the cap: untouched
    assert a["ev"] == "turn" and a["cfg"] == "default" and a["t"] == 0.0 and b["t"] >= a["t"]


def test_writer_cfg_flush_and_never_raises(tmp_path, capsys):
    p = tmp_path / "t.jsonl"
    w = tr.TraceWriter(str(p), cfg="bench1", run_id="rid")
    w.emit("heartbeat", tok_s=1)
    assert json.loads(p.read_text())["cfg"] == "bench1" and w.run_id == "rid"  # flushed without close
    bad = tr.TraceWriter(str(tmp_path / "no/such/dir/t.jsonl"))
    bad.emit("heartbeat")
    bad.emit("heartbeat")
    assert capsys.readouterr().err.count("trace: write") == 1


def test_thread_safe_interleaving(tmp_path):
    p = tmp_path / "t.jsonl"
    w = tr.TraceWriter(str(p))

    def work(i):
        for n in range(50):
            w.emit("tool_result", task=f"t{i}", att=0, turn=n, name="python", text=f"{i}-{n}\n" * 200, dur_s=0.0)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lines = p.read_text().splitlines()
    rows = [json.loads(ln) for ln in lines]  # every line parses
    assert len(rows) == 800
    assert [r["t"] for r in rows] == sorted(r["t"] for r in rows)


def _write(path, events, tail=""):
    path.write_text("".join(json.dumps(e) + "\n" for e in events) + tail)
    return str(path)


def test_validate_file_ok_and_truncated_last_line(tmp_path):
    evs = [ev("run_start"), ev("turn"), ev("submit"), ev("tool_result")]
    n, errs = tr.validate_file(_write(tmp_path / "a.jsonl", evs))
    assert (n, errs) == (4, [])
    n, errs = tr.validate_file(_write(tmp_path / "b.jsonl", evs, tail='{"ev": "turn", "t": 1.0, "cf'))
    assert (n, errs) == (4, [])
    n, errs = tr.validate_file(_write(tmp_path / "c.jsonl", evs, tail='{"bad\n'))  # complete line: a real error
    assert n == 4 and errs == ["line 5: not valid JSON"]


def test_validate_file_ordering_rules(tmp_path):
    _, errs = tr.validate_file(_write(tmp_path / "a.jsonl", [ev("turn"), ev("run_start")]))
    assert errs == ["line 1: first event is 'turn', not run_start"]
    _, errs = tr.validate_file(_write(tmp_path / "b.jsonl", [ev("run_start"), ev("tool_result")]))
    assert errs == ["line 2: tool_result without a preceding turn"]
    _, errs = tr.validate_file(_write(tmp_path / "c.jsonl", [ev("run_start"), ev("turn"), ev("tool_result"),
                                                              ev("submit")]))
    assert errs == ["line 4: submit after its tool_result"]
    _, errs = tr.validate_file(_write(tmp_path / "d.jsonl", [ev("run_start"), ev("run_end"), ev("turn")]))
    assert errs == ["line 3: 'turn' after run_end"]
    other = ev("turn", task="b")  # a tool_result matches its own task's turn, not another's
    _, errs = tr.validate_file(_write(tmp_path / "e.jsonl", [ev("run_start"), other, ev("tool_result")]))
    assert errs == ["line 3: tool_result without a preceding turn"]
