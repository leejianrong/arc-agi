import json
import sys
import time
import urllib.error
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_kaggle_agent as ta

sa, sy = ta.sa, ta.sy
tr = sa.tr


def _run(tmp_path, monkeypatch, replies, tasks=None, *extra):
    monkeypatch.setattr(sa, "_default_repl", ta.StubRepl)
    srv, _ = ta.server(replies)
    tasks = tasks or {"t1": ta.TASK}
    ch, sol = tmp_path / "c.json", tmp_path / "s.json"
    ch.write_text(json.dumps(tasks))
    sol.write_text(json.dumps({tid: [ta.ANS] for tid in tasks}))
    tp = tmp_path / "trace.jsonl"
    try:
        summary = sa.main(["--challenges", str(ch), "--out", str(tmp_path / "o.json"), "--solutions", str(sol),
                           "--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--log-dir", str(tmp_path / "logs"),
                           "--trace", str(tp), *extra])
    finally:
        srv.shutdown()
    return summary, tp


def _events(tp):
    return [json.loads(ln) for ln in tp.read_text().splitlines()]


def test_scripted_run_trace_is_valid_with_expected_sequence(tmp_path, monkeypatch):
    replies = [ta.call_reply(ta.tc("python", "print(1)")), ta.call_reply(ta.tc("submit", ta.WRONG)),
               ta.call_reply(ta.tc("submit", ta.GOOD))]
    summary, tp = _run(tmp_path, monkeypatch, replies, None, "--n-parallel", "1")
    n, errs = tr.validate_file(str(tp))
    assert errs == [] and summary["solved_test_inputs"] == "1/1"
    evs = _events(tp)
    assert n == len(evs)
    assert [e["ev"] for e in evs] == [
        "run_start", "task_start", "turn", "tool_result", "turn", "submit", "tool_result", "turn", "submit",
        "tool_result", "attempt_end", "task_end", "run_end"]
    assert all(e["cfg"] == "default" for e in evs) and [e["t"] for e in evs] == sorted(e["t"] for e in evs)
    assert evs[0]["tasks"] == ["t1"] and evs[0]["mode"] == "agent" and evs[1]["answers"] == [ta.ANS]
    bad, good = (e for e in evs if e["ev"] == "submit")
    assert bad["turn"] == 2 and bad["reproduced"] == 0 and not bad["verified"] and bad["n_train"] == 2
    assert bad["outputs"] == [[[1, 2], [3, 4]], [[5, 0], [0, 6]]]
    assert [w["ex"] for w in bad["wrong"]] == [1, 2] and bad["wrong"][0] == {
        "ex": 1, "total": 4, "cells": [[0, 0, 1, 4], [0, 1, 2, 3], [1, 0, 3, 2], [1, 1, 4, 1]]}
    assert good["verified"] and good["wrong"] == [] and good["test_outputs"] == [ta.ANS]
    turn = evs[2]
    assert turn["tool_calls"] == [{"id": "c_python", "name": "python", "args": {"code": "print(1)"}}]
    assert turn["completion_tokens"] == 50 and turn["finish"] == "tool_calls"
    assert evs[3]["text"] == "out:print(1)" and evs[3]["name"] == "python"
    end = next(e for e in evs if e["ev"] == "attempt_end")
    assert (end["status"], end["verified"], end["turns"], end["python_calls"], end["submits"]) == (
        "verified", True, 3, 1, 2)
    assert evs[-2]["solved"] == [True] and evs[-2]["verified"] == 1 and evs[-2]["rows"][0]["attempt_1"] == ta.ANS
    assert evs[-1]["summary"]["tasks"] == 1 and evs[-1]["summary"]["solved_test_inputs"] == "1/1"


def test_parallel_tasks_interleave_validly(tmp_path, monkeypatch):
    tasks = {f"t{i}": ta.TASK for i in range(4)}
    _, tp = _run(tmp_path, monkeypatch, [ta.call_reply(ta.tc("python", "1")), ta.call_reply(ta.tc("submit", ta.GOOD))],
                 tasks, "--n-parallel", "4")
    _, errs = tr.validate_file(str(tp))
    assert errs == []
    c = Counter(e["ev"] for e in _events(tp))
    assert c["task_start"] == c["task_end"] == c["attempt_end"] == 4 and c["run_start"] == c["run_end"] == 1
    assert {e["task"] for e in _events(tp) if e["ev"] == "submit"} <= set(tasks)


def test_dead_server_still_yields_valid_file(tmp_path, monkeypatch):
    real, calls = ta.ls.chat_full, []

    def dying(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise urllib.error.URLError("connection refused")
        return real(*a, **k)

    monkeypatch.setattr(ta.ls, "chat_full", dying)
    _, tp = _run(tmp_path, monkeypatch, [ta.call_reply(ta.tc("python", "1"))])
    _, errs = tr.validate_file(str(tp))
    assert errs == []
    evs = _events(tp)
    assert [e["finish"] for e in evs if e["ev"] == "turn"] == ["tool_calls", "error"]
    assert next(e for e in evs if e["ev"] == "attempt_end")["status"] == "error"
    # a hard crash leaves no run_end and possibly a cut line; the validator still accepts it
    cut = tmp_path / "cut.jsonl"
    cut.write_text("".join(ln + "\n" for ln in tp.read_text().splitlines()[:-1]) + '{"ev": "run_en')
    assert tr.validate_file(str(cut))[1] == []


def test_deadline_attempt_end_and_rescue_event(tmp_path):
    tp = tmp_path / "t.jsonl"
    w = tr.TraceWriter(str(tp))
    srv, _ = ta.server([ta.prose("thinking out loud", finish="length"), ta.call_reply(ta.tc("submit", ta.GOOD))])
    try:
        att = sa.run_attempt(ta.TASK, base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="m",
                             budget=sy.Budget(time.time() + 600, 1, 1), deadline=time.time() + 600,
                             repl_factory=ta.StubRepl, max_hints=0, trace=w, task_id="t1")
        sa.run_attempt(ta.TASK, base_url="http://x/v1", model="m", budget=sy.Budget(time.time() - 1, 1, 1),
                       deadline=time.time() - 1, repl_factory=ta.StubRepl, trace=w, task_id="t1", att_idx=1)
    finally:
        srv.shutdown()
    evs = _events(tp)
    assert [e["ev"] for e in evs] == ["turn", "rescue", "turn", "submit", "attempt_end", "attempt_end"]
    assert att.verified and evs[1]["reason"] == "cut" and evs[1]["turn"] == 2
    assert [(e["att"], e["status"]) for e in evs if e["ev"] == "attempt_end"] == [(0, "verified"), (1, "deadline")]


def test_without_trace_no_file_and_no_events(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "_default_repl", ta.StubRepl)
    srv, _ = ta.server([ta.call_reply(ta.tc("submit", ta.GOOD))])
    summary, out = ta._main(tmp_path, srv)
    srv.shutdown()
    assert summary["verified"] == 1 and out["t1"][0]["attempt_1"] == ta.ANS
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c.json", "logs", "o.json", "s.json"]


def test_cfg_flag_labels_every_event(tmp_path, monkeypatch):
    _, tp = _run(tmp_path, monkeypatch, [ta.call_reply(ta.tc("submit", ta.GOOD))], None, "--cfg", "agent_xhigh_fresh")
    assert {e["cfg"] for e in _events(tp)} == {"agent_xhigh_fresh"}
