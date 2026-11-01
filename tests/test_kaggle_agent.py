import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parent.parent
_dir = ROOT / "kaggle/arc2_synth"
sys.path.insert(0, str(_dir))
_spec = importlib.util.spec_from_file_location("synth_agent_under_test", _dir / "synth_agent.py")
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)
sy, ls = sa.sy, sa.ls

TASK = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
              {"input": [[5, 0], [0, 6]], "output": [[6, 0], [0, 5]]}],
    "test": [{"input": [[5, 6], [7, 8]]}],
}
GOOD = "def transform(grid):\n    return [row[::-1] for row in grid[::-1]]"
WRONG = "def transform(grid):\n    return grid"
ANS = [[8, 7], [6, 5]]


class StubRepl:
    made: ClassVar[list] = []

    def __init__(self, task, persist=True):
        self.persist, self.codes, self.closed = persist, [], False
        StubRepl.made.append(self)

    def run(self, code, timeout=None):
        self.codes.append(code)
        return f"out:{code}"

    def close(self):
        self.closed = True


def tc(name, code, form="str", cid=None):
    args = {"code": code}
    return {"id": cid or f"c_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args) if form == "str" else args}}


def call_reply(*calls, bare=False):
    return {"bare": bare, "content": None, "tool_calls": list(calls), "finish": "tool_calls"}


def prose(text, finish="stop"):
    return {"bare": False, "content": text, "tool_calls": [], "finish": finish}


def server(replies):
    """Replies served in order (last repeats); records each request body."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(req)
            r = replies[min(len(seen) - 1, len(replies) - 1)]
            msg = {"content": r["content"], "tool_calls": r["tool_calls"] or None}
            out = {"message": msg} if r["bare"] else {
                "choices": [{"message": msg, "finish_reason": r["finish"]}], "usage": {"completion_tokens": 50}}
            body = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def attempt(replies, tag="", **kw):
    srv, seen = server(replies)
    StubRepl.made = []
    budget = sy.Budget(time.time() + 600, 1, 1)
    try:
        att = sa.run_attempt(TASK, base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="m", budget=budget,
                             deadline=time.time() + 600, tag=tag, repl_factory=StubRepl, **kw)
    finally:
        srv.shutdown()
    return att, seen


def test_python_then_verified_submit_string_args():
    att, seen = attempt([call_reply(tc("python", "print(1)")), call_reply(tc("submit", GOOD))])
    assert att.status == "verified" and att.verified and len(seen) == 2
    first = seen[0]
    assert [t["function"]["name"] for t in first["tools"]] == ["python", "submit"] and first["tool_choice"] == "auto"
    assert first["messages"][0]["role"] == "system" and "## Example 1" in first["messages"][1]["content"]
    assert "Input (2x2):\n12\n34" in first["messages"][1]["content"]
    m = seen[1]["messages"]
    assert m[-2]["role"] == "assistant" and m[-2]["tool_calls"][0]["function"]["arguments"] == '{"code": "print(1)"}'
    assert m[-1] == {"role": "tool", "tool_call_id": "c_python", "content": "out:print(1)"}
    assert StubRepl.made[0].codes == ["print(1)"] and StubRepl.made[0].closed and StubRepl.made[0].persist
    assert att.cands[0]["test_outputs"] == [ANS]


def test_failed_submit_diff_then_fixed_object_args():
    att, seen = attempt([call_reply(tc("submit", WRONG, "obj")), call_reply(tc("submit", GOOD, "obj"))], tag="t1")
    assert att.status == "verified" and len(att.cands) == 2
    reply = seen[1]["messages"][-1]["content"]
    assert seen[1]["messages"][-1]["role"] == "tool"
    assert reply.startswith("[task t1] Reproduced 0 of 2 examples.")
    assert "## Example 1 failed" in reply and "## Example 2 failed" in reply
    assert "4 of 4 cells are wrong. Expected output:\n43\n21" in reply
    assert "Your output ('.' = correct cell, digit = wrong cell):\n12\n34" in reply
    assert "2 of 4 cells are wrong" in reply and "5.\n.6" in reply
    assert seen[0]["messages"][1]["content"].startswith("[task t1] ")


def test_implicit_submit_from_fenced_code():
    att, seen = attempt([prose(f"I think:\n```python\n{GOOD}\n```")], tag="x")
    assert att.status == "verified" and att.turns[0]["implicit_submit"]
    att, seen = attempt([prose(f"try:\n```python\n{WRONG}\n```"), prose(f"fixed:\n```python\n{GOOD}\n```")], tag="x")
    assert att.status == "verified"
    fb = seen[1]["messages"][-1]
    assert fb["role"] == "user" and fb["content"].startswith("[task x] Reproduced 0 of 2 examples.")


def test_prose_only_gets_one_hint_then_stops():
    att, seen = attempt([prose("let me think"), call_reply(tc("submit", GOOD))])
    assert att.status == "verified"
    assert seen[1]["messages"][-1] == {"role": "user", "content": sa.PROSE_HINT}
    att, seen = attempt([prose("hmm"), prose("hmm again")])
    assert att.status == "prose" and len(seen) == 2
    att, seen = attempt([prose("cut", finish="length"), call_reply(tc("submit", GOOD))])
    assert seen[1]["messages"][-1]["content"] == sa.CUT_HINT


def test_identical_resubmit_flagged_not_double_counted():
    att, seen = attempt([call_reply(tc("submit", WRONG)), call_reply(tc("submit", WRONG)),
                         call_reply(tc("submit", GOOD))])
    reply = seen[2]["messages"][-1]["content"]
    assert "identical to your submission #1; it was not re-run" in reply and "Reproduced 0 of 2" in reply
    assert len(att.cands) == 2 and att.status == "verified" and att.submits == 3


def test_max_turns_and_unknown_tool_and_multiple_calls():
    att, seen = attempt([call_reply(tc("python", "1", cid="a"), tc("nope", "x", cid="b"))], max_turns=3)
    assert att.status == "max_turns" and len(seen) == 3
    tools = [m for m in seen[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["a", "b"] and "Unknown tool 'nope'" in tools[1]["content"]


def test_bad_arguments_and_bare_message_shape():
    bad = {"id": "z", "type": "function", "function": {"name": "submit", "arguments": "{not json"}}
    att, seen = attempt([call_reply(bad, bare=True), call_reply(tc("submit", GOOD, "obj"), bare=True)])
    assert att.status == "verified"
    assert "Missing 'code' argument" in seen[1]["messages"][-1]["content"]
    assert att.turns[0]["tokens"] == 0  # bare body carries no usage


def test_chat_full_shapes():
    srv, seen = server([call_reply(tc("python", "p", "obj"))])
    r = ls.chat_full(f"http://127.0.0.1:{srv.server_port}/v1", "m", "hi", 10, 5, 0.1, None, sa.TOOLS, None, 99)
    srv.shutdown()
    assert r["tool_calls"] == [{"id": "c_python", "name": "python", "arguments": {"code": "p"}}]
    assert seen[0]["thinking_token_budget"] == 99 and seen[0]["tool_choice"] == "auto"
    assert ls._json_args('{"a": 1}') == {"a": 1} and ls._json_args(None) == {}
    calls, rest = ls.recover_markup_calls("hi <tool_call><function=submit><parameter=code>\nx=1\n</parameter></function></tool_call>")
    assert calls[0]["name"] == "submit" and calls[0]["arguments"] == {"code": "x=1"} and rest == "hi"


def _write(tmp_path, tasks):
    ch, sol = tmp_path / "c.json", tmp_path / "s.json"
    ch.write_text(json.dumps(tasks))
    sol.write_text(json.dumps({"t1": [ANS]}))
    return ch, sol


def _main(tmp_path, srv, *extra):
    ch, sol = _write(tmp_path, {"t1": TASK})
    summary = sa.main(["--challenges", str(ch), "--out", str(tmp_path / "o.json"), "--solutions", str(sol),
                       "--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--log-dir", str(tmp_path / "logs"),
                       "--n-parallel", "1", *extra])
    return summary, json.loads((tmp_path / "o.json").read_text())


def test_main_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "_default_repl", StubRepl)
    srv, _ = server([call_reply(tc("python", "print(2)")), call_reply(tc("submit", GOOD))])
    summary, out = _main(tmp_path, srv, "--tag-prompts", "1")
    srv.shutdown()
    assert summary["verified"] == 1 and summary["solved_test_inputs"] == "1/1"
    assert out["t1"][0]["attempt_1"] == ANS
    log = json.loads((tmp_path / "logs/t1.json").read_text())
    assert log["verified"] == 1 and log["attempts"][0]["status"] == "verified" and log["attempts"][0]["python_calls"] == 1


def test_deadline_falls_back_to_baseline(tmp_path):
    srv, seen = server([call_reply(tc("submit", GOOD))])
    summary, out = _main(tmp_path, srv, "--deadline-min", "0")
    srv.shutdown()
    assert not seen and summary["verified"] == 0
    assert out["t1"] == ls.load_fallback()(TASK)
    rows, log = sa.solve_task(TASK, type("A", (), {"attempts": 1})(), time.time() - 1, fallback_rows=out["t1"],
                              budget=sy.Budget(time.time() - 1, 1, 1))
    assert rows == out["t1"] and log["candidates"] == 0


def test_real_repl_integration_explore_then_verified_submit():
    """The loop drives the real sandboxed REPL: state set in one python call is visible in the next,
    preloaded helpers work, and the real tool output reaches the model before it submits."""
    srv, seen = server([
        call_reply(tc("python", "k = len(train)\nprint(show(train[0]['input']))")),
        call_reply(tc("python", "print(k, diff(train[0]['input'], train[0]['output']))")),
        call_reply(tc("submit", GOOD)),
    ])
    budget = sy.Budget(time.time() + 600, 1, 1)
    try:
        att = sa.run_attempt(TASK, base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="m", budget=budget,
                             deadline=time.time() + 600)  # default factory: the real repl.Repl
    finally:
        srv.shutdown()
    assert att.status == "verified" and att.python_calls == 2 and att.submits == 1
    tool_msgs = [m["content"] for m in seen[-1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0].strip() == "12\n34"  # show() output from the real worker
    assert tool_msgs[1].startswith("2 ") and "cells differ" in tool_msgs[1]  # `k` persisted; diff() ran


def test_tool_text_matches_real_namespace_and_numpy_wording():
    tools = {t["function"]["name"]: t["function"]["description"] for t in sa.TOOLS}
    py = tools["python"]
    assert "train_inputs" not in py and "train_outputs" not in py and "tuple" not in py
    for name in ("train", "test_inputs", "show(", "diff(", "objects(", "colors(", "crop(", "rot90(", "flip(", "transpose("):
        assert name in py
    assert "dunder" in py and "getattr" in py and "dotted" in py
    assert all(m in py for m in sa.rp.ALLOWED_MODULES if m != "numpy")
    assert ("np = numpy" in py) == sa.NUMPY and ("numpy" in sa.INSTRUCTIONS) == sa.NUMPY
    assert "separate plain Python process" in tools["submit"]
    # the names promised really exist in the worker, and the invented ones do not
    with sa.rp.Repl(TASK, timeout=10) as r:
        for n in ("train", "test_inputs", "show", "diff", "objects", "colors", "crop", "rot90", "flip", "transpose"):
            assert "NameError" not in r.run(n), n
        assert "NameError" in r.run("train_inputs")
        assert r.has_numpy == sa.NUMPY and ("NameError" not in r.run("np")) == sa.NUMPY


def test_tool_replies_carry_task_tag_and_trim_still_reads_headline():
    _, seen = attempt([call_reply(tc("python", "1")), call_reply(tc("submit", WRONG)),
                       call_reply(tc("submit", GOOD))], tag="abc")
    tools = [m["content"] for m in seen[-1]["messages"] if m["role"] == "tool"]
    assert len(tools) == 2 and all(t.startswith("[task abc] ") for t in tools)
    assert sa._fail_headline(tools[1]) == "Reproduced 0 of 2 examples."
    _, seen = attempt([call_reply(tc("python", "1")), call_reply(tc("submit", GOOD))])
    assert seen[1]["messages"][-1]["content"] == "out:1"  # no tag set: unchanged


def _grid(n, f):
    return [[f(r, c) for c in range(n)] for r in range(n)]


def test_submit_reply_shows_full_30x30_diff_and_caps_length():
    want = _grid(30, lambda r, c: (r + c) % 10)
    task = {"train": [{"input": want, "output": want}, {"input": want, "output": want}], "test": []}
    code = ("def transform(grid):\n    g = [row[:] for row in grid]\n    g[29][29] = 9 - g[29][29]\n"
            "    g[0][0] = 9 - g[0][0]\n    return g")
    cand = sy.evaluate(code, task)
    text = sa.submit_reply(cand, task)
    assert text.count("## Example") == 2 and "2 of 900 cells are wrong" in text
    assert text.count(sa.ls.grid_text(want)) == 2  # both examples' expected grids, all 30 rows
    assert sum(ln.startswith("." * 29) for ln in text.splitlines()) >= 2 and "reply truncated" not in text
    assert len(text) <= sa.REPLY_CAP
    # an oversized reply says how much was cut
    big = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
    cand = {"code": "", "verified": False, "frac": 0.0, "fails": ["Example 1: exception: " + "x" * 9000]}
    text = sa.submit_reply(cand, big)
    assert len(text) <= sa.REPLY_CAP and "chars cut" in text


def test_submit_reply_colour_pairs_and_wrong_cells(monkeypatch):
    want = _grid(30, lambda r, c: (r + c) % 10)
    task = {"train": [{"input": want, "output": want}], "test": []}
    cand = sy.evaluate("def transform(grid):\n    return [[7] * 30 for _ in grid]", task)
    text = sa.submit_reply(cand, task)
    assert "Colour pairs (got->expected): 7->" in text and text.count(" x") >= 5
    assert "Wrong cells: (0,0) got 7 expected 0; (0,1) got 7 expected 1" in text and len(text) <= sa.REPLY_CAP
    assert text.count("got 7 expected") == 10  # capped at 10 cells
    # over 900 cells (the sandbox caps grids at 30x30, so stub the run): count and pair summary only
    w = _grid(31, lambda r, c: 8)
    task = {"train": [{"input": w, "output": w}], "test": []}
    monkeypatch.setattr(sa.sy, "run_program", lambda code, ins: [{"status": "ok", "grid": _grid(31, lambda r, c: 1)}])
    cand = {"code": "", "verified": False, "frac": 0.0, "fails": ["Example 1: 961 of 961 cells wrong"]}
    text = sa.submit_reply(cand, task)
    assert "961 of 961 cells are wrong\nColour pairs (got->expected): 1->8 x961" in text and "Wrong cells" not in text
    monkeypatch.undo()
    # small example, exact text
    small = {"train": [{"input": [[1, 2]], "output": [[1, 3]]}], "test": []}
    cand = sy.evaluate("def transform(grid):\n    return [[1, 2]]", small)
    assert sa.submit_reply(cand, small) == (
        "Reproduced 0 of 1 examples.\n## Example 1 failed\n1 of 2 cells are wrong. Expected output:\n"
        f"{sa.ls.grid_text([[1, 3]])}\nYour output ('.' = correct cell, digit = wrong cell):\n.2\n"
        "Wrong cells: (0,1) got 2 expected 3\nColour pairs (got->expected): 2->3 x1\nFix the rule and submit again.")


def test_persist_flag_reaches_repl_factory_and_log():
    for flag, want in ((0, False), (1, True)):
        att, _ = attempt([call_reply(tc("python", "1")), call_reply(tc("submit", GOOD))], persist=flag)
        assert StubRepl.made[0].persist is want and att.log()["persist"] is want
    att, _ = attempt([call_reply(tc("submit", GOOD))])
    assert att.log()["persist"] is True


def test_persist_flag_real_repl_state_survives_only_when_persisting():
    for persist, kept in ((False, False), (True, True)):
        with sa.rp.Repl(TASK, persist=persist, timeout=10) as r:
            r.run("k = 5")
            assert ("NameError" not in r.run("print(k)")) is kept


def test_main_persist_flag_default_and_zero(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(sa, "_default_repl", lambda task, persist=True: seen.append(persist) or StubRepl(task, persist))
    for extra, want in (([], True), (["--persist", "0"], False)):
        srv, requests = server([call_reply(tc("python", "1")), call_reply(tc("submit", GOOD))])
        n = len(seen)
        summary, _ = _main(tmp_path, srv, *extra)
        srv.shutdown()
        assert requests and summary["verified"] == 1  # the model was actually called under the default reserve
        assert len(seen) == n + 1 and seen[-1] is want


# Suspect-submit loop: train outputs never contain 9, the test input does.
LEAK_TASK = {
    "train": [{"input": [[9, 1], [3, 4]], "output": [[4, 3], [1, 1]]},
              {"input": [[9, 5], [0, 6]], "output": [[6, 0], [5, 5]]}],
    "test": [{"input": [[9, 6], [7, 8]]}],
}
LEAK = ("def transform(grid):\n    if grid[0][0] == 9 and grid[0][1] in (1, 5):\n        return [[grid[1][1], grid[1][0]], "
        "[grid[0][1], grid[0][1]]]\n    return [[grid[1][1], grid[1][0]], [9, grid[0][1]]]")
CLEAN = LEAK + "\n# fixed\n" + "def transform(grid):\n    return [[grid[1][1], grid[1][0]], [grid[0][1], grid[0][1]]]"


def leak_attempt(replies, **kw):
    global TASK
    saved, TASK = TASK, LEAK_TASK
    try:
        return attempt(replies, **kw)
    finally:
        TASK = saved


def test_suspect_submit_continues_then_clean_submit_ends():
    att, seen = leak_attempt([call_reply(tc("submit", LEAK)), call_reply(tc("submit", CLEAN))])
    reply = seen[1]["messages"][-1]["content"]
    assert reply.startswith("Accepted on all 2 examples, but the output for test input 1 looks wrong: leaked colour 9")
    assert "\n87\n96\nFix the rule so it generalizes" in reply
    assert att.status == "verified" and len(att.cands) == 2 and att.cands[0]["verified"]
    assert att.suspect_rounds == 1 and att.submits == 2


def test_suspect_cap_identical_resubmit_and_off():
    two = [call_reply(tc("submit", LEAK)), call_reply(tc("submit", LEAK + "\n# v2"))]
    att, seen = leak_attempt(two, suspect_rounds=1)  # second suspect submit is accepted
    assert att.status == "verified" and att.submits == 2 and len(seen) == 2
    att, seen = leak_attempt(two, suspect_rounds=0)  # feature off: first submit ends it
    assert att.status == "verified" and att.submits == 1 and len(seen) == 1
    att, seen = leak_attempt([call_reply(tc("submit", LEAK)), call_reply(tc("submit", LEAK))])
    assert att.status == "verified" and att.submits == 2 and "identical to your submission #1" in \
        att.messages[-1]["content"]
    att, seen = leak_attempt([call_reply(tc("submit", LEAK)), call_reply(tc("submit", WRONG)),
                              call_reply(tc("submit", CLEAN))])  # a failing submit keeps the suspect pending
    assert att.status == "verified" and att.submits == 3 and att.verified


def test_select_attempts_unchanged_for_unflagged():
    cands = [{"verified": True, "frac": 1.0, "code": "x", "test_outputs": [ANS]}]
    fb = [{"attempt_1": [[0]], "attempt_2": [[0]]}]
    assert sy.select_attempts(TASK, cands, fb)[0]["attempt_1"] == ANS


# progress block in failing submit replies -------------------------------------------------------------

def entry(n, rep, w, of=2):
    return {"n": n, "reproduced": rep, "of": of, "wrong_cells": w, "code": f"c{n}"}


def test_progress_block_worsening_improving_equal():
    h = [entry(1, 1, 6), entry(2, 0, 33), entry(3, 1, 20)]
    text, gave = sa.progress_block(h)
    assert text == ("Submissions so far: #1 1 of 2 reproduced, 6 wrong cells; #2 0 of 2, 33 wrong; #3 1 of 2, 20 wrong. "
                    "Best: #1. This one (#3) is worse than your best.") and not gave
    text, _ = sa.progress_block([entry(1, 1, 6), entry(2, 1, 3)])
    assert text.endswith("Best: #2. This one (#2) is better than your previous best.")
    text, _ = sa.progress_block([entry(1, 1, 6), entry(2, 1, 6)])
    assert "Best: #1. This one (#2) is same as your best." in text
    assert "This one" not in sa.progress_block([entry(1, 1, 6)])[0]


def test_progress_block_hint_after_three_worse_once():
    h = [entry(1, 1, 6), entry(2, 0, 33), entry(3, 0, 30), entry(4, 1, 7)]
    text, gave = sa.progress_block(h)
    assert gave and text.endswith("Your last 3 attempts did not beat #1; consider building on #1's idea or trying "
                                  "a different approach.")
    text, gave = sa.progress_block(h + [entry(5, 0, 9)], hinted=True)
    assert not gave and "Your last 3" not in text
    assert not sa.progress_block(h[:3])[1]  # only two since the best
    assert not sa.progress_block([entry(1, 1, 6), entry(2, 0, 9), entry(3, 1, 2), entry(4, 0, 9)])[1]


def test_progress_block_keeps_best_and_last_five():
    h = [entry(i, 0, 100 + i) for i in range(1, 13)]
    h[2] = entry(3, 1, 4)  # the best is old
    text, _ = sa.progress_block(h)
    assert text.startswith("Submissions so far: #3 1 of 2 reproduced, 4 wrong cells; ...; #8 0 of 2, 108 wrong;")
    assert "#4" not in text and "#7 " not in text and text.count("#") >= 7 and "#12 0 of 2, 112 wrong" in text
    assert text.count("...") == 1


def test_reply_cap_holds_with_long_history_on_30x30():
    big = [[(r + c) % 10 for c in range(30)] for r in range(30)]
    task = {"train": [{"input": big, "output": [[0] * 30 for _ in range(30)]}] * 2, "test": [{"input": big}]}
    code = "def transform(grid):\n    return grid"
    cand = sy.evaluate(code, task)
    hist = [entry(i, 0, 800 + i) for i in range(1, 31)]
    prog, _ = sa.progress_block(hist)
    reply = sa.submit_reply(cand, task, progress=prog)
    assert len(reply) <= sa.REPLY_CAP and prog in reply and reply.endswith("Fix the rule and submit again.")
    assert reply.index(prog) < reply.index("Fix the rule")


def test_wrong_cells_counts_shape_errors_as_all_cells():
    cand = sy.evaluate("def transform(grid):\n    return [[0]]", TASK)
    assert sa.wrong_cells(cand, TASK) == 8
    assert sa.wrong_cells(sy.evaluate(WRONG, TASK), TASK) == 6


PARTIAL = "def transform(grid):\n    return [r[::-1] for r in grid[::-1]] if grid[0][0] == 5 else grid"


def test_agent_run_shows_progress_line_and_pushes_submissions():
    class PushRepl(StubRepl):
        pushed: ClassVar[list] = []

        def push(self, name, value):
            PushRepl.pushed.append((name, value))

    PushRepl.pushed = []
    srv, seen = server([call_reply(tc("python", "1")), call_reply(tc("submit", WRONG)),
                        call_reply(tc("submit", PARTIAL)), call_reply(tc("submit", GOOD))])
    budget = sy.Budget(time.time() + 600, 1, 1)
    try:
        att = sa.run_attempt(TASK, base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="m", budget=budget,
                             deadline=time.time() + 600, repl_factory=PushRepl)
    finally:
        srv.shutdown()
    assert att.verified
    replies = [m["content"] for m in seen[-1]["messages"] if m["role"] == "tool"]
    assert "Submissions so far: #1 0 of 2 reproduced, 6 wrong cells. Best: #1." in replies[1]
    assert ("#2 1 of 2, 4 wrong. Best: #2. This one (#2) is better than your previous best.") in replies[2]
    name, subs = PushRepl.pushed[-1]
    assert name == "submissions" and [s["n"] for s in subs] == [1, 2, 3]
    assert subs[1] == {"n": 2, "reproduced": 1, "of": 2, "wrong_cells": 4, "code": PARTIAL}
    assert subs[2]["reproduced"] == 2 and subs[2]["wrong_cells"] == 0


def test_submissions_capped_and_stub_without_push_still_works():
    srv, _ = server([call_reply(tc("python", "1")), call_reply(tc("submit", WRONG)), call_reply(tc("submit", GOOD))])
    budget = sy.Budget(time.time() + 600, 1, 1)
    try:  # StubRepl has no push
        att = sa.run_attempt(TASK, base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="m", budget=budget,
                             deadline=time.time() + 600, repl_factory=StubRepl)
    finally:
        srv.shutdown()
    assert att.verified
    pushed = []

    class R(StubRepl):
        def push(self, name, value):
            pushed.append(value)

    codes = [f"def transform(grid):\n    return [[{i}]]\n# {'x' * 7000}" for i in range(20)]
    srv, _ = server([call_reply(tc("python", "1"))] + [call_reply(tc("submit", c)) for c in codes])
    try:
        sa.run_attempt(TASK, base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="m", budget=budget,
                       deadline=time.time() + 600, repl_factory=R, max_turns=21)
    finally:
        srv.shutdown()
    last = pushed[-1]
    assert len(last) <= 13 and last[-1]["n"] == 20 and all(len(s["code"]) <= sa.CODE_CAP for s in last)
