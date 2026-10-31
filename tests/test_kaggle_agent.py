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
    assert reply.startswith("Reproduced 0 of 2 examples.")
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
    assert "identical to a previous submission" in seen[2]["messages"][-1]["content"]
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
