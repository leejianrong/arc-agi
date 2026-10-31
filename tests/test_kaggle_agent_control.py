import importlib.util
import itertools
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_t = Path(__file__).resolve().parent / "test_kaggle_agent.py"
_spec = importlib.util.spec_from_file_location("kaggle_agent_helpers", _t)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
sa, sy, tc, call_reply, prose = base.sa, base.sy, base.tc, base.call_reply, base.prose
TASK, GOOD, WRONG, StubRepl = base.TASK, base.GOOD, base.WRONG, base.StubRepl


def server(replies):
    """Like the base fake server, plus per-reply `status` (an HTTP error body), `tokens` and `prompt`."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(req)
            r = replies[min(len(seen) - 1, len(replies) - 1)]
            if r.get("status"):
                body = json.dumps({"error": r["body"]}).encode()
                self.send_response(r["status"])
            else:
                out = {"choices": [{"message": {"content": r["content"], "tool_calls": r["tool_calls"] or None},
                                    "finish_reason": r["finish"]}],
                       "usage": {"completion_tokens": r.get("tokens", 50), "prompt_tokens": r.get("prompt", 0)}}
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


def attempt(replies, **kw):
    srv, seen = server(replies)
    StubRepl.made = []
    try:
        att = sa.run_attempt(TASK, base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="m",
                             budget=sy.Budget(time.time() + 600, 1, 1), deadline=time.time() + 600,
                             repl_factory=StubRepl, **kw)
    finally:
        srv.shutdown()
    return att, seen


def py(i, size=0):
    return call_reply(tc("python", f"# {i} " + "x" * size, cid=f"p{i}"))


def tool_texts(req):
    return [m["content"] for m in req["messages"] if m["role"] == "tool"]


def test_nudge_on_sixth_consecutive_python_only():
    att, seen = attempt([py(i) for i in range(7)] + [call_reply(tc("submit", GOOD))])
    t = tool_texts(seen[-1])
    assert [sa.NUDGE.format(n=6).strip() in x for x in t] == [False] * 5 + [True, False]
    assert att.events["nudges"] == 1 and att.log()["control"]["nudge_every"] == 6
    # a submit resets the counter: 4 + submit + 5 python calls never reach 6 in a row
    replies = [py(i) for i in range(4)] + [call_reply(tc("submit", WRONG))] + [py(i) for i in range(10, 15)]
    att, seen = attempt(replies + [call_reply(tc("submit", GOOD))])
    assert att.status == "verified" and "nudges" not in att.events
    assert not any("without a submit" in x for x in tool_texts(seen[-1]))
    att, seen = attempt([py(i) for i in range(7)] + [call_reply(tc("submit", GOOD))],
                        control=sa.Control(nudge_every=0))
    assert not any("without a submit" in x for x in tool_texts(seen[-1]))


def test_trim_prefix_stable_marker_hysteresis_no_orphans():
    ctl = sa.Control(nudge_every=0, trim_budget=4000, trim_drain=2000, keep_recent=4)
    replies = [call_reply(tc("submit", WRONG, cid="s0"))] + [py(i, 900) for i in range(1, 16)]
    att, seen = attempt(replies + [call_reply(tc("submit", GOOD))], control=ctl)
    assert att.status == "verified"
    evictions = 0
    for a, b in itertools.pairwise(seen):
        ma, mb = a["messages"], b["messages"]
        if mb[:len(ma)] == ma:
            continue  # no eviction between these requests: the previous prompt is a byte-identical prefix
        evictions += 1
        assert mb[:2] == ma[:2] and mb[2]["role"] == "user" and "removed to save context" in mb[2]["content"]
        assert mb[-2:] == ma[-2:] or mb[-1]["role"] == "tool"  # newest exchange retained
    assert evictions == att.events["trims"] and 1 <= evictions < len(seen) // 2
    for req in seen:
        ids = set()
        for m in req["messages"]:
            ids.update(c["id"] for c in m.get("tool_calls") or [])
            assert m["role"] != "tool" or m["tool_call_id"] in ids
    last = seen[-1]["messages"]
    assert last[0] == seen[0]["messages"][0] and last[1] == seen[0]["messages"][1]
    assert sum("removed to save context" in str(m["content"]) for m in last) == 1
    assert "1 failed submits" in last[2]["content"] and "Reproduced 0 of 2 examples." in last[2]["content"]
    assert "python calls" in last[2]["content"]


def test_trim_uses_prompt_tokens_and_keeps_recent():
    replies = [dict(py(i), prompt=30000 if i == 6 else 0) for i in range(8)]
    att, seen = attempt(replies + [call_reply(tc("submit", GOOD))], control=sa.Control(nudge_every=0))
    msgs = seen[-1]["messages"]
    assert att.events["trims"] >= 1 and "removed to save context" in msgs[2]["content"]
    assert len(msgs) - 3 >= 6


def test_cut_reply_gets_cap_hint_when_thinking_capped():
    _, seen = attempt([prose("cut", finish="length"), call_reply(tc("submit", GOOD))], thinking_budget=100)
    assert seen[1]["messages"][-1] == {"role": "user", "content": sa.CAP_HINT}


def test_rescue_turn_after_cut_and_on_token_exhaustion():
    cut = prose("cut", finish="length")
    att, seen = attempt([cut, cut, call_reply(tc("submit", GOOD))])
    assert att.status == "verified" and len(seen) == 3 and att.events["rescue"] == "cut"
    assert seen[2]["chat_template_kwargs"] == {"enable_thinking": False} and seen[2]["max_tokens"] == 8192
    assert seen[2]["messages"][-1]["content"] == sa.RESCUE_PROMPT
    assert "chat_template_kwargs" not in seen[1] or seen[1]["chat_template_kwargs"] != {"enable_thinking": False}
    att, seen = attempt([cut, cut, prose(f"```python\n{GOOD}\n```")])
    assert att.verified
    att, seen = attempt([cut, cut, call_reply(tc("submit", GOOD))], control=sa.Control(rescue=False))
    assert att.status == "prose" and len(seen) == 2
    att, seen = attempt([dict(py(1), tokens=2200), call_reply(tc("submit", GOOD))], attempt_tokens=3000)
    assert att.verified and att.events["rescue"] == "tokens" and len(seen) == 2
    assert seen[1]["chat_template_kwargs"] == {"enable_thinking": False}
    att, seen = attempt([prose("hmm"), prose("hmm")])  # a plain stop is not a cut: no rescue
    assert att.status == "prose" and len(seen) == 2 and "rescue" not in att.events


def test_context_overflow_shrinks_trims_and_retries_once():
    overflow = {"status": 400, "body": "This model's maximum context length is 8192 tokens"}
    ctl = sa.Control(keep_recent=0, nudge_every=0)
    att, seen = attempt([py(1, 30000), overflow, call_reply(tc("submit", GOOD))], control=ctl)
    assert att.status == "verified" and len(seen) == 3
    assert att.events["overflow_retries"] == 1 and att.events["trim_budget"] == 18000
    assert len(seen[1]["messages"]) == 4 and len(seen[2]["messages"]) == 3
    assert "removed to save context" in seen[2]["messages"][2]["content"]
    att, seen = attempt([py(1, 30000), overflow, overflow], control=ctl)
    assert att.status == "error" and len(seen) == 3 and att.events["overflow_retries"] == 1
    other = {"status": 400, "body": "bad request"}
    att, seen = attempt([other], control=ctl)
    assert att.status == "error" and len(seen) == 1


def test_repetition_guard_fires_once():
    att, seen = attempt([py(0)] * 5 + [call_reply(tc("submit", GOOD))])
    users = [m["content"] for m in seen[-1]["messages"] if m["role"] == "user"]
    assert users.count(sa.REPEAT_HINT) == 1 and att.events["repeat_guard"] == 1
    assert seen[3]["messages"][-1] == {"role": "user", "content": sa.REPEAT_HINT}
    att, seen = attempt([py(0)] * 5 + [call_reply(tc("submit", GOOD))], control=sa.Control(repeat_guard=False))
    assert not any(m["content"] == sa.REPEAT_HINT for m in seen[-1]["messages"] if m["role"] == "user")


def test_implicit_submit_and_markup_recovery_still_work():
    att, _ = attempt([prose(f"```python\n{WRONG}\n```"), prose(f"```python\n{GOOD}\n```")])
    assert att.verified and att.turns[0]["implicit_submit"]
    markup = f"<tool_call><function=submit><parameter=code>\n{GOOD}\n</parameter></function></tool_call>"
    att, _ = attempt([prose(markup)])
    assert att.status == "verified" and att.turns[0]["calls"] == ["submit"]
