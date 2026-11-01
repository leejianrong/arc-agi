"""Scripted player: a deterministic fake model that drives the REAL agent loop and the REAL sandboxed REPL.

Why it exists: a Haiku answerer only exercises the harness as far as its own skill reaches (it gave up on hard tasks, so
the persistence, progress-block and trimming paths never ran). A scripted player forces each path on purpose, reads the
harness's actual replies, and checks invariants on what the harness sent. It says nothing about Qwen's accuracy; it
checks that the tooling and memory mechanisms behave as designed.

    uv run python kaggle/arc2_synth/scripted_player.py            # all scenarios, exit 1 on any broken invariant
    uv run python kaggle/arc2_synth/scripted_player.py plateau    # one scenario
    uv run python kaggle/arc2_synth/scripted_player.py --json     # machine-readable report

Each scenario is ``(policy, run options, checker)``. A policy maps (request index, messages) to one assistant action:
``("python", code)``, ``("submit", code)`` or ``("text", prose)``. A checker returns a list of failure strings.
"""
import argparse
import itertools
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repl as repl_mod
import synth as sy
import synth_agent as sa

# ---- the task: output is the input transposed (6x6), so a program can be wrong by a chosen number of cells ----

N = 6


def _grid(seed):
    return [[(seed * 7 + r * 3 + c * 5) % 9 + 1 for c in range(N)] for r in range(N)]


def _t(g):
    return [list(r) for r in zip(*g)]


TASK = {"train": [{"input": _grid(s), "output": _t(_grid(s))} for s in (1, 2, 3)], "test": [{"input": _grid(4)}]}


def program(wrong):
    """A transpose that zeroes `wrong` cells of the first row (a distinct program for each count)."""
    return ("def transform(grid):\n    out = [list(r) for r in zip(*grid)]\n"
            f"    for c in range({wrong}):\n        out[0][c] = 0\n    return out\n")


GOOD = program(0)

# ---- fake server --------------------------------------------------------------------------------------------------


def serve(policy):
    """OpenAI-style chat server answering with ``policy``; returns (server, requests-seen list)."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(json.loads(json.dumps(req)))  # snapshot: the agent mutates its list in place
            kind, payload, *delay = policy(len(seen) - 1, req["messages"])
            time.sleep(delay[0] if delay else 0)
            if kind == "text":
                msg = {"content": payload, "tool_calls": None}
            else:
                msg = {"content": None, "tool_calls": [{"id": f"call_{len(seen)}", "type": "function",
                                                        "function": {"name": kind, "arguments": json.dumps({"code": payload})}}]}
            body = json.dumps({"choices": [{"message": msg, "finish_reason": "stop" if kind == "text" else "tool_calls"}],
                               "usage": {"completion_tokens": 40}}).encode()
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


def run_scenario(policy, persist=1, control=None, max_turns=40, sandbox_timeout=10, time_limit=300, min_call=2048):
    srv, seen = serve(policy)
    try:
        att = sa.run_attempt(TASK, base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="scripted",
                             budget=sy.Budget(time.time() + time_limit, 1, 1, reserve_sec=1, min_call_tokens=min_call),
                             deadline=time.time() + time_limit, max_turns=max_turns,
                             persist=persist, control=control, tool_timeout=sandbox_timeout, sandbox_timeout=sandbox_timeout)
    finally:
        srv.shutdown()
    return att, seen

# ---- what the harness sent: helpers for checkers -----------------------------------------------------------------


def tool_replies(msgs, name=None):
    """Text of tool replies in order (``name`` filters by the call's tool name)."""
    calls = {c["id"]: c["function"]["name"] for m in msgs for c in (m.get("tool_calls") or [])}
    return [m["content"] for m in msgs if m["role"] == "tool" and (name is None or calls.get(m["tool_call_id"]) == name)]


def last_replies(seen, name=None):
    return tool_replies(seen[-1]["messages"], name)


def orphans(msgs):
    """Tool messages whose assistant call is missing, or assistant calls with no reply (bad after trimming)."""
    out, have = [], set()
    for m in msgs:
        for c in m.get("tool_calls") or []:
            have.add(c["id"])
        if m["role"] == "tool" and m["tool_call_id"] not in have:
            out.append(m["tool_call_id"])
    return out


def prefix_breaks(seen):
    """Number of consecutive request pairs whose shared head differs outside a trim (a trim rewrites index 2 onward).
    Between trims every request must extend the previous one byte for byte."""
    bad = 0
    for a, b in itertools.pairwise(seen):
        ma, mb = a["messages"], b["messages"]
        k = min(len(ma), len(mb))
        if ma[:k] == mb[:k]:
            continue
        trimmed = len(mb) < len(ma) or any("Older turns were removed" in str(m.get("content")) for m in mb[:4])
        if not trimmed or ma[:2] != mb[:2]:
            bad += 1
    return bad

# ---- scenarios ---------------------------------------------------------------------------------------------------


def _seq(*steps):
    """Fixed script; the last step repeats."""
    return lambda i, msgs: steps[min(i, len(steps) - 1)]


def sc_plateau():
    # best is #1 (2 wrong cells); #2..#5 are all worse and distinct -> comparison text, then the hint exactly once
    pol = _seq(("submit", program(2)), ("submit", program(4)), ("submit", program(5)), ("submit", program(6)),
               ("submit", program(7)), ("python", "print(len(submissions), [s['n'] for s in submissions])"),
               ("submit", GOOD))

    def check(att, seen):
        f = []
        r = last_replies(seen, "submit")
        if not any("is worse than your best" in x for x in r):
            f.append("no 'worse than your best' comparison after a regression")
        hints = sum("did not beat #1" in x for x in r)
        if hints != 1:
            f.append(f"plateau hint shown {hints} times, expected exactly once")
        if not any("Best: #1" in x for x in r):
            f.append("progress block never named #1 as best")
        py = tool_replies(seen[-1]["messages"], "python")
        if not py or "5 [1, 2, 3, 4, 5]" not in py[-1]:
            f.append(f"`submissions` not visible in the REPL as expected: {py[-1:]}")
        if att.status != "verified":
            f.append(f"expected the final GOOD submit to verify, status={att.status}")
        return f
    return pol, {}, check


def sc_repeat():
    pol = _seq(("submit", program(3)), ("submit", program(3)), ("submit", program(3)), ("submit", GOOD))

    def check(att, seen):
        r = last_replies(seen, "submit")
        f = []
        if sum("identical to your submission #1" in x for x in r) != 2:
            f.append("each identical resubmit should be flagged as identical to #1")
        if att.submits != 4 or len(att.cands) != 2:
            f.append(f"identical resubmits must not create candidates: submits={att.submits} cands={len(att.cands)}")
        return f
    return pol, {}, check


def sc_persist(persist):
    pol = _seq(("python", "x = 41"), ("python", "print(x + 1)"), ("submit", GOOD))

    def check(att, seen):
        out = tool_replies(seen[-1]["messages"], "python")
        f = []
        if persist and out[-1].strip() != "42":
            f.append(f"variable did not survive between python calls with persist=1: {out[-1]!r}")
        if not persist and "NameError" not in out[-1]:
            f.append(f"variable survived with persist=0: {out[-1]!r}")
        return f
    return pol, {"persist": int(persist)}, check


def sc_survives_failures():
    """`submissions` must outlive a timeout kill, and the REPL must recover for the next call."""
    pol = _seq(("submit", program(2)), ("python", "while True: pass"), ("python", "print(len(submissions))"),
               ("submit", GOOD))

    def check(att, seen):
        py = tool_replies(seen[-1]["messages"], "python")
        f = []
        if not py or "timed out" not in py[0]:
            f.append(f"expected a timeout notice, got {py[:1]}")
        if len(py) < 2 or py[1].strip() != "1":
            f.append(f"`submissions` lost after a timeout kill: {py[1:2]}")
        return f
    return pol, {"sandbox_timeout": 2}, check


def sc_flood():
    pol = _seq(("python", "print('x' * 2_000_000)"), ("python", "print('ok')"), ("submit", GOOD))

    def check(att, seen):
        py = tool_replies(seen[-1]["messages"], "python")
        f = []
        if len(py[0]) > repl_mod.MAX_OUT + 400:
            f.append(f"flooded REPL reply not capped: {len(py[0])} chars (cap {repl_mod.MAX_OUT})")
        if "ok" not in py[1]:
            f.append("REPL unusable after a flood")
        return f
    return pol, {}, check


def sc_nudge():
    pol = _seq(*([("python", f"print({i})") for i in range(7)] + [("submit", GOOD)]))

    def check(att, seen):
        py = tool_replies(seen[-1]["messages"], "python")
        f = []
        if "6 python calls without a submit" not in py[5]:
            f.append("nudge missing on the sixth python call")
        if any("python calls without a submit" in x for x in py[:5]):
            f.append("nudge shown too early")
        return f
    return pol, {}, check


def sc_long(persist=1):
    """Many bulky python replies under a tiny context budget: trims happen, the head stays stable, no orphans."""
    steps = [("python", f"print('{i} ' + 'data ' * 700)") for i in range(30)] + [("submit", GOOD)]
    ctl = sa.Control(trim_budget=9000, trim_drain=3000, keep_recent=6, nudge_every=0)

    def check(att, seen):
        f = []
        if not att.events.get("trims"):
            f.append("no trim happened under a 9k budget")
        if prefix_breaks(seen):
            f.append(f"{prefix_breaks(seen)} request pairs changed their shared head outside a trim")
        for i, r in enumerate(seen):
            if orphans(r["messages"]):
                f.append(f"request {i} has orphaned tool messages")
                break
        notes = [m["content"] for m in seen[-1]["messages"] if "Older turns were removed" in str(m.get("content"))]
        if len(notes) != 1:
            f.append(f"expected exactly one removal note in the final request, found {len(notes)}")
        elif not persist and "variables persist" in notes[0]:
            f.append("removal note claims python variables persist although persist=0")
        if att.status != "verified":
            f.append(f"attempt did not finish verified: {att.status}")
        return f
    return _seq(*steps), {"persist": persist, "control": ctl}, check


def _slow_explorer(trigger, answer):
    """Explores with slow python turns (the model 'thinks' for a while) until the harness asks for a submit."""
    def pol(i, msgs):
        last = str(msgs[-1].get("content"))
        if trigger in last:
            return ("submit", answer)
        return ("python", f"print({i})", 2.5)
    return pol


def sc_forced_submit():
    """No submit by half of the time share: the harness asks for one while there is still time to refine."""
    pol = _slow_explorer("Time is running out", program(1))

    def check(att, seen):
        f = []
        if att.events.get("force") != "time":
            f.append(f"no forced-submit turn: events={att.events}")
        if att.submits < 1:
            f.append("attempt ended with zero submits despite the forced-submit rule")
        if not any("Time is running out" in str(m.get("content")) for m in seen[-1]["messages"]):
            f.append("the forced-submit prompt never reached the model")
        return f
    return pol, {"time_limit": 14, "min_call": 64, "control": sa.Control(rescue_min_tokens=16, nudge_every=0)}, check


def sc_time_exhausted_rescue():
    """With the forced-submit rule off, running out of time still ends with a rescue turn that can submit."""
    pol = _slow_explorer("Out of thinking budget", program(1))

    def check(att, seen):
        f = []
        if att.events.get("rescue") != "time":
            f.append(f"no rescue turn when the time share ran out: events={att.events}, status={att.status}")
        if att.submits < 1:
            f.append("attempt ended with zero submits after time ran out")
        return f
    return pol, {"time_limit": 14, "min_call": 64, "control": sa.Control(submit_frac=0, nudge_every=0, rescue_min_tokens=16)}, check


SCENARIOS = {
    "plateau": sc_plateau,
    "repeat": sc_repeat,
    "persist_on": lambda: sc_persist(True),
    "persist_off": lambda: sc_persist(False),
    "survives_failures": sc_survives_failures,
    "flood": sc_flood,
    "nudge": sc_nudge,
    "long_trim": sc_long,
    "long_trim_no_persist": lambda: sc_long(0),
    "forced_submit": sc_forced_submit,
    "time_exhausted_rescue": sc_time_exhausted_rescue,
}


def run_one(name):
    pol, opts, check = SCENARIOS[name]()
    t0 = time.time()
    att, seen = run_scenario(pol, **opts)
    fails = check(att, seen)
    return {"scenario": name, "ok": not fails, "failures": fails, "turns": len(seen), "status": att.status,
            "trims": att.events.get("trims", 0), "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenarios", nargs="*", help=f"any of: {', '.join(SCENARIOS)} (default all)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    names = a.scenarios or list(SCENARIOS)
    rows = [run_one(n) for n in names]
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['scenario']:<22} turns={r['turns']:<3} trims={r['trims']} "
                  f"status={r['status']} {r['seconds']}s")
            for x in r["failures"]:
                print(f"      - {x}")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
