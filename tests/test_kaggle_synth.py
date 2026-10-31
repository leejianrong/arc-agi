import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_dir = ROOT / "kaggle/arc2_synth"
sys.path.insert(0, str(_dir))
_spec = importlib.util.spec_from_file_location("synth", _dir / "synth.py")
sy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sy)

TASK = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
              {"input": [[5, 0], [0, 6]], "output": [[6, 0], [0, 5]]}],
    "test": [{"input": [[5, 6], [7, 8]]}],
}
GOOD = "def transform(grid):\n    return [row[::-1] for row in grid[::-1]]"
WRONG = "def transform(grid):\n    return grid"
ANS = [[8, 7], [6, 5]]


def test_sandbox_returns_grids_and_exceptions():
    r = sy.run_program("def transform(grid):\n    return [[c + 1 for c in row] for row in grid]", [[[1, 2]], [[9]]])
    assert r[0] == {"status": "ok", "grid": [[2, 3]]}
    assert r[1]["status"] == "invalid"  # 10 is not a color
    r = sy.run_program("def transform(grid):\n    return grid[5][5]", [[[1]]])
    assert r[0]["status"] == "exception" and "IndexError" in r[0]["error"]
    assert sy.run_program("def nope(): pass", [[[1]]])[0]["status"] == "exception"


def test_sandbox_rejects_malformed_output():
    for body in ("return 'hi'", "return [[1], [1, 2]]", "return []", "return [[1.5]]", "return [[0]] * 31"):
        assert sy.run_program(f"def transform(grid):\n    {body}", [[[1]]])[0]["status"] == "invalid", body
    r = sy.run_program("import sys\ndef transform(grid):\n    sys.stdout.write('junk')\n    return grid", [[[1]]])
    assert r[0] == {"status": "ok", "grid": [[1]]}  # prints are swallowed, not mixed into the result


def test_sandbox_kills_infinite_loop():
    t0 = time.time()
    r = sy.run_program("def transform(grid):\n    while True:\n        pass", [[[1]]], timeout_s=1)
    assert r[0]["status"] == "timeout" and time.time() - t0 < 5


def test_sandbox_contains_memory_bomb():
    t0 = time.time()
    r = sy.run_program("def transform(grid):\n    bytearray(10**11)\n    return grid", [[[1]]],
                       timeout_s=5, mem_mb=512)
    assert r[0]["status"] == "exception" and "MemoryError" in r[0]["error"] and time.time() - t0 < 5


def test_verified_and_feedback():
    assert sy.evaluate(GOOD, TASK)["verified"]
    c = sy.evaluate(WRONG, TASK)
    assert not c["verified"] and 0 < c["frac"] < 1
    fb = sy.feedback_text(c, 2)
    assert "passed 0 of 2" in fb
    assert "Example 1: 4 of 4 cells wrong; first: (0,0) got 1 expected 4" in fb
    shape = sy.evaluate("def transform(grid):\n    return [[1, 2, 3]]", TASK)
    assert "wrong shape: got 1x3, expected 2x2" in sy.feedback_text(shape, 2)
    boom = sy.evaluate("def transform(grid):\n    return 1 / 0", TASK)
    assert "ZeroDivisionError" in sy.feedback_text(boom, 2)


def _cand(out, verified=True, frac=1.0, code=""):
    return {"verified": verified, "frac": frac, "test_outputs": [out], "code": code}


def test_selection_majority_ties_partial_and_fallback():
    fb = [{"attempt_1": [[0]], "attempt_2": [[9]]}]
    a, b, c = [[1]], [[2]], [[3]]
    rows = sy.select_attempts(TASK, [_cand(a), _cand(b), _cand(b), _cand(c)], fb)
    assert rows == [{"attempt_1": b, "attempt_2": a}]  # majority first, then earlier round on ties
    rows = sy.select_attempts(TASK, [_cand(a, code="long program"), _cand(b, code="short")], fb)
    assert rows == [{"attempt_1": b, "attempt_2": a}]  # equal votes: shorter source wins
    rows = sy.select_attempts(TASK, [_cand(a, code="xx"), _cand(b, code="xx"), _cand(b, code="xxxxx")], fb)
    assert rows[0]["attempt_1"] == b  # more votes beat shorter code
    rows = sy.select_attempts(TASK, [_cand(a), _cand(c, False, 0.2), _cand(b, False, 0.8)], fb)
    assert rows == [{"attempt_1": a, "attempt_2": b}]  # best partial fills slot 2
    rows = sy.select_attempts(TASK, [_cand(a)], fb, direct_rows=[{"attempt_1": None, "attempt_2": c}])
    assert rows == [{"attempt_1": a, "attempt_2": c}]  # direct answer before baseline
    assert sy.select_attempts(TASK, [], fb) == fb


def _task(pairs, test_in):
    return {"train": [{"input": i, "output": o} for i, o in pairs], "test": [{"input": test_in}]}


def _flags(task, out):
    return sy.plausibility(task, {"test_outputs": [out]})[0]


def test_plausibility_shape_rules():
    same = _task([([[1, 2]], [[2, 1]]), ([[3], [4]], [[4], [3]])], [[1, 2, 3]])
    assert _flags(same, [[3, 2, 1]])[0] == 0
    n, why = _flags(same, [[1, 2]])
    assert n == 1 and "keep the input shape" in why[0]
    const = _task([([[1, 2]], [[5]]), ([[3], [4]], [[6]])], [[1, 2, 3]])
    assert _flags(const, [[7]])[0] == 0
    n, why = _flags(const, [[7, 7]])
    assert n == 1 and "share one shape" in why[0]
    tile = _task([([[1, 2]], [[1, 2, 1, 2]]), ([[3], [4]], [[3, 3], [4, 4]])], [[1], [2], [3]])  # width x2
    assert _flags(tile, [[1, 1], [2, 2], [3, 3]])[0] == 0
    assert "of the input" in _flags(tile, [[1], [2], [3]])[1][0]
    mixed = _task([([[1, 2]], [[1, 2, 1, 2]]), ([[3]], [[3, 3, 3]])], [[1, 2, 3]])  # no common factor
    assert _flags(mixed, [[1]])[0] == 0


def test_plausibility_identity_mono_palette():
    t = _task([([[1, 2]], [[2, 1]]), ([[3, 4]], [[4, 3]])], [[5, 6]])
    assert _flags(t, [[6, 5]])[0] == 0
    assert "equals its input" in _flags(t, [[5, 6]])[1][0]
    assert "one colour" in _flags(t, [[5, 5]])[1][0]
    assert "colour seen in no train" in _flags(t, [[6, 9]])[1][0]
    # rules stay quiet when the train set does not support them
    mono = _task([([[1, 2]], [[1, 1]]), ([[3, 4]], [[2, 3]])], [[5, 6]])
    assert _flags(mono, [[5, 5]])[0] == 0
    ident = _task([([[1, 2]], [[1, 2]]), ([[3, 4]], [[4, 3]])], [[5, 6]])
    assert _flags(ident, [[5, 6]])[0] == 0
    new = _task([([[1, 2]], [[7, 1]]), ([[3, 4]], [[3, 8]])], [[5, 6]])  # outputs add colours: palette rule off
    assert _flags(new, [[9, 5]])[0] == 0


def test_selection_overfit_ranks_below_general():
    fb = [{"attempt_1": [[0]], "attempt_2": [[9]]}]
    t = _task([([[1, 2]], [[2, 1]]), ([[3, 4]], [[4, 3]])], [[5, 6]])
    good, bad = [[6, 5]], [[5], [6]]  # bad has the wrong shape
    rows = sy.select_attempts(t, [_cand(bad, code="x"), _cand(good, code="xxxx"), _cand(good, code="xxxxx")], fb)
    assert rows == [{"attempt_1": good, "attempt_2": bad}]
    rows = sy.select_attempts(t, [_cand(bad, code="x"), _cand(good, code="xxxx")], fb)
    assert rows == [{"attempt_1": good, "attempt_2": bad}]  # fewer flags beat shorter source
    rows = sy.select_attempts(t, [_cand(bad)], fb)
    assert rows[0]["attempt_1"] == bad  # a lone suspect still beats partials and fallback
    a, b = [[5, 5]], [[6, 6]]  # both one-colour: equally suspect, so votes then source decide
    rows = sy.select_attempts(t, [_cand(a, code="x"), _cand(b, code="xx"), _cand(b, code="xx")], fb)
    assert rows[0]["attempt_1"] == b
    rows = sy.select_attempts(t, [_cand(a, code="xx"), _cand(b, code="x")], fb)
    assert rows[0]["attempt_1"] == b
    c = _cand(good)
    sy.select_attempts(t, [c], fb)
    assert c["flags"] == [[]]


def _server(replies, delay=0):
    """Chat replies are served in order (last repeats); records each request's messages."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(delay)
            if self.path.endswith("/chat/completions"):
                seen.append(req["messages"])
                content, reasoning, fin = replies[min(len(seen) - 1, len(replies) - 1)]
                out = {"choices": [{"message": {"content": content, "reasoning": reasoning}, "finish_reason": fin}],
                       "usage": {"completion_tokens": 100}}
            else:
                out = {"choices": [{"text": "    return [row[::-1] for row in grid[::-1]]\n"}],
                       "usage": {"completion_tokens": 9}}
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


def _run(tmp_path, srv, *extra):
    ch, sol = tmp_path / "c.json", tmp_path / "s.json"
    ch.write_text(json.dumps({"t1": TASK}))
    sol.write_text(json.dumps({"t1": [ANS]}))
    summary = sy.main(["--challenges", str(ch), "--out", str(tmp_path / "o.json"), "--solutions", str(sol),
                       "--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--log-dir", str(tmp_path / "logs"),
                       "--n-parallel", "1", "--chains", "1", *extra])
    return summary, json.loads((tmp_path / "o.json").read_text())


def _py(code):
    return f"thinking done\n```python\n{code}\n```"


def test_refine_loop_converges_on_round_two(tmp_path):
    srv, seen = _server([(_py(WRONG), "r1", "stop"), (_py(GOOD), "r2", "stop")])
    summary, out = _run(tmp_path, srv)
    srv.shutdown()
    assert summary["verified"] == 1 and summary["solved_test_inputs"] == "1/1"
    assert out["t1"][0]["attempt_1"] == ANS
    assert len(seen) == 2 and "passed 0 of 2" in seen[1][-1]["content"] and seen[1][1]["role"] == "assistant"
    log = json.loads((tmp_path / "logs/t1.json").read_text())
    assert [r["status"] for r in log["rounds"]] == ["failed", "verified"] and log["final_program"] == GOOD


def test_truncated_call_is_budget_forced_into_code(tmp_path):
    srv, _ = _server([("", "long thinking", "length")])
    summary, out = _run(tmp_path, srv)
    srv.shutdown()
    assert summary["verified"] == 1 and out["t1"][0]["attempt_1"] == ANS


def test_unverified_task_falls_back_without_dropping(tmp_path):
    srv, _ = _server([("no code at all", "", "stop")])
    summary, out = _run(tmp_path, srv, "--rounds", "2")
    srv.shutdown()
    assert summary["verified"] == 0 and set(out["t1"][0]) == {"attempt_1", "attempt_2"}


def test_deadline_respected(tmp_path):
    srv, seen = _server([(_py(WRONG), "", "stop")])
    _, out = _run(tmp_path, srv, "--deadline-min", "0")
    srv.shutdown()
    assert seen == [] and "t1" in out


def test_real_tasks_verify_with_handwritten_programs():
    data = ROOT / "third_party/ARC-AGI/data/training"
    programs = {
        "6150a2bd": "def transform(grid):\n    return [r[::-1] for r in grid[::-1]]",
        "68b16354": "def transform(grid):\n    return grid[::-1]",
        "a416b8f3": "def transform(grid):\n    return [r + r for r in grid]",
        "c8f0f002": "def transform(grid):\n    return [[5 if c == 7 else c for c in r] for r in grid]",
        "a79310a0": (
            "def transform(grid):\n    out = [[0] * len(grid[0]) for _ in grid]\n"
            "    for r in range(len(grid) - 1):\n        for c, v in enumerate(grid[r]):\n"
            "            if v == 8:\n                out[r + 1][c] = 2\n    return out"
        ),
    }
    for tid, code in programs.items():
        task = json.loads((data / f"{tid}.json").read_text())
        assert sy.evaluate(code, task)["verified"], tid
    task = json.loads((data / "68b16354.json").read_text())
    assert not sy.evaluate(programs["6150a2bd"], task)["verified"]


def test_feedback_shows_grids_color_pairs_and_flags_repeats():
    task = {"train": [{"input": [[1, 1], [1, 1]], "output": [[8, 8], [8, 1]]}], "test": [{"input": [[1]]}]}
    cand = sy.evaluate("def transform(grid):\n    return grid\n", task)
    assert cand["detail"][0] == 1
    text = sy.feedback_text(cand, 1, repeated=True)
    assert "identical to your previous one" in text
    assert "1->8 x3" in text and "expected output:\n88\n81" in text and "your output:\n11\n11" in text


def test_budget_share_rate_and_limits(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(sy, "clock", lambda: now[0])
    b = sy.Budget(deadline=1000 + 10_000, n_tasks=40, n_parallel=4, reserve_sec=400, min_call_tokens=2048)
    assert b.share() == (10_000 - 400) / 10  # 10 waves of 4 tasks
    assert b.rate() == 20
    assert b.call_limits(32768, 3600) == (960 * 20, 960 * 1.2 + 30)
    b.observe(3000, 100)
    b.observe(1000, 100)
    assert b.rate() == 20  # ratio of sums: 4000 tokens / 200 s
    b.observe(6000, 100)
    assert b.rate() == 10_000 / 300
    b.done = 38
    assert b.share() == 9600  # last wave: one task gets everything
    assert b.call_limits(4096, 3600)[0] == 4096  # flag values are upper bounds
    now[0] += 9_700  # 300 s left, 400 s reserve
    assert b.share() < 0 and b.call_limits(32768, 3600) is None
    now[0] = 1000 + 9_600 - 50  # 50 s of share: 50 * 33 tokens < min_call_tokens
    assert b.call_limits(32768, 3600) is None


def test_tight_budget_skips_model_and_keeps_fallback(tmp_path):
    srv, seen = _server([(_py(GOOD), "", "stop")])
    _, out = _run(tmp_path, srv, "--deadline-min", "3", "--reserve-sec", "120")  # 60 s share * 20 tok/s < 2048
    srv.shutdown()
    assert seen == [] and set(out["t1"][0]) == {"attempt_1", "attempt_2"}


def test_calls_are_capped_by_share_times_rate(tmp_path):
    srv, _ = _server([(_py(GOOD), "", "stop")])
    caps = []
    orig = sy.ls.chat
    sy.ls.chat = lambda *a, **k: (caps.append((a[3], a[4])), orig(*a, **k))[1]
    try:
        _run(tmp_path, srv, "--deadline-min", "10", "--reserve-sec", "60", "--max-tokens", "32768")
    finally:
        sy.ls.chat = orig
        srv.shutdown()
    assert len(caps) == 1 and caps[0][0] <= 540 * 20 and caps[0][1] <= 540 * 1.2 + 30


def test_easy_tasks_first(tmp_path):
    big = {"train": [{"input": [[1] * 9] * 9, "output": [[1] * 9] * 9}], "test": [{"input": [[1] * 9] * 9}]}
    assert sy.task_size(TASK) < sy.task_size(big)
    ch = tmp_path / "c.json"
    ch.write_text(json.dumps({"a_big": big, "z_small": TASK}))
    srv, seen = _server([(_py(GOOD), "", "stop")])
    sy.main(["--challenges", str(ch), "--out", str(tmp_path / "o.json"), "--log-dir", str(tmp_path / "l"),
             "--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--n-parallel", "1", "--chains", "1",
             "--rounds", "1"])
    srv.shutdown()
    assert "Input:\n12\n34" in seen[0][0]["content"]  # the small task is asked first despite sorting last by id


def test_progress_lines_fire_without_completions(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sy, "PROGRESS_SEC", 0.05)
    srv, _ = _server([(_py(GOOD), "", "stop")], delay=0.4)
    _run(tmp_path, srv)
    srv.shutdown()
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("PROGRESS")]
    assert len(lines) >= 4 and "done=0/1" in lines[0] and "done=1/1" in lines[-1]
    assert any("inflight=1" in x for x in lines)
    for key in ("verified=", "tokens=", "tok_s=", "seq_rate=", "left_s="):
        assert key in lines[0]


def test_sigterm_flushes_complete_submission(tmp_path):
    import signal
    import subprocess
    ch = tmp_path / "c.json"
    ch.write_text(json.dumps({"t1": TASK, "t2": TASK}))
    srv, _ = _server([(_py(GOOD), "", "stop")], delay=30)
    out = tmp_path / "o.json"
    p = subprocess.Popen([sys.executable, str(_dir / "synth.py"), "--challenges", str(ch), "--out", str(out),
                          "--log-dir", str(tmp_path / "l"), "--base-url", f"http://127.0.0.1:{srv.server_port}/v1",
                          "--n-parallel", "1"], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().startswith("PROGRESS")  # startup line: handler is installed by now
    time.sleep(0.3)
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=10)
    srv.shutdown()
    sub = json.loads(out.read_text())
    assert rc == 128 + signal.SIGTERM
    assert set(sub) == {"t1", "t2"} and all(set(r) == {"attempt_1", "attempt_2"} for v in sub.values() for r in v)


def test_tight_budget_attempts_the_head_not_skips_it(monkeypatch):
    """Regression: with 20 tasks, 6 workers and a 6-minute deadline the share used to fall below one useful call,
    so the easiest tasks were skipped first. Every task now gets at least min_call_tokens while time lasts."""
    now = [0.0]
    monkeypatch.setattr(sy, "clock", lambda: now[0])
    b = sy.Budget(deadline=360.0, n_tasks=20, n_parallel=6, reserve_sec=20, min_call_tokens=2048)
    lim = b.call_limits(3000, 600)
    assert lim is not None and lim[0] >= 2048
    now[0] = 330.0  # only 10 s of working time left: now the tail is skipped
    assert b.call_limits(3000, 600) is None


def test_plausibility_leaked_colour():
    t = _task([([[8, 1, 2]], [[1, 2, 2]]), ([[8, 3, 4]], [[4, 3, 3]])], [[8, 5, 6]])
    assert _flags(t, [[5, 6, 6]])[0] == 0
    n, why = _flags(t, [[8, 6, 6]])
    assert n == 1 and "leaked colour 8 (in every train input, in no train output)" in why[0]
    one = _task([([[8, 1, 2]], [[1, 2, 2]])], [[8, 5, 6]])  # needs two train pairs
    assert _flags(one, [[8, 6, 6]])[0] == 0
    kept = _task([([[8, 1]], [[8, 1]]), ([[8, 3]], [[3, 8]])], [[8, 5]])  # 8 appears in an output
    assert _flags(kept, [[5, 8]])[0] == 0


def test_plausibility_change_envelope():
    base = [[1] * 10 for _ in range(10)]
    one = [r[:] for r in base]
    one[0][0] = 2
    t = _task([(base, one), (base, one)], base)  # train changes 1% of cells
    ok = [r[:] for r in base]
    ok[0][0] = 2
    ok[1][1] = 2  # 2% < 3 * 1%
    assert _flags(t, ok)[0] == 0
    big = [r[:] for r in base]
    for c in range(6):
        big[0][c] = 2  # 6%
    n, why = _flags(t, big)
    assert n == 1 and "changes 6% of cells (train changes at most 1%)" in why[0]
    resized = _task([([[1, 2]], [[1, 2, 1, 2]]), ([[3]], [[3, 3]])], [[1]])  # not same-shape: rule off
    assert _flags(resized, [[1, 1]])[0] == 0
    assert _flags(t, [[2]])[0] >= 1  # shape change is caught by the shape rule, not the envelope
    assert all("changes" not in w for w in _flags(t, [[2]])[1])
