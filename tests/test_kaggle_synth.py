import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
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


def _cand(out, verified=True, frac=1.0):
    return {"verified": verified, "frac": frac, "test_outputs": [out]}


def test_selection_majority_ties_partial_and_fallback():
    fb = [{"attempt_1": [[0]], "attempt_2": [[9]]}]
    a, b, c = [[1]], [[2]], [[3]]
    rows = sy.select_attempts(TASK, [_cand(a), _cand(b), _cand(b), _cand(c)], fb)
    assert rows == [{"attempt_1": b, "attempt_2": a}]  # majority first, then earlier round on ties
    rows = sy.select_attempts(TASK, [_cand(a), _cand(c, False, 0.2), _cand(b, False, 0.8)], fb)
    assert rows == [{"attempt_1": a, "attempt_2": b}]  # best partial fills slot 2
    rows = sy.select_attempts(TASK, [_cand(a)], fb, direct_rows=[{"attempt_1": None, "attempt_2": c}])
    assert rows == [{"attempt_1": a, "attempt_2": c}]  # direct answer before baseline
    assert sy.select_attempts(TASK, [], fb) == fb


def _server(replies):
    """Chat replies are served in order (last repeats); records each request's messages."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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

    srv = HTTPServer(("127.0.0.1", 0), H)
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
