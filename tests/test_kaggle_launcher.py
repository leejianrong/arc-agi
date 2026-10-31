"""Dry-run the generated Kaggle notebook against a fake OpenAI server (no GPU, no install)."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLIP = "```python\ndef transform(grid):\n    return [r[::-1] for r in grid]\n```"


def _tasks():
    train = [{"input": [[1, 2, 3], [4, 5, 6]], "output": [[3, 2, 1], [6, 5, 4]]}]
    return {f"t{i}": {"train": train, "test": [{"input": [[7, 8, 9], [1, 2, 3]]}]} for i in range(3)}


def _server(die_after=None, smoke_fails=False, tools_fail=False):
    count = {"n": 0}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            count["n"] += 1
            smoke_req = "Count from 1" in json.dumps(req)
            if (die_after is not None and count["n"] > die_after) or (smoke_fails and smoke_req):
                self.send_response(500)
                self.end_headers()
                return
            tool_smoke = "Call the python tool" in json.dumps(req)
            if tools_fail and req.get("tools"):
                self.send_response(400)  # a server started without --enable-auto-tool-choice
                self.end_headers()
                return
            if req.get("tools"):  # agent mode: the tool smoke gets a python call, every task a verified submit
                name, code = ("python", "print(2)") if tool_smoke else ("submit", FLIP.strip("`").removeprefix("python\n"))
                msg = {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": name, "arguments": json.dumps({"code": code})}}]}
            else:
                msg = {"content": "1 2 3" if smoke_req else FLIP}
            body = json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}],
                               "usage": {"completion_tokens": 5}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _run(tmp_path, srv, mode="synth", extra=()):
    inp = tmp_path / "input"
    inp.mkdir()
    (inp / "arc-agi_test_challenges.json").write_text(json.dumps(_tasks()))
    kern = tmp_path / "kern"
    subprocess.run([sys.executable, str(ROOT / "kaggle/arc2_synth/build_kernel.py"), str(kern), "--mode", mode,
                    "--synth-args", "--rounds 2 --chains 1", "--budget-min", "17", *extra], check=True, capture_output=True)
    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ, ARC_EXTERNAL_SERVER=f"http://127.0.0.1:{srv.server_port}/v1", ARC_INPUT_ROOT=str(inp),
               ARC_WORK=str(work), ARC_EFFORT="", ARC_MODEL="qwen")
    r = subprocess.run([sys.executable, str(kern / "main.py")], env=env, capture_output=True, text=True, timeout=120, check=False)
    sub = json.loads((work / "submission.json").read_text())
    return r, sub


def _valid(sub):
    return set(sub) == set(_tasks()) and all(
        len(rows) == 1 and rows[0]["attempt_1"] and rows[0]["attempt_2"] for rows in sub.values()
    )


def test_healthy_server_solves_and_exits_clean(tmp_path):
    srv = _server()
    r, sub = _run(tmp_path, srv)
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    assert _valid(sub)
    assert all(rows[0]["attempt_1"] == [[9, 8, 7], [3, 2, 1]] for rows in sub.values())
    assert "submission validated: 3 tasks, 0 repaired" in r.stdout


def test_server_dying_mid_run_still_leaves_a_valid_file(tmp_path):
    srv = _server(die_after=9)  # 8 smoke calls pass, then everything fails
    r, sub = _run(tmp_path, srv)
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    assert _valid(sub)


def test_failed_smoke_keeps_the_baseline_submission(tmp_path):
    srv = _server(smoke_fails=True)
    r, sub = _run(tmp_path, srv)
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    assert "failed the smoke test" in r.stdout
    assert _valid(sub)


def test_agent_mode_solves_via_tool_calls_and_exits_clean(tmp_path):
    srv = _server()
    r, sub = _run(tmp_path, srv, "agent", ["--npar", "2"])
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    assert "tool smoke tool_call True" in r.stdout
    assert _valid(sub)
    assert all(rows[0]["attempt_1"] == [[9, 8, 7], [3, 2, 1]] for rows in sub.values())
    assert "submission validated: 3 tasks, 0 repaired" in r.stdout
    subprocess.run([sys.executable, "-m", "py_compile", str(tmp_path / "kern" / "main.py")], check=True)


def test_agent_mode_server_dying_mid_run_still_leaves_a_valid_file(tmp_path):
    srv = _server(die_after=9)  # 8 smoke calls and the tool smoke pass, then everything fails
    r, sub = _run(tmp_path, srv, "agent", ["--npar", "2"])
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    assert _valid(sub)


def test_agent_mode_failed_smoke_keeps_the_baseline_submission(tmp_path):
    srv = _server(smoke_fails=True)
    r, sub = _run(tmp_path, srv, "agent")
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    assert "failed the smoke test" in r.stdout
    assert _valid(sub)


def test_agent_mode_rejected_tools_parameter_fails_the_smoke_test(tmp_path):
    srv = _server(tools_fail=True)
    r, sub = _run(tmp_path, srv, "agent")
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    assert "tool smoke failed" in r.stdout and "failed the smoke test" in r.stdout
    assert _valid(sub)


def test_builder_agent_defaults_and_flags(tmp_path):
    def build(*flags):
        subprocess.run([sys.executable, str(ROOT / "kaggle/arc2_synth/build_kernel.py"), str(tmp_path), *flags],
                       check=True, capture_output=True)
        return (tmp_path / "main.py").read_text()

    src = build("--mode", "agent")
    assert 'EFFORT = "xhigh"' in src and "MAX_LEN = 98304" in src
    assert "AGENT_ARGS = ['--thinking-budget', '0', '--attempts', '1', '--max-turns', '40', '--task-budget', '120000']" in src
    src = build("--mode", "synth")
    assert 'EFFORT = "medium"' in src and "MAX_LEN = 65536" in src
