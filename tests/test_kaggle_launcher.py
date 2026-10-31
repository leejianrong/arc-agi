"""Dry-run the generated Kaggle notebook against a fake OpenAI server (no GPU, no install)."""
import base64
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLIP = "```python\ndef transform(grid):\n    return [r[::-1] for r in grid]\n```"


def _tasks():
    train = [{"input": [[1, 2, 3], [4, 5, 6]], "output": [[3, 2, 1], [6, 5, 4]]}]
    return {f"t{i}": {"train": train, "test": [{"input": [[7, 8, 9], [1, 2, 3]]}]} for i in range(3)}


def _server(die_after=None, smoke_fails=False, tools_fail=False, delay=0.0):
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
            if delay and not smoke_req and not tool_smoke:
                time.sleep(delay)
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


def _strip_from_bundle(main_py, name):
    src = main_py.read_text()
    m = re.search(r'^BUNDLE_B64 = "([^"]*)"', src, re.MULTILINE)
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(m.group(1))), mode="r:gz") as old, \
            tarfile.open(fileobj=out, mode="w:gz") as new:
        for ti in old.getmembers():
            if ti.name != name:
                new.addfile(ti, old.extractfile(ti))
    main_py.write_text(src.replace(m.group(1), base64.b64encode(out.getvalue()).decode()))


def _run(tmp_path, srv, mode="synth", extra=(), eval_files=None, env_extra=None, work_setup=None, strip_trace=False):
    inp = tmp_path / "input"
    inp.mkdir()
    (inp / "arc-agi_test_challenges.json").write_text(json.dumps(_tasks()))
    for name, data in (eval_files or {}).items():
        (inp / name).write_text(json.dumps(data))
    kern = tmp_path / "kern"
    subprocess.run([sys.executable, str(ROOT / "kaggle/arc2_synth/build_kernel.py"), str(kern), "--mode", mode,
                    "--synth-args", "--rounds 2 --chains 1", "--budget-min", "17", *extra], check=True, capture_output=True)
    if strip_trace:  # a kernel whose bundle predates trace.py
        _strip_from_bundle(kern / "main.py", "trace.py")
    work = tmp_path / "work"
    work.mkdir()
    if work_setup:
        work_setup(work)
    env = dict(os.environ, ARC_EXTERNAL_SERVER=f"http://127.0.0.1:{srv.server_port}/v1", ARC_INPUT_ROOT=str(inp),
               ARC_WORK=str(work), ARC_EFFORT="", ARC_MODEL="qwen", **(env_extra or {}))
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


def _bench(tmp_path, srv, configs, solutions=True):
    cfg = tmp_path / "bench.json"
    cfg.write_text(json.dumps(configs))
    files = {"arc-agi_evaluation_challenges.json": _tasks()}
    if solutions:
        files["arc-agi_evaluation_solutions.json"] = {t: [[[9, 8, 7], [3, 2, 1]]] for t in _tasks()}
    r, sub = _run(tmp_path, srv, "bench", ["--npar", "2", "--bench-config", str(cfg), "--bench-tasks", "3"], files)
    lines = [ln for ln in r.stdout.splitlines() if " BENCH name=" in ln]
    return r, sub, lines, tmp_path / "work" / "bench_results.json"


def test_bench_mode_scores_each_config_on_one_server(tmp_path):
    srv = _server()
    r, sub, lines, res = _bench(tmp_path, srv, [{"name": "ag", "mode": "agent", "effort": "xhigh", "persist": 0},
                                                {"name": "dir", "mode": "direct", "effort": "xhigh"}])
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    assert len(lines) == 2 and "name=ag tasks=3 solved=3/3 " in lines[0] and "name=dir tasks=3 solved=3/3 " in lines[1]
    assert "tok_s=" in lines[0] and "wall_s=" in lines[0]
    rows = json.loads(res.read_text())
    assert [x["name"] for x in rows] == ["ag", "dir"] and rows[0]["score"] == [3, 3] and rows[0]["tokens"] > 0
    assert _valid(sub) and "submission validated: 3 tasks, 0 repaired" in r.stdout


def test_bench_mode_without_solutions_is_unscored(tmp_path):
    srv = _server()
    r, sub, lines, res = _bench(tmp_path, srv, [{"name": "ag", "mode": "agent"}], solutions=False)
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    assert len(lines) == 1 and "solved=UNSCORED" in lines[0]
    assert json.loads(res.read_text())[0]["score"] is None
    assert _valid(sub)


def test_bench_mode_a_crashing_config_does_not_stop_the_rest(tmp_path):
    srv = _server()
    r, sub, lines, res = _bench(tmp_path, srv, [{"name": "bad", "mode": "direct", "no_such_flag": 1},
                                                {"name": "ag", "mode": "agent"}])
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    assert "bench bad crashed" in r.stdout and len(lines) == 2 and "name=ag tasks=3 solved=3/3 " in lines[1]
    assert "error" in json.loads(res.read_text())[0]
    assert _valid(sub)


def test_builder_bench_defaults(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "kaggle/arc2_synth/build_kernel.py"), str(tmp_path), "--mode", "bench"],
                   check=True, capture_output=True)
    src = (tmp_path / "main.py").read_text()
    for n in ("agent_xhigh_persist", "agent_xhigh_fresh", "direct_xhigh"):
        assert n in src
    assert "MAX_LEN = 98304" in src and '"12"' in src and '"150"' in src
    subprocess.run([sys.executable, "-m", "py_compile", str(tmp_path / "main.py")], check=True)


def _trace_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("arc2_trace_t", ROOT / "kaggle/arc2_synth/trace.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _events(path):
    return [json.loads(ln) for ln in path.read_text().splitlines()]


def test_bench_agent_configs_write_one_valid_trace_each(tmp_path):
    srv = _server()
    r, sub, *_ = _bench(tmp_path, srv, [{"name": "a1", "mode": "agent", "persist": 0},
                                                {"name": "a2", "mode": "agent"},
                                                {"name": "dir", "mode": "direct"}])
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    work = tmp_path / "work"
    assert not (work / "trace_dir.jsonl").exists() and not (work / "trace.jsonl").exists()
    assert r.stdout.count("trace ok ") == 2
    for name in ("a1", "a2"):
        path = work / f"trace_{name}.jsonl"
        n, errs = _trace_mod().validate_file(str(path))
        assert errs == [] and n > 0
        evs = _events(path)
        assert evs[0]["ev"] == "run_start" and evs[-1]["ev"] == "run_end"
        starts = [e for e in evs if e["ev"] == "task_start"]
        ends = [e for e in evs if e["ev"] == "task_end"]
        assert len(starts) == len(ends) == 3
        assert all(e["answers"] == [[[9, 8, 7], [3, 2, 1]]] for e in starts)
        assert all(e["solved"] == [True] for e in ends)
    assert _valid(sub)


def test_bench_trace_without_solutions_omits_answers(tmp_path):
    srv = _server()
    r, *_ = _bench(tmp_path, srv, [{"name": "ag", "mode": "agent"}], solutions=False)
    srv.shutdown()
    evs = _events(tmp_path / "work" / "trace_ag.jsonl")
    assert r.returncode == 0 and "answers" not in next(e for e in evs if e["ev"] == "task_start")
    assert next(e for e in evs if e["ev"] == "task_end")["solved"] is None


def test_agent_mode_writes_trace_jsonl_with_heartbeats(tmp_path):
    stats = ("INFO Avg prompt throughput: 1.0 tokens/s, Avg generation throughput: 42.5 tokens/s, Running: 3 reqs, "
             "Waiting: 1 reqs, GPU KV cache usage: 12.5%, Prefix cache hit rate: 0.0%\n")
    srv = _server(delay=0.4)
    r, sub = _run(tmp_path, srv, "agent", ["--npar", "1", "--budget-min", "40"], env_extra={"ARC_HB_SEC": "0.1"},
                  work_setup=lambda w: (w / "vllm_server.log").write_text("noise\n" + stats))
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    path = tmp_path / "work" / "trace.jsonl"
    n, errs = _trace_mod().validate_file(str(path))
    assert errs == [] and n > 0
    beats = [e for e in _events(path) if e["ev"] == "heartbeat"]
    assert beats and beats[0]["tok_s"] == 42.5 and beats[0]["running"] == 3 and beats[0]["waiting"] == 1
    assert beats[0]["kv_pct"] == 12.5
    assert _valid(sub)


def test_direct_and_synth_modes_write_no_trace(tmp_path):
    for mode in ("direct", "synth"):
        d = tmp_path / mode
        d.mkdir()
        srv = _server()
        r, sub = _run(d, srv, mode)
        srv.shutdown()
        assert r.returncode == 0, r.stderr[-800:]
        assert _valid(sub) and not list((d / "work").glob("trace*"))


def test_bundle_without_trace_py_still_finishes_with_a_valid_submission(tmp_path):
    srv = _server()
    r, sub = _run(tmp_path, srv, "agent", ["--npar", "2"], strip_trace=True)
    srv.shutdown()
    assert r.returncode == 0, r.stderr[-800:]
    assert _valid(sub) and "submission validated: 3 tasks, 0 repaired" in r.stdout


def test_bundle_contains_trace_py(tmp_path):
    for mode in ("direct", "agent"):
        subprocess.run([sys.executable, str(ROOT / "kaggle/arc2_synth/build_kernel.py"), str(tmp_path), "--mode", mode],
                       check=True, capture_output=True)
        m = re.search(r'^BUNDLE_B64 = "([^"]*)"', (tmp_path / "main.py").read_text(), re.MULTILINE)
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(m.group(1))), mode="r:gz") as tf:
            assert "trace.py" in tf.getnames()
