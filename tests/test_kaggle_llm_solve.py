import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "llm_solve", Path(__file__).resolve().parent.parent / "kaggle/arc2_synth/llm_solve.py"
)
ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ls)

TASK = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
    "test": [{"input": [[5, 6], [7, 8]]}],
}


def test_parse_takes_last_json_block_and_drops_invalid_grids():
    text = (
        '<think>draft ```json {"outputs": [{"attempt_1": [[9]]}]} ```</think>\nfinal\n'
        '```json\n{"outputs": [{"attempt_1": [[8, 7], [6, 5]], "attempt_2": [[1, 99]]}]}\n```'
    )
    rows = ls.parse_outputs(text, 1)
    assert rows == [{"attempt_1": [[8, 7], [6, 5]], "attempt_2": None}]


def test_parse_garbage_is_none():
    assert ls.parse_outputs("no json here", 1) is None


def test_merge_never_returns_none_and_prefers_model_answer():
    fb = [{"attempt_1": [[0]], "attempt_2": [[1]]}]
    assert ls.merge(fb, [{"attempt_1": [[7]], "attempt_2": None}]) == [{"attempt_1": [[7]], "attempt_2": [[1]]}]
    assert ls.merge(fb, None) == fb


def _server(reply, think_reply=None):
    """think_reply: chat reply text (default same as reply). /completions always returns reply's tail."""

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if self.path.endswith("/completions") and not self.path.endswith("chat/completions"):
                out = {"choices": [{"text": reply.split('{"outputs": [', 1)[1] if '{"outputs": [' in reply else reply}],
                       "usage": {"completion_tokens": 7}}
            else:
                content = reply if think_reply is None else think_reply
                out = {"choices": [{"message": {"content": content, "reasoning": "step one... step two..."},
                                    "finish_reason": "length" if think_reply == "" else "stop"}],
                       "usage": {"completion_tokens": 5}}
            body = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _args(tmp_path, srv, **kw):
    ch = tmp_path / "c.json"
    ch.write_text(json.dumps({"t1": TASK}))
    sol = tmp_path / "s.json"
    sol.write_text(json.dumps({"t1": [[[8, 7], [6, 5]]]}))
    argv = ["--challenges", str(ch), "--out", str(tmp_path / "o.json"), "--solutions", str(sol),
            "--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--log-dir", str(tmp_path / "logs"),
            "--n-parallel", "2"]
    return ls.argparse.ArgumentParser, argv


def test_end_to_end_solves_with_fake_model(tmp_path):
    srv = _server('```json\n{"outputs": [{"attempt_1": [[8, 7], [6, 5]], "attempt_2": [[0, 0], [0, 0]]}]}\n```')
    _, argv = _args(tmp_path, srv)
    ls.main(argv)
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["t1"][0]["attempt_1"] == [[8, 7], [6, 5]]
    srv.shutdown()


def test_unreachable_model_still_writes_valid_fallback(tmp_path):
    srv = _server("x")
    _, argv = _args(tmp_path, srv)
    srv.shutdown()
    srv.server_close()
    ls.main([*argv, "--retry-pause", "0.01"])
    out = json.loads((tmp_path / "o.json").read_text())
    assert set(out["t1"][0]) == {"attempt_1", "attempt_2"}


def test_forced_answer_rescues_a_truncated_thinking_call(tmp_path):
    good = '```json\n{"outputs": [{"attempt_1": [[8, 7], [6, 5]], "attempt_2": [[0, 0], [0, 0]]}]}\n```'
    srv = _server(good, think_reply="")  # thinking "runs out of tokens": empty content, finish=length
    _, argv = _args(tmp_path, srv)
    ls.main(argv)
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["t1"][0]["attempt_1"] == [[8, 7], [6, 5]]
    srv.shutdown()


def test_call_cap_follows_the_time_left_so_a_slow_stream_still_gets_a_forced_answer(tmp_path):
    """Hybrid dev run: 16 xhigh calls hit the phase deadline mid-generation, returned nothing, and left no time to force an answer."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body.get("max_tokens"))
            out = {"choices": [{"message": {"content": "", "reasoning": "r"}, "finish_reason": "length"}],
                   "usage": {"completion_tokens": 5}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _, argv = _args(tmp_path, srv)
    ls.main([*argv, "--max-tokens", "40000", "--deadline-min", "10", "--reserve-sec", "300", "--default-rate", "10",
             "--n-parallel", "1", "--force-answer", "0"])
    srv.shutdown()
    assert 2900 <= seen[0] <= 3000  # (600 - 300) s share x 10 tok/s, far below the 40000 ceiling


def test_server_restart_midway_loses_no_task(tmp_path):
    """A vLLM crash/restart shows as refused connections for minutes; the task must wait and be answered afterwards."""
    good = '```json\n{"outputs": [{"attempt_1": [[8, 7], [6, 5]], "attempt_2": [[0, 0], [0, 0]]}]}\n```'
    srv = _server(good)
    port = srv.server_port
    srv.shutdown()
    srv.server_close()  # down at first
    started = []

    def revive():
        import time as _t
        _t.sleep(0.5)
        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                data = json.dumps({"choices": [{"message": {"content": good, "reasoning": "r"}, "finish_reason": "stop"}],
                                   "usage": {"completion_tokens": 5}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        s2 = HTTPServer(("127.0.0.1", port), H)
        started.append(s2)
        s2.serve_forever()

    threading.Thread(target=revive, daemon=True).start()
    class Fake:
        server_port = port
    _, argv = _args(tmp_path, Fake)
    ls.main([*argv, "--retry-pause", "0.2"])
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["t1"][0]["attempt_1"] == [[8, 7], [6, 5]]
    started[0].shutdown()
