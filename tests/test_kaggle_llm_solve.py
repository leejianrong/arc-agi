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
    """think_reply: what a thinking-mode request gets (default: same as reply)."""

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            nothink = req.get("chat_template_kwargs", {}).get("enable_thinking") is False
            content = reply if (nothink or think_reply is None) else think_reply
            body = json.dumps(
                {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                 "usage": {"completion_tokens": 5}}
            ).encode()
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
    ls.main(argv)
    out = json.loads((tmp_path / "o.json").read_text())
    assert set(out["t1"][0]) == {"attempt_1", "attempt_2"}


def test_nothink_retry_rescues_a_truncated_thinking_call(tmp_path):
    good = '```json\n{"outputs": [{"attempt_1": [[8, 7], [6, 5]], "attempt_2": [[0, 0], [0, 0]]}]}\n```'
    srv = _server(good, think_reply="")  # thinking mode "runs out of tokens": empty content
    _, argv = _args(tmp_path, srv)
    ls.main(argv)
    out = json.loads((tmp_path / "o.json").read_text())
    assert out["t1"][0]["attempt_1"] == [[8, 7], [6, 5]]
    srv.shutdown()
