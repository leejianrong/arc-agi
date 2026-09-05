#!/usr/bin/env python3
"""V1 visualizer backend (ADR-0006, ADR-0007, SLICES.md V1 step 4).

Serves a `runs/` directory tree as JSON over local HTTP - read-only, no
database, no write path. Also serves the built frontend (`viz/frontend/dist`)
as static files, so `python -m viz.backend.server` is the one command that
brings the whole visualizer up.

Routes:
    GET /api/runs                              -> [{run_id, algo, created_at, task_ids}, ...]
    GET /api/runs/<run_id>/meta                 -> run_meta.json, verbatim
    GET /api/runs/<run_id>/metrics              -> [metrics.jsonl row, ...] (ADR-0006, V2 training dashboard)
    GET /api/runs/<run_id>/episodes             -> [episode_id, ...]
    GET /api/runs/<run_id>/episodes/<episode_id> -> {start, steps: [...], end}
    GET /api/runs/<run_id>/thumbnail            -> {task_id, input, output} (run picker thumbnails)
    GET /* (anything else)                      -> static files from frontend/dist, index.html for unknown paths
"""

import argparse
import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from arc_env.task_loader import load_task

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
FRONTEND_DIST = REPO_ROOT / "viz" / "frontend" / "dist"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_EPISODE_ID_RE = _RUN_ID_RE


def _safe_id(value: str) -> bool:
    return bool(_RUN_ID_RE.match(value)) and value not in (".", "..")


def list_runs(runs_dir: Path) -> list:
    if not runs_dir.is_dir():
        return []
    out = []
    for run_path in sorted(runs_dir.iterdir()):
        meta_path = run_path / "run_meta.json"
        if not meta_path.is_file():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        out.append({
            "run_id": meta.get("run_id", run_path.name),
            "algo": meta.get("algo"),
            "created_at": meta.get("created_at"),
            "task_ids": meta.get("task_ids", []),
        })
    return out


def read_run_meta(runs_dir: Path, run_id: str) -> dict:
    with open(runs_dir / run_id / "run_meta.json") as f:
        return json.load(f)


def read_metrics(runs_dir: Path, run_id: str) -> list:
    path = runs_dir / run_id / "metrics.jsonl"
    if not path.is_file():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_run_thumbnail(runs_dir: Path, run_id: str) -> dict | None:
    """The run's task's first train pair (input/output grids), for the run
    picker's thumbnail - not episode data, so this works even for a run with
    no episodes logged yet. `None` when the run has no `task_ids` (nothing to
    show), which callers should turn into a 404, not a crash."""

    meta = read_run_meta(runs_dir, run_id)
    task_ids = meta.get("task_ids", [])
    if not task_ids:
        return None
    task = load_task(task_ids[0])
    pair = task.train[0]
    return {
        "task_id": task_ids[0],
        "input": [list(row) for row in pair.input],
        "output": [list(row) for row in pair.output],
    }


def list_episode_ids(runs_dir: Path, run_id: str) -> list:
    episodes_dir = runs_dir / run_id / "episodes"
    if not episodes_dir.is_dir():
        return []
    return sorted(p.stem for p in episodes_dir.glob("*.jsonl"))


def read_episode(runs_dir: Path, run_id: str, episode_id: str) -> dict:
    path = runs_dir / run_id / "episodes" / f"{episode_id}.jsonl"
    start, steps, end = None, [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["type"] == "start":
                start = record
            elif record["type"] == "step":
                steps.append(record)
            elif record["type"] == "end":
                end = record
    return {"start": start, "steps": steps, "end": end}


def make_handler(runs_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # quieter default logging
            pass

        def _json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message):
            self._json({"error": message}, status=status)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            parts = [p for p in path.split("/") if p]

            try:
                if parts[:2] == ["api", "runs"] and len(parts) == 2:
                    return self._json(list_runs(runs_dir))

                if parts[:2] == ["api", "runs"] and len(parts) == 4 and parts[3] == "meta":
                    run_id = parts[2]
                    if not _safe_id(run_id):
                        return self._error(HTTPStatus.BAD_REQUEST, "invalid run_id")
                    return self._json(read_run_meta(runs_dir, run_id))

                if parts[:2] == ["api", "runs"] and len(parts) == 4 and parts[3] == "metrics":
                    run_id = parts[2]
                    if not _safe_id(run_id):
                        return self._error(HTTPStatus.BAD_REQUEST, "invalid run_id")
                    return self._json(read_metrics(runs_dir, run_id))

                if parts[:2] == ["api", "runs"] and len(parts) == 4 and parts[3] == "episodes":
                    run_id = parts[2]
                    if not _safe_id(run_id):
                        return self._error(HTTPStatus.BAD_REQUEST, "invalid run_id")
                    return self._json(list_episode_ids(runs_dir, run_id))

                if parts[:2] == ["api", "runs"] and len(parts) == 5 and parts[3] == "episodes":
                    run_id, episode_id = parts[2], parts[4]
                    if not (_safe_id(run_id) and _safe_id(episode_id)):
                        return self._error(HTTPStatus.BAD_REQUEST, "invalid id")
                    return self._json(read_episode(runs_dir, run_id, episode_id))

                if parts[:2] == ["api", "runs"] and len(parts) == 4 and parts[3] == "thumbnail":
                    run_id = parts[2]
                    if not _safe_id(run_id):
                        return self._error(HTTPStatus.BAD_REQUEST, "invalid run_id")
                    thumbnail = read_run_thumbnail(runs_dir, run_id)
                    if thumbnail is None:
                        return self._error(HTTPStatus.NOT_FOUND, "run has no task_ids")
                    return self._json(thumbnail)

                if parts and parts[0] == "api":
                    return self._error(HTTPStatus.NOT_FOUND, "no such API route")

            except FileNotFoundError:
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            except (json.JSONDecodeError, KeyError, IndexError):
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "malformed run data")

            self._serve_static(path)

        def _serve_static(self, path: str):
            if not FRONTEND_DIST.is_dir():
                return self._error(
                    HTTPStatus.NOT_FOUND,
                    "frontend not built - run `npm run build` in viz/frontend",
                )
            rel = path.lstrip("/") or "index.html"
            candidate = (FRONTEND_DIST / rel).resolve()
            if FRONTEND_DIST.resolve() not in candidate.parents and candidate != FRONTEND_DIST.resolve():
                return self._error(HTTPStatus.BAD_REQUEST, "invalid path")
            if not candidate.is_file():
                candidate = FRONTEND_DIST / "index.html"
            if not candidate.is_file():
                return self._error(HTTPStatus.NOT_FOUND, "not found")

            content_type = {
                ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                ".json": "application/json", ".svg": "image/svg+xml",
            }.get(candidate.suffix, "application/octet-stream")
            body = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(runs_dir: Path = DEFAULT_RUNS_DIR, port: int = 8000) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(runs_dir))
    server.daemon_threads = True  # don't let a stuck in-flight request block Ctrl-C
    print(f"viz backend serving {runs_dir} on http://127.0.0.1:{port} (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nshutting down")
        server.server_close()  # release the port immediately for the next `make viz`


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.runs_dir, args.port)


if __name__ == "__main__":
    main()
