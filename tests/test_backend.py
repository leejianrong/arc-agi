"""Tests for `viz/backend/server.py` - the read-only JSON API over `runs/`."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from viz.backend import server as backend


@pytest.fixture
def sample_runs_dir(tmp_path) -> Path:
    run_dir = tmp_path / "run-a"
    (run_dir / "episodes").mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "run-a", "algo": "random",
        "created_at": "2026-08-27T00:00:00Z", "task_ids": ["67a3c6ac"], "config": {},
    }))
    episode_lines = [
        {"type": "start", "episode_id": "67a3c6ac-p0", "task_id": "67a3c6ac", "pair_index": 0,
         "grid": [[1, 2], [3, 4]], "target_grid": [[2, 1], [4, 3]], "max_steps": 25},
        {"type": "step", "step": 0, "grid_before": [[1, 2], [3, 4]],
         "action": {"name": "vmirror", "args": {}}, "grid_after": [[2, 1], [4, 3]],
         "reward": 1.0, "terminated": True, "truncated": False, "done": True, "valid_action": True},
        {"type": "end", "n_steps": 1, "success": True, "total_reward": 1.0},
    ]
    (run_dir / "episodes" / "67a3c6ac-p0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in episode_lines) + "\n"
    )
    return tmp_path


def test_list_runs(sample_runs_dir):
    runs = backend.list_runs(sample_runs_dir)
    assert runs == [{"run_id": "run-a", "algo": "random", "created_at": "2026-08-27T00:00:00Z", "task_ids": ["67a3c6ac"]}]


def test_list_runs_on_missing_dir_is_empty(tmp_path):
    assert backend.list_runs(tmp_path / "does-not-exist") == []


def test_read_metrics_on_run_without_metrics_file_is_empty(sample_runs_dir):
    # V1's random-rollout runs never write metrics.jsonl - only trainers do.
    assert backend.read_metrics(sample_runs_dir, "run-a") == []


def test_read_metrics_parses_jsonl_rows(sample_runs_dir):
    rows = [
        {"update": 0, "timestamp": "2026-08-27T00:00:00Z", "mean_reward": 0.1, "success_rate": 0.0},
        {"update": 1, "timestamp": "2026-08-27T00:00:05Z", "mean_reward": 0.5, "success_rate": 0.2},
    ]
    (sample_runs_dir / "run-a" / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert backend.read_metrics(sample_runs_dir, "run-a") == rows


def test_read_run_meta(sample_runs_dir):
    meta = backend.read_run_meta(sample_runs_dir, "run-a")
    assert meta["algo"] == "random"


def test_list_episode_ids(sample_runs_dir):
    assert backend.list_episode_ids(sample_runs_dir, "run-a") == ["67a3c6ac-p0"]


def test_read_run_thumbnail_loads_first_train_pair(sample_runs_dir):
    thumbnail = backend.read_run_thumbnail(sample_runs_dir, "run-a")
    assert thumbnail["task_id"] == "67a3c6ac"
    assert thumbnail["input"] and thumbnail["output"]
    assert isinstance(thumbnail["input"][0], list)


def test_read_run_thumbnail_no_task_ids_is_none(tmp_path):
    run_dir = tmp_path / "no-tasks"
    run_dir.mkdir()
    (run_dir / "run_meta.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "no-tasks", "algo": "random",
        "created_at": "2026-08-27T00:00:00Z", "task_ids": [], "config": {},
    }))
    assert backend.read_run_thumbnail(tmp_path, "no-tasks") is None


def test_read_episode_splits_start_steps_end(sample_runs_dir):
    episode = backend.read_episode(sample_runs_dir, "run-a", "67a3c6ac-p0")
    assert episode["start"]["task_id"] == "67a3c6ac"
    assert len(episode["steps"]) == 1
    assert episode["steps"][0]["action"]["name"] == "vmirror"
    assert episode["end"]["success"] is True


@pytest.fixture
def running_server(sample_runs_dir):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), backend.make_handler(sample_runs_dir))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    thread.join()


def _get_json(url):
    with urllib.request.urlopen(url) as resp:
        return resp.status, json.loads(resp.read())


def test_http_get_runs(running_server):
    status, body = _get_json(f"{running_server}/api/runs")
    assert status == 200
    assert body[0]["run_id"] == "run-a"


def test_http_get_episode(running_server):
    status, body = _get_json(f"{running_server}/api/runs/run-a/episodes/67a3c6ac-p0")
    assert status == 200
    assert body["start"]["grid"] == [[1, 2], [3, 4]]
    assert body["steps"][0]["grid_after"] == [[2, 1], [4, 3]]


def test_http_get_metrics_on_run_without_metrics_is_empty_list(running_server):
    status, body = _get_json(f"{running_server}/api/runs/run-a/metrics")
    assert status == 200
    assert body == []


def test_http_get_thumbnail(running_server):
    status, body = _get_json(f"{running_server}/api/runs/run-a/thumbnail")
    assert status == 200
    assert body["task_id"] == "67a3c6ac"
    assert body["input"] and body["output"]


def test_http_rejects_path_traversal_ids(running_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{running_server}/api/runs/../meta")
    assert exc_info.value.code == 400


def test_http_unknown_run_is_404(running_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{running_server}/api/runs/does-not-exist/meta")
    assert exc_info.value.code == 404
