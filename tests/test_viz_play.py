"""Tests for `viz/backend/play.py` and the `/api/play/*`, `/api/actions`
routes it wires into `viz/backend/server.py` (F13 Stage 0, ADR-0017)."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from viz.backend import play
from viz.backend import server as backend


@pytest.fixture(autouse=True)
def _clear_sessions():
    # `play._sessions` is module-level, in-memory state shared across tests
    # in this process - reset it around every test so tests don't leak
    # sessions into each other.
    play._sessions.clear()
    yield
    play._sessions.clear()


@pytest.fixture
def running_server(tmp_path):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), backend.make_handler(tmp_path))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", tmp_path
    httpd.shutdown()
    thread.join()


def _post_json(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def _get_json(url):
    with urllib.request.urlopen(url) as resp:
        return resp.status, json.loads(resp.read())


def _http_error_body(exc: urllib.error.HTTPError) -> dict:
    return json.loads(exc.read())


# ---------------------------------------------------------------------------
# GET /api/actions
# ---------------------------------------------------------------------------


def test_list_actions_matches_arc_env_actions():
    from arc_env import actions as arc_actions

    listed = play.list_actions()
    assert len(listed) == len(arc_actions.ACTIONS)
    assert {a["name"] for a in listed} == set(arc_actions.ACTION_BY_NAME)
    vmirror = next(a for a in listed if a["name"] == "vmirror")
    assert vmirror["kind"] == "transform"
    assert vmirror["args"] == []
    fill_cell = next(a for a in listed if a["name"] == "fill_cell")
    assert [arg["name"] for arg in fill_cell["args"]] == ["color", "row", "col"]
    assert all("kind" in arg for arg in fill_cell["args"])


def test_http_get_actions(running_server):
    base, _ = running_server
    status, body = _get_json(f"{base}/api/actions")
    assert status == 200
    assert any(a["name"] == "vmirror" for a in body)


# ---------------------------------------------------------------------------
# Full start -> step -> save flow, direct function calls
# ---------------------------------------------------------------------------


def test_start_step_save_solves_67a3c6ac(tmp_path):
    # 67a3c6ac's curated solver is a single vmirror (arc_env/task_loader.py).
    start = play.start_session("67a3c6ac", 0)
    assert start["task_id"] == "67a3c6ac"
    assert start["terminated"] is False
    assert start["exact_match"] is False
    session_id = start["session_id"]

    step = play.step_session(session_id, "vmirror", [])
    assert step["valid_action"] is True
    assert step["exact_match"] is True
    assert step["terminated"] is True
    assert step["grid"] == step["target_grid"]

    saved = play.save_session(session_id, tmp_path, run_id="human-test-run")
    assert saved == {"run_id": "human-test-run", "episode_id": "67a3c6ac-p0"}

    # Read the saved run back through the *existing* read-only endpoints.
    runs = backend.list_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["algo"] == "human"
    assert runs[0]["task_ids"] == ["67a3c6ac"]

    episode_ids = backend.list_episode_ids(tmp_path, "human-test-run")
    assert episode_ids == ["67a3c6ac-p0"]

    episode = backend.read_episode(tmp_path, "human-test-run", "67a3c6ac-p0")
    assert episode["start"]["task_id"] == "67a3c6ac"
    assert len(episode["steps"]) == 1
    assert episode["steps"][0]["action"]["name"] == "vmirror"
    assert episode["steps"][0]["exact_match"] is True
    assert episode["end"]["success"] is True


def test_start_step_save_over_http(running_server):
    base, _runs_dir = running_server
    status, start = _post_json(f"{base}/api/play/start", {"task_id": "67a3c6ac", "pair_index": 0})
    assert status == 200
    session_id = start["session_id"]

    status, step = _post_json(f"{base}/api/play/{session_id}/step", {"primitive": "vmirror", "args": []})
    assert status == 200
    assert step["exact_match"] is True

    status, saved = _post_json(f"{base}/api/play/{session_id}/save", {})
    assert status == 200
    assert saved["run_id"].startswith("human-67a3c6ac-")

    # Confirm it's readable back through the existing read-only routes.
    status, runs = _get_json(f"{base}/api/runs")
    assert status == 200
    assert any(r["run_id"] == saved["run_id"] and r["algo"] == "human" for r in runs)


def test_save_defaults_run_id_when_not_given(tmp_path):
    start = play.start_session("67a3c6ac", 0)
    saved = play.save_session(start["session_id"], tmp_path)
    assert saved["run_id"].startswith("human-67a3c6ac-")


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_step_unknown_session_is_404():
    with pytest.raises(play.PlayError) as exc_info:
        play.step_session("does-not-exist", "vmirror", [])
    assert exc_info.value.status == 404


def test_step_unknown_primitive_is_400():
    start = play.start_session("67a3c6ac", 0)
    with pytest.raises(play.PlayError) as exc_info:
        play.step_session(start["session_id"], "not_a_real_action", [])
    assert exc_info.value.status == 400


def test_save_unknown_session_is_404(tmp_path):
    with pytest.raises(play.PlayError) as exc_info:
        play.save_session("does-not-exist", tmp_path)
    assert exc_info.value.status == 404


def test_start_rejects_path_traversal_task_id():
    with pytest.raises(play.PlayError) as exc_info:
        play.start_session("../../../etc/passwd", 0)
    assert exc_info.value.status == 400


def test_start_rejects_unknown_task_id():
    with pytest.raises(play.PlayError) as exc_info:
        play.start_session("this-task-does-not-exist", 0)
    assert exc_info.value.status == 400


def test_save_rejects_path_traversal_run_id(tmp_path):
    start = play.start_session("67a3c6ac", 0)
    with pytest.raises(play.PlayError) as exc_info:
        play.save_session(start["session_id"], tmp_path, run_id="../../etc/passwd")
    assert exc_info.value.status == 400


def test_http_start_rejects_path_traversal_task_id(running_server):
    base, _ = running_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{base}/api/play/start", {"task_id": "../../../etc/passwd"})
    assert exc_info.value.code == 400


def test_http_step_unknown_session_is_404(running_server):
    base, _ = running_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{base}/api/play/does-not-exist/step", {"primitive": "vmirror", "args": []})
    assert exc_info.value.code == 404


def test_http_step_unknown_primitive_is_400(running_server):
    base, _ = running_server
    _, start = _post_json(f"{base}/api/play/start", {"task_id": "67a3c6ac", "pair_index": 0})
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{base}/api/play/{start['session_id']}/step", {"primitive": "nope", "args": []})
    assert exc_info.value.code == 400


def test_read_only_routes_unaffected_by_play_routes(running_server):
    # Sanity check the new dispatch didn't regress the existing read-only
    # endpoints (additive-only change).
    base, _ = running_server
    status, body = _get_json(f"{base}/api/runs")
    assert status == 200
    assert body == []
