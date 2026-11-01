"""Headless-browser smoke test for tools/agent_watch (skipped unless Playwright and Chromium are installed).

Playwright is not a project dependency. To run: `mkdir /tmp/pw-verify && cd /tmp/pw-verify && npm i -D playwright &&
npx playwright install chromium`, then `ARC_WATCH_PW_DIR=/tmp/pw-verify uv run pytest tests/test_agent_watch_browser.py`.
It exists because a fake-DOM unit test accepted an array child that a real DOM rejects.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PW_DIR = Path(os.environ.get("ARC_WATCH_PW_DIR", "/tmp/pw-verify"))
SMOKE = ROOT / "tools/agent_watch/test/browser_smoke.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (PW_DIR / "node_modules/playwright").exists(),
    reason="needs node and a Playwright install (ARC_WATCH_PW_DIR)",
)


def test_viewer_renders_in_a_real_browser_without_errors():
    env = dict(os.environ, NODE_PATH=str(PW_DIR / "node_modules"))
    r = subprocess.run(["node", str(SMOKE)], capture_output=True, text=True, env=env, timeout=120, check=False)
    assert r.returncode == 0, r.stderr[-800:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["errors"] == [] and out["external"] == []
    assert out["watch"]["left"] > 0 and out["watch"]["center"] > 0 and out["watch"]["right"] > 0
    assert out["wall"]["panels"] >= 3 and out["wall"]["tiles"] == 5 and out["wall"]["rows"] == 2
    assert out["phoneNoHScroll"] is True
