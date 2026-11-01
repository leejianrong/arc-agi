"""Runs the agent-watch viewer's node unit tests when node is available."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_agent_watch_node_tests():
    proc = subprocess.run(
        ["node", "--test", "tools/agent_watch/test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
