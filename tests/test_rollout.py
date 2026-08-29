"""End-to-end test (SLICES.md V1): running the rollout script against a
fixture task produces an `episodes/*.jsonl` file with the correct starting
grid and the correct grid after each logged action - the same shape of file
the visualizer reads."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_rollout_script_produces_a_valid_replayable_episode(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "rollout_random.py"),
            "--task_id", "67a3c6ac",
            "--run_id", "pytest-run",
            "--seed", "1",
            "--runs_dir", str(tmp_path),
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr

    run_dir = tmp_path / "pytest-run"
    assert (run_dir / "run_meta.json").exists()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["schema_version"] == 1
    assert meta["algo"] == "random"

    episode_path = run_dir / "episodes" / "67a3c6ac-p0.jsonl"
    assert episode_path.exists()
    lines = [json.loads(line) for line in episode_path.read_text().splitlines()]

    assert lines[0]["type"] == "start"
    assert lines[-1]["type"] == "end"

    grid = lines[0]["grid"]
    for record in lines[1:-1]:
        assert record["type"] == "step"
        assert record["grid_before"] == grid
        grid = record["grid_after"]  # replay: each step's grid_after feeds the next step's grid_before
    assert lines[-1]["n_steps"] == len(lines) - 2
