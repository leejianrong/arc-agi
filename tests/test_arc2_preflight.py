"""The Kaggle pre-flight gate: checkers on broken inputs, plus the fast path end to end."""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("arc2_preflight", ROOT / "scripts/arc2_preflight.py")
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def _good_meta():
    with tempfile.TemporaryDirectory() as d:
        _, metaf = pf.build(d, "direct")
        return json.loads(metaf.read_text())


def test_good_metadata_passes_and_wrong_machine_shape_fails():
    meta = _good_meta()
    assert pf.check_metadata(meta) == []
    assert any("machine_shape" in p for p in pf.check_metadata({**meta, "machine_shape": "NvidiaL4X4"}))


def test_metadata_flags_internet_sources_and_privacy():
    meta = _good_meta()
    bad = pf.check_metadata({**meta, "enable_internet": "true", "is_private": "false", "competition_sources": [],
                             "dataset_sources": [], "model_sources": []})
    assert len(bad) == 5


def test_placeholder_detection():
    assert pf.find_placeholders("LIMIT = @@LIMIT@@\nx = 1") == ["@@LIMIT@@"]
    assert pf.find_placeholders("LIMIT = 0") == []


def test_bundle_modules_of_agent_build():
    with tempfile.TemporaryDirectory() as d:
        main, _ = pf.build(d, "agent")
        assert {"repl.py", "synth_agent.py"} <= pf.bundle_modules(main.read_text())


def test_fast_path_end_to_end():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/arc2_preflight.py"), "--fast", "--modes", "direct"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().endswith("PREFLIGHT OK")
