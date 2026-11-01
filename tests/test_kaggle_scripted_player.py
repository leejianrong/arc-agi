"""The scripted player drives the real agent loop and REPL through each mechanism; every scenario must hold."""
import importlib.util
import sys
from pathlib import Path

import pytest

_dir = Path(__file__).resolve().parent.parent / "kaggle/arc2_synth"
sys.path.insert(0, str(_dir))
_spec = importlib.util.spec_from_file_location("scripted_player_under_test", _dir / "scripted_player.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


@pytest.mark.parametrize("name", list(sp.SCENARIOS))
def test_scenario(name):
    r = sp.run_one(name)
    assert r["ok"], r["failures"]


def test_the_checks_can_fail():
    """A checker must not be vacuous: a policy that never repeats a submit fails the 'repeat' expectations."""
    _, opts, check = sp.SCENARIOS["repeat"]()
    pol = sp._seq(("submit", sp.program(1)), ("submit", sp.program(2)), ("submit", sp.GOOD))
    att, seen = sp.run_scenario(pol, **opts)
    assert check(att, seen)
