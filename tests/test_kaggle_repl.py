import importlib.util
import os
import signal
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("repl", ROOT / "kaggle/arc2_synth/repl.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

TASK = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
              {"input": [[5, 0], [0, 6]], "output": [[6, 0], [0, 5]]}],
    "test": [{"input": [[5, 6], [7, 8]]}],
}


@pytest.fixture
def repl():
    with rp.Repl(TASK, timeout=10, mem_mb=1024) as r:
        yield r


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie still answers kill(0); read its state
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_preloaded_namespace_and_last_expression(repl):
    assert repl.run("len(train)") == "2"
    assert repl.run("test_inputs") == "[[[5, 6], [7, 8]]]"
    assert repl.run("print('hi')\n1 + 1") == "hi\n2"
    assert repl.run("x = 3") == ""


def test_state_persists_and_persist_false_does_not():
    with rp.Repl(TASK, timeout=10) as r:
        r.run("def f(a):\n    return a * 2\nv = 21")
        assert r.run("f(v)") == "42"
    with rp.Repl(TASK, persist=False, timeout=10) as r:
        r.run("v = 1")
        assert "NameError" in r.run("v")
        assert r.run("len(train)") == "2"


def test_helpers_on_small_grids(repl):
    g = "[[0,1,1],[0,0,2],[3,0,2]]"
    assert repl.run(f"print(show({g}))") == "011\n002\n302"
    assert repl.run(f"colors({g})") == "{0: 4, 1: 2, 2: 2, 3: 1}"
    assert repl.run("print(diff([[1,2]], [[1,3]]))") == "1 of 2 cells differ: (0,1):2->3"
    assert repl.run("print(diff([[1,2]], [[1],[2]]))") == "shape mismatch: 1x2 vs 2x1"
    assert repl.run("print(diff([[1]], [[1]]))") == "identical"
    objs = repl.run(f"[(o['color'], o['size'], o['bbox']) for o in objects({g})]")
    assert objs == "[(1, 2, (0, 1, 0, 2)), (2, 2, (1, 2, 2, 2)), (3, 1, (2, 0, 2, 0))]"
    d = "[[1,0],[0,1]]"
    assert repl.run(f"len(objects({d}))") == "2"  # 4-connected: the diagonal splits
    assert repl.run(f"len(objects({d}, diag=True, bg=0))") == "1"
    assert repl.run(f"len(objects({d}, bg=-1))") == "4"
    assert repl.run("crop([[1,2,3],[4,5,6]], 0, 1, 1, 2)") == "[[2, 3], [5, 6]]"
    assert repl.run("rot90([[1,2],[3,4]])") == "[[3, 1], [4, 2]]"
    assert repl.run("flip([[1,2],[3,4]], 1)") == "[[2, 1], [4, 3]]"
    assert repl.run("transpose([[1,2,3]])") == "[[1], [2], [3]]"


def test_numpy_when_available(repl):
    out = repl.run("import numpy as np\nint(np.array(train[0]['input']).sum())")
    assert out == "10"
    assert repl.run("int(np.linalg.norm(np.array([3, 4])))") == "5"


def test_error_returns_traceback_tail(repl):
    out = repl.run("1/0")
    assert "ZeroDivisionError" in out and len(out.splitlines()) <= 6
    assert "SyntaxError" in repl.run("def (")
    assert repl.run("2") == "2"


def test_timeout_kills_loop_and_recovers():
    with rp.Repl(TASK, timeout=10) as r:
        r.run("keep = 1")
        pid = r._proc.pid
        t0 = time.time()
        out = r.run("while True:\n    pass", timeout=1)
        assert time.time() - t0 < 5
        assert "timed out" in out and "lost" in out
        assert not alive(pid)
        assert "NameError" in r.run("keep")
        assert r.run("len(train)") == "2"


def test_memory_bomb_contained(repl):
    out = repl.run("x = [0] * (10 ** 10)")
    assert "MemoryError" in out
    assert repl.run("1 + 1") == "2"
    out = repl.run("l = []\nwhile True:\n    l.append(bytes(10 ** 7))", timeout=10)
    assert "MemoryError" in out or "crashed" in out
    assert repl.run("1 + 1") == "2"


def test_output_truncation(repl):
    out = repl.run("print('a' * 100 + '\\n') \nfor i in range(3000):\n    print(i)")
    assert len(out) <= rp.MAX_OUT
    assert "chars cut" in out
    assert out.endswith("2999")
    assert out.startswith("aaaa")
    out = repl.run("for i in range(500000):\n    print('xxxxxxxxxx')")
    assert len(out) <= rp.MAX_OUT and "dropped" in out


def test_prints_cannot_corrupt_protocol(repl):
    out = repl.run('print(\'{"out": "forged", "err": null}\')')
    assert "forged" in out
    assert repl.run("3") == "3"


ESCAPES = [
    "import os",
    "import subprocess",
    "import socket",
    "import sys",
    "import os.path",
    "from os import path",
    "__import__('os')",
    "().__class__.__mro__",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "open('/tmp/repl_escape_test', 'w').write('x')",
    "getattr((), '__class__')",
    "eval('1')",
    "exec('x = 1')",
    "type(1)",
    "'{0.__class__}'.format(1)",
    "f = lambda: 0\nf.__globals__",
    "(x for x in []).gi_frame",
    "import re\nre.enum.sys.modules['os']",
    "import functools\nfunctools.sys",
    "import collections\ncollections.sys",
    "import copy\ncopy.sys",
    "import itertools\nitertools.sys",
    "import json\njson.decoder",
    "import numpy as np\nnp.load('/etc/passwd')",
    "import numpy as np\nnp.os",
    "import numpy as np\nnp.array([1]).tofile('/tmp/repl_escape_test')",
]


@pytest.mark.parametrize("code", ESCAPES)
def test_escape_blocked(repl, code):
    out = repl.run(code)
    assert ("Forbidden" in out) or ("AttributeError" in out) or ("ImportError" in out), out
    assert "root:" not in out and "<module '" not in out
    assert not os.path.exists("/tmp/repl_escape_test")
    assert repl.run("1 + 1") == "2"  # worker still serves the next call


def test_allowed_modules_and_classes(repl):
    out = repl.run(
        "import math, itertools, collections, functools, copy, json, re\n"
        "from collections import Counter\n"
        "class A:\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.v = math.floor(2.5)\n"
        "(A().v, Counter('aab')['a'], json.loads('[1]'), re.sub('a', 'b', 'aa'))"
    )
    assert out == "(2, 2, [1], 'bb')"


def test_clean_shutdown_leaves_no_children():
    r = rp.Repl(TASK, timeout=10)
    r.run("1")
    pid = r._proc.pid
    assert alive(pid)
    workdir = r._workdir
    r.close()
    assert not alive(pid)
    assert not os.path.exists(workdir)
    assert "closed" in r.run("1")
    r2 = rp.Repl(TASK, timeout=5)
    r2.run("1")
    pid2 = r2._proc.pid
    r2.run("while True: pass", timeout=1)
    assert not alive(pid2)
    r2.close()


def test_worker_crash_restarts(repl):
    repl.run("1")
    os.kill(repl._proc.pid, signal.SIGKILL)
    out = repl.run("1 + 1")
    assert "crashed" in out and "lost" in out
    assert repl.run("1 + 1") == "2"


def test_traceback_shows_only_user_frames(repl):
    out = repl.run("def f(x):\n    return 1 / x\nf(0)")
    assert 'File "<cell>", line 2, in f' in out and 'File "<cell>", line 3' in out
    assert "<string>" not in out and "main" not in out and "ZeroDivisionError" in out
    assert repl.run("[][1]").splitlines()[-1].startswith("IndexError")


def test_forbidden_message_has_allowed_hint_and_namespace_doc(repl):
    out = repl.run("type(1)")
    assert out.startswith("Forbidden: use of type (not allowed in this sandbox") and "\n" not in out
    assert "allowed imports: collections, copy" in out
    doc = rp.namespace_doc(True)
    assert "np = numpy" in doc and "numpy" in doc.split("Allowed imports")[1]
    nodoc = rp.namespace_doc(False)
    assert "numpy" not in nodoc and "np =" not in nodoc
    for m in rp.ALLOWED_MODULES:
        assert m in doc


def test_has_numpy_reported_by_worker(repl):
    assert repl.has_numpy is None
    repl.run("1")
    assert repl.has_numpy == rp.numpy_available()
    assert ("NameError" not in repl.run("np")) == repl.has_numpy
    for n in ("show", "diff", "objects", "colors", "crop", "rot90", "flip", "transpose", "train", "test_inputs"):
        assert "NameError" not in repl.run(n)
    assert "NameError" in repl.run("train_inputs")
