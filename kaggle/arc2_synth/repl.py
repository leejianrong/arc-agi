# MIT No Attribution (MIT-0) — see ../arc2_baseline/solve.py for the full notice.
"""Persistent Python REPL sandbox for the ARC-AGI-2 tool-using agent.

Stdlib only (numpy is used inside the worker when importable). ``Repl(task)`` owns one long-lived
``python -I`` worker whose namespace survives between ``run`` calls and is preloaded with ``train``,
``test_inputs`` and a few grid helpers (``show``, ``diff``, ``objects``, ``colors``, ``crop``, ``rot90``,
``flip``, ``transpose``; ``np`` if numpy imports). Verification of candidate programs is NOT done here
(see ``synth.evaluate``).

This is best-effort isolation, not a security boundary against a determined attacker: restricted
builtins, an AST filter (no dunder or frame/code attributes, no ``getattr``/``open``/``exec``), an import
allow-list served as module proxies, rlimits set by the child itself, and a wall-clock timeout enforced by
the parent, which kills the whole process group. It stops buggy and lazy-hostile code (infinite loops,
memory bombs, file writes, ``import os``). It does not stop a skilled escape through the object graph
and does not hide files the user can read. Acceptable because the agent runs offline with no secrets;
the worker gets a minimal environment and an empty temporary working directory anyway.

Protocol: JSON lines on the worker's stdin/stdout. The worker moves the real stdout to a private fd and
points fd 1/2 at /dev/null, and the user's ``print`` goes to a captured buffer, so model output can never
corrupt it.
"""
import atexit
import importlib.util
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import weakref

MAX_OUT = 4000       # characters returned by run()
MAX_REPLY = 1 << 20  # bytes accepted from the worker per reply
ALLOWED_MODULES = ("math", "itertools", "collections", "functools", "copy", "json", "re", "numpy")



def numpy_available():
    """Whether the worker interpreter (this one) can import numpy."""
    return importlib.util.find_spec("numpy") is not None


def namespace_doc(has_numpy):
    """Exact description of what the REPL namespace holds and forbids, for the model's tool text."""
    mods = ", ".join(m for m in ALLOWED_MODULES if has_numpy or m != "numpy")
    return (
        "Preloaded names (grids are plain lists of lists of ints"
        + (", not numpy arrays" if has_numpy else "") + "): "
        "train = list of {\"input\": grid, \"output\": grid} dicts, one per example; test_inputs = list of grids; "
        "show(g) -> str, one digit row per line; diff(a, b) -> str listing differing cells, 'identical' or a shape "
        "mismatch; objects(g, diag=False, bg=None) -> list of single-colour components "
        "{\"color\", \"cells\" [(r, c)], \"bbox\" (r0, c0, r1, c1 inclusive), \"size\"}, bg defaults to the most "
        "common colour (bg=-1 keeps every colour); colors(g) -> {colour: count}, most common first; "
        "crop(g, r0, c0, r1, c1) with inclusive corners; rot90(g, k=1) clockwise; flip(g, axis=0) (0 = upside down, "
        "1 = left-right); transpose(g)." + (" np = numpy (no file I/O)." if has_numpy else "")
        + f" Allowed imports (plain `import x` / `from x import y`, no dotted imports): {mods}. Forbidden here: "
        "type, exec, eval, compile, open, getattr, setattr, vars, globals, dir, id, input, dunder names and "
        "attributes starting with an underscore.")


_WORKER = r"""
import ast, json, os, resource, sys, traceback, types

MAX_BUF = 200000
FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "input", "breakpoint", "exit", "quit", "help", "memoryview", "type", "dir", "id",
}
FORBIDDEN_ATTR_PREFIXES = ("_", "gi_", "f_", "co_", "cr_", "ag_", "tb_")
FORBIDDEN_ATTRS = {"mro", "format", "format_map", "ctypes", "tofile", "dump", "dumps", "tobytes_file"}
ALLOWED_DUNDER_ATTRS = {"__init__"}
# numpy names that touch files, other processes or introspection hooks
NP_DENY = {
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "fromregex",
    "memmap", "DataSource", "lookfor", "info", "show_config", "show_runtime", "test", "get_include", "who",
    "source", "recfromcsv", "recfromtxt", "ctypeslib", "testing", "f2py", "distutils", "fromstring",
}
EXPORTS = {
    "math": None, "itertools": None, "json": {"loads", "dumps"},
    "collections": {"deque", "Counter", "defaultdict", "OrderedDict", "namedtuple"},
    "functools": {"reduce", "lru_cache", "cache", "partial", "cmp_to_key"},
    "copy": {"copy", "deepcopy"},
    "re": {"compile", "match", "fullmatch", "search", "sub", "subn", "findall", "finditer", "split",
           "escape", "I", "IGNORECASE", "M", "MULTILINE", "S", "DOTALL", "X", "VERBOSE", "A", "ASCII"},
}


class Forbidden(BaseException):
    pass


class SafeModule:
    pass


def make_proxy(real, names, deny=()):
    # Real modules leak others (re.enum.sys.modules reaches os), so no module-valued attribute is exposed.
    proxy = SafeModule()
    for attr in dir(real):
        if attr.startswith("_") or attr in deny:
            continue
        value = getattr(real, attr)
        if isinstance(value, types.ModuleType):
            continue
        if names is None or attr in names:
            setattr(proxy, attr, value)
    return proxy


def check_ast(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED:
                    raise Forbidden("import of " + a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED:
                raise Forbidden("import from " + str(node.module))
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise Forbidden("use of " + node.id)
        elif isinstance(node, ast.Attribute):
            if node.attr in ALLOWED_DUNDER_ATTRS:
                continue
            if node.attr.startswith(FORBIDDEN_ATTR_PREFIXES) or node.attr in FORBIDDEN_ATTRS:
                raise Forbidden("attribute ." + node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise Forbidden("dunder string constant")


# helpers (trusted code, preloaded into the user's namespace) -------------------------------------------

def _g(grid):
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    return [list(r) for r in grid]


def show(grid):
    return "\n".join("".join(str(c) for c in r) for r in _g(grid))


def diff(a, b):
    a, b = _g(a), _g(b)
    sa = (len(a), len(a[0]) if a else 0)
    sb = (len(b), len(b[0]) if b else 0)
    if sa != sb:
        return "shape mismatch: %dx%d vs %dx%d" % (sa + sb)
    cells = [(r, c, a[r][c], b[r][c]) for r in range(sa[0]) for c in range(sa[1]) if a[r][c] != b[r][c]]
    if not cells:
        return "identical"
    shown = " ".join("(%d,%d):%d->%d" % x for x in cells[:60])
    return "%d of %d cells differ: %s%s" % (len(cells), sa[0] * sa[1], shown,
                                            " ..." if len(cells) > 60 else "")


def colors(grid):
    counts = {}
    for r in _g(grid):
        for c in r:
            counts[c] = counts.get(c, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def objects(grid, diag=False, bg=None):
    # Single-colour connected components; bg defaults to the most common colour, pass bg=-1 to keep all.
    g = _g(grid)
    if not g:
        return []
    if bg is None:
        bg = next(iter(colors(g)))
    h, w = len(g), len(g[0])
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    seen, out = set(), []
    for r0 in range(h):
        for c0 in range(w):
            if (r0, c0) in seen or g[r0][c0] == bg:
                continue
            col, stack, cells = g[r0][c0], [(r0, c0)], []
            seen.add((r0, c0))
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in steps:
                    n = (r + dr, c + dc)
                    if 0 <= n[0] < h and 0 <= n[1] < w and n not in seen and g[n[0]][n[1]] == col:
                        seen.add(n)
                        stack.append(n)
            cells.sort()
            rs, cs = [x[0] for x in cells], [x[1] for x in cells]
            out.append({"color": col, "cells": cells, "bbox": (min(rs), min(cs), max(rs), max(cs)),
                        "size": len(cells)})
    return out


def crop(grid, r0, c0, r1, c1):
    # Inclusive corners, so crop(g, *obj["bbox"]) cuts out an object.
    return [r[c0:c1 + 1] for r in _g(grid)[r0:r1 + 1]]


def rot90(grid, k=1):
    g = _g(grid)
    for _ in range(k % 4):
        g = [list(r) for r in zip(*g[::-1])]
    return g


def flip(grid, axis=0):
    g = _g(grid)
    return g[::-1] if axis == 0 else [r[::-1] for r in g]


def transpose(grid):
    return [list(r) for r in zip(*_g(grid))]


def main():
    import builtins

    rin = sys.stdin.buffer
    rout = os.fdopen(os.dup(1), "wb", buffering=0)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    json_dumps, json_loads = json.dumps, json.loads

    def send(obj):
        rout.write(json_dumps(obj).encode() + b"\n")

    init = json_loads(rin.readline())
    np_proxy = None
    if "numpy" in init["allowed"]:
        try:
            import numpy
            np_proxy = make_proxy(numpy, None, NP_DENY)
            for sub in ("linalg", "random"):
                setattr(np_proxy, sub, make_proxy(getattr(numpy, sub), None, NP_DENY))
        except Exception:
            np_proxy = None
    allowed = frozenset(a for a in init["allowed"] if a != "numpy" or np_proxy is not None)
    globals()["ALLOWED"] = allowed
    mem = init["mem"]
    for res, value in ((resource.RLIMIT_CORE, 0), (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_AS, mem)):
        resource.setrlimit(res, (value, value))
    real_import = builtins.__import__
    proxies = {}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if np_proxy is not None and name.startswith("numpy."):
            # numpy's C code imports its own submodules lazily through the caller's __import__; user code
            # cannot reach this branch because the AST filter rejects dotted imports and __import__.
            return real_import(name, globals, locals, fromlist, level)
        if level or name not in allowed:
            raise Forbidden("import of " + name)
        if name not in proxies:
            proxies[name] = np_proxy if name == "numpy" else make_proxy(real_import(name), EXPORTS[name])
        return proxies[name]

    buf, size = [], [0]

    def safe_print(*args, sep=" ", end="\n", file=None, flush=False):
        text = (sep if isinstance(sep, str) else " ").join(str(a) for a in args) + (end if isinstance(end, str) else "\n")
        if size[0] < MAX_BUF:
            buf.append(text)
        size[0] += len(text)

    names = (
        "abs all any bool bytes bytearray chr dict divmod enumerate filter float frozenset hash hex bin oct int "
        "isinstance issubclass iter len list map max min next object ord pow property range repr reversed "
        "round set slice sorted staticmethod classmethod str sum super tuple zip callable "
        "Exception ValueError TypeError IndexError KeyError ZeroDivisionError StopIteration AssertionError "
        "ArithmeticError RuntimeError NotImplementedError OverflowError LookupError AttributeError "
        "MemoryError RecursionError"
    ).split()
    safe = {n: getattr(builtins, n) for n in names}
    safe.update(print=safe_print, __import__=safe_import, __build_class__=builtins.__build_class__)
    glb = {"__builtins__": safe, "__name__": "__repl__"}
    task = init["task"]
    glb["train"] = [{"input": _g(p["input"]), "output": _g(p["output"])} for p in task.get("train", [])]
    glb["test_inputs"] = [_g(p["input"] if isinstance(p, dict) else p) for p in task.get("test", [])]
    for fn in (show, diff, objects, colors, crop, rot90, flip, transpose):
        glb[fn.__name__] = fn
    if np_proxy is not None:
        glb["np"] = np_proxy
    send({"ready": True, "numpy": np_proxy is not None})

    while True:
        line = rin.readline()
        if not line:
            return
        req = json_loads(line)
        del buf[:]
        size[0] = 0
        used = resource.getrusage(resource.RUSAGE_SELF)
        soft = int(used.ru_utime + used.ru_stime) + int(req["cpu"]) + 2
        resource.setrlimit(resource.RLIMIT_CPU, (soft, resource.RLIM_INFINITY))
        value, err = None, None
        try:
            tree = ast.parse(req["code"], "<cell>")
            check_ast(tree)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<cell>", "exec"), glb)
            if last is not None:
                v = eval(compile(ast.fix_missing_locations(last), "<cell>", "eval"), glb)
                if v is not None:
                    value = repr(v)[:MAX_BUF]
        except Forbidden as e:
            err = ("Forbidden: " + str(e) + " (not allowed in this sandbox; allowed imports: "
                   + ", ".join(sorted(ALLOWED)) + "; no type/exec/eval/open/getattr/vars, dunder names or dotted imports)")
        except SyntaxError as e:
            err = "SyntaxError: %s (line %s)" % (e.msg, e.lineno)
        except Exception as e:
            # only the user's own frames: the harness's (main, exec) would show as File "<string>"
            frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<cell>"][-4:]
            err = ("Traceback (most recent call last):\n" if frames else "") + "".join(
                traceback.format_list(frames) + traceback.format_exception_only(type(e), e))
        send({"out": "".join(buf), "dropped": max(0, size[0] - MAX_BUF), "value": value, "err": err})


try:
    main()
except BaseException:
    os._exit(1)
"""


def _truncate(text, limit=MAX_OUT):
    if len(text) <= limit:
        return text
    head, tail = limit // 4, limit - limit // 4 - 60
    return f"{text[:head]}\n[... {len(text) - head - tail} chars cut ...]\n{text[-tail:]}"


_LIVE = weakref.WeakSet()


@atexit.register
def _cleanup():
    for r in list(_LIVE):
        r.close()


class Repl:
    """One sandboxed worker with a persistent namespace; ``run(code) -> str`` never raises."""

    def __init__(self, task, persist=True, timeout=30, mem_mb=3072):
        self.task, self.persist, self.timeout, self.mem_mb = task, persist, timeout, mem_mb
        self.restarts = 0
        self.has_numpy = None  # set from the worker's ready message on first start
        self._proc = None
        self._buf = b""
        self._workdir = tempfile.mkdtemp(prefix="repl-")
        self._closed = False
        _LIVE.add(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, code, timeout=None):
        """Execute ``code``; return stdout plus the repr of a final expression, or a traceback tail."""
        if self._closed:
            return "[REPL is closed]"
        timeout = self.timeout if timeout is None else timeout
        note = ""
        try:
            if self._proc is None or self._proc.poll() is not None:
                self._start(timeout)
            reply = self._exchange({"code": code, "cpu": timeout}, timeout)
        except TimeoutError:
            self._kill()
            note = f"[timed out after {timeout:g}s; the worker was killed and all variables and definitions were lost]"
            reply = None
        except (EOFError, OSError, ValueError) as e:
            code_ = self._proc.poll() if self._proc else None
            self._kill()
            how = type(e).__name__ + ("" if code_ is None else f", exit code {code_}")
            note = f"[worker crashed ({how}; out of memory/CPU limit or a fatal error); all variables and definitions were lost]"
            reply = None
        if reply is None:
            return note
        if not self.persist:
            self._kill()
        parts = [reply["out"].rstrip("\n")] if reply["out"] else []
        if reply["dropped"]:
            parts.append(f"[{reply['dropped']} printed chars dropped]")
        if reply["value"] is not None:
            parts.append(reply["value"])
        if reply["err"]:
            parts.append(reply["err"].rstrip("\n"))
        return _truncate("\n".join(parts))

    def close(self):
        self._closed = True
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=1)
            except Exception:  # noqa: BLE001, S110
                pass
            self._reap(proc)
        shutil.rmtree(self._workdir, ignore_errors=True)

    # internals -------------------------------------------------------------------------------------

    def _start(self, timeout):
        self._buf = b""
        env = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "PATH": os.environ.get("PATH", "")}
        os.makedirs(self._workdir, exist_ok=True)
        self._proc = subprocess.Popen([sys.executable, "-I", "-c", _WORKER], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
                                      cwd=self._workdir, start_new_session=True)
        self.restarts += 1
        init = {"task": self.task, "mem": self.mem_mb << 20, "allowed": list(ALLOWED_MODULES)}
        ready = self._exchange(init, max(timeout, 20))
        if not ready.get("ready"):
            raise ValueError("worker did not start")
        self.has_numpy = bool(ready.get("numpy"))

    def _exchange(self, obj, timeout):
        self._proc.stdin.write(json.dumps(obj).encode() + b"\n")
        self._proc.stdin.flush()
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError
            if not select.select([fd], [], [], left)[0]:
                raise TimeoutError
            chunk = os.read(fd, 65536)
            if not chunk:
                raise EOFError("worker exited")
            self._buf += chunk
            if len(self._buf) > MAX_REPLY:
                raise ValueError("oversized reply")
        line, _, self._buf = self._buf.partition(b"\n")
        return json.loads(line)

    def _kill(self):
        proc, self._proc = self._proc, None
        if proc is not None:
            self._reap(proc)

    @staticmethod
    def _reap(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # also covers any stray descendants
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001, S110
            pass
        for f in (proc.stdin, proc.stdout):
            try:
                f.close()
            except Exception:  # noqa: BLE001, S110
                pass
