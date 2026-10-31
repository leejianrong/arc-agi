# MIT No Attribution (MIT-0) — see ../arc2_baseline/solve.py for the full notice.
"""Agent trace writer and validator (schema: docs/arc2-trace-schema.md).

Stdlib only. ``TraceWriter`` appends one JSON object per line, flushed after every event, thread-safe, with
text fields capped; it never raises into the caller. ``validate_event`` / ``validate_file`` check a trace.

    w = TraceWriter("trace.jsonl"); w.emit("turn", task="t1", att=0, turn=1, ...)
"""
import contextlib
import json
import sys
import threading
import time
import uuid

CAPS = {"reasoning": 4000, "content": 2000, "text": 4000, "program": 6000, "reason": 1000}
ARG_CAP = 6000  # `code` inside a tool call's args


def cap(s, n):
    """First n characters, then a marker with the number of characters cut."""
    return s if len(s) <= n else f"{s[:n]}…[cut {len(s) - n}]"


def cap_head_tail(s, n):
    """Head and tail (n // 2 each) around the marker; used for reasoning, where the end matters."""
    if len(s) <= n:
        return s
    h = n // 2
    return f"{s[:h]}…[cut {len(s) - 2 * h}]{s[len(s) - h:]}"


def _capped(fields):
    out = dict(fields)
    for k, n in CAPS.items():
        if isinstance(out.get(k), str):
            out[k] = (cap_head_tail if k == "reasoning" else cap)(out[k], n)
    if isinstance(out.get("tool_calls"), list):
        calls = []
        for c in out["tool_calls"]:
            a = c.get("args") if isinstance(c, dict) else None
            if isinstance(a, dict) and isinstance(a.get("code"), str):
                c = dict(c, args=dict(a, code=cap(a["code"], ARG_CAP)))
            calls.append(c)
        out["tool_calls"] = calls
    return out


class TraceWriter:
    def __init__(self, path, cfg="default", run_id=None):
        self.path, self.cfg = path, cfg
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._f = None
        self._t0 = None
        self._warned = False

    def emit(self, ev, **fields):
        """Append one event. Never raises: a write failure warns once on stderr and the event is dropped."""
        try:
            with self._lock:
                now = time.monotonic()
                self._t0 = now if self._t0 is None else self._t0
                rec = {"ev": ev, "t": round(now - self._t0, 3), "cfg": self.cfg}
                rec.update(_capped(fields))
                line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
                if self._f is None:
                    self._f = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 - held open for the run
                self._f.write(line)
                self._f.flush()
        except Exception as e:  # noqa: BLE001 - tracing must never break a run
            if not self._warned:
                self._warned = True
                print(f"trace: write to {self.path} failed ({e!r}); further failures are silent", file=sys.stderr)

    def close(self):
        with self._lock:
            if self._f is not None:
                with contextlib.suppress(Exception):
                    self._f.close()
                self._f = None


# ---- validation ----

def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _str(v):
    return isinstance(v, str)


def _grid(v):
    return (isinstance(v, list) and 0 < len(v) <= 30 and all(isinstance(r, list) and 0 < len(r) <= 30 for r in v)
            and len({len(r) for r in v}) == 1 and all(_int(c) and 0 <= c <= 9 for r in v for c in r))


def _opt(f):
    return lambda v: v is None or f(v)


def _list_of(f):
    return lambda v: isinstance(v, list) and all(f(x) for x in v)


def _pairs(v):
    return isinstance(v, list) and all(isinstance(p, dict) and _grid(p.get("input")) and (
        "output" not in p or _grid(p["output"])) for p in v)


def _tool_calls(v):
    return isinstance(v, list) and all(isinstance(c, dict) and _str(c.get("id")) and _str(c.get("name"))
                                       and isinstance(c.get("args"), dict) for c in v)


def _wrong(v):
    return isinstance(v, list) and all(
        isinstance(w, dict) and _int(w.get("ex")) and _int(w.get("total")) and isinstance(w.get("cells"), list)
        and all(isinstance(c, list) and len(c) == 4 and all(_int(x) for x in c) for c in w["cells"]) for w in v)


def _rows(v):
    return isinstance(v, list) and all(isinstance(r, dict) and _opt(_grid)(r.get("attempt_1"))
                                       and _opt(_grid)(r.get("attempt_2")) for r in v)


_SUMMARY = {"tasks": _int, "verified": _int, "solved_test_inputs": _opt(lambda v: _str(v) or _int(v)),
            "tokens": _int, "tok_s": _num, "wall_s": _num}


def _summary(v):
    return isinstance(v, dict) and all(k in v and f(v[k]) for k, f in _SUMMARY.items())


_TASK, _ATT = {"task": _str}, {"task": _str, "att": _int}
_TURN = {**_ATT, "turn": _int}
STATUSES = ("verified", "deadline", "no_budget", "error", "prose", "max_turns")
# ev -> (required fields, optional fields); each maps a name to a predicate
SCHEMA = {
    "run_start": ({"run": _str, "model": _str, "effort": _str, "mode": lambda v: v in ("agent", "direct", "synth"),
                   "persist": _opt(lambda v: v in (0, 1)), "tasks": _list_of(_str), "started_at": _str}, {}),
    "task_start": ({**_TASK, "train": _pairs, "test": _pairs}, {"answers": _opt(_list_of(_opt(_grid)))}),
    "turn": ({**_TURN, "prompt_tokens": _int, "completion_tokens": _int, "reasoning_tokens": _int,
              "finish": lambda v: v in ("stop", "length", "tool_calls", "error"), "dur_s": _num, "reasoning": _str,
              "content": _str, "tool_calls": _tool_calls}, {}),
    "tool_result": ({**_TURN, "name": lambda v: v in ("python", "submit"), "text": _str, "dur_s": _num}, {}),
    "submit": ({**_TURN, "program": _str, "reproduced": _int, "n_train": _int, "verified": lambda v: isinstance(v, bool),
                "outputs": _list_of(_opt(_grid)), "wrong": _wrong, "flags": _list_of(_str),
                "test_outputs": _list_of(_opt(_grid))}, {}),
    "rescue": ({**_TURN, "reason": _str}, {}),
    "attempt_end": ({**_ATT, "status": lambda v: v in STATUSES, "verified": lambda v: isinstance(v, bool),
                     "tokens": _int, "turns": _int, "python_calls": _int, "submits": _int, "wall_s": _num}, {}),
    "task_end": ({**_TASK, "rows": _rows, "verified": _int, "solved": _opt(_list_of(lambda v: isinstance(v, bool)))}, {}),
    "heartbeat": ({"tok_s": _opt(_num), "running": _opt(_int), "waiting": _opt(_int), "kv_pct": _opt(_num),
                   "gpu_mem_mib": _opt(_num)}, {}),
    "run_end": ({"summary": _summary}, {}),
}


def validate_event(e):
    """Problems with one event dict (empty list = valid). Unknown extra fields are ignored."""
    if not isinstance(e, dict):
        return ["event is not an object"]
    ev = e.get("ev")
    if ev not in SCHEMA:
        return [f"unknown ev {ev!r}"]
    errs = []
    req, opt = SCHEMA[ev]
    for k, f in {"t": _num, "cfg": _str, **req}.items():
        if k not in e:
            errs.append(f"{ev}: missing {k}")
        elif not f(e[k]):
            errs.append(f"{ev}: bad {k}")
    errs += [f"{ev}: bad {k}" for k, f in opt.items() if k in e and not f(e[k])]
    return errs


def validate_file(path):
    """(n_events, errors) for a trace file. A truncated last line (crash mid-write) is ignored."""
    errs, n = [], 0
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    elif lines:  # no trailing newline: the last line may be cut mid-write
        try:
            json.loads(lines[-1])
        except ValueError:
            lines.pop()
    turns, results, ended = {}, set(), set()  # turns: key -> submit calls it requested
    for i, ln in enumerate(lines, 1):
        try:
            e = json.loads(ln)
        except ValueError:
            errs.append(f"line {i}: not valid JSON")
            continue
        n += 1
        errs += [f"line {i}: {m}" for m in validate_event(e)]
        if not isinstance(e, dict):
            continue
        ev, cfg = e.get("ev"), e.get("cfg")
        if n == 1 and ev != "run_start":
            errs.append(f"line {i}: first event is {ev!r}, not run_start")
        if cfg in ended:
            errs.append(f"line {i}: {ev!r} after run_end")
        key = (cfg, e.get("task"), e.get("att"), e.get("turn"))
        if ev == "turn":
            calls = e.get("tool_calls")
            turns[key] = sum(isinstance(c, dict) and c.get("name") == "submit" for c in calls) if isinstance(
                calls, list) else 0
        elif ev == "submit":
            if key in results and turns.get(key, 0) <= 1:  # several submit calls in one turn alternate legitimately
                errs.append(f"line {i}: submit after its tool_result")
        elif ev == "tool_result":
            if key not in turns:
                errs.append(f"line {i}: tool_result without a preceding turn")
            if e.get("name") == "submit":
                results.add(key)
        elif ev == "run_end":
            ended.add(cfg)
    return n, errs
