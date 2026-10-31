# MIT No Attribution (MIT-0) — see ../arc2_baseline/solve.py for the full notice.
"""Explore-then-submit agent for ARC: the model measures the examples with a `python` tool and
proposes programs with a `submit` tool; a submit that reproduces every demo ends the attempt.

Stdlib only; reuses ``synth`` (sandboxed evaluation, selection, budget) and ``llm_solve`` (chat client).
Control rules (``Control``: submit nudge, prefix-stable context trimming, thinking-cap hint, no-think rescue
turn, overflow retry, repetition guard) are individually switchable and counted in the attempt log.

    python synth_agent.py --challenges C.json --out sub.json [--solutions S.json]
"""
import argparse
import contextlib
import importlib.util
import json
import os
import signal
import sys
import threading
import time
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_solve as ls
import repl as rp
import synth as sy

# loaded by path (the stdlib has a module named `trace`); optional so a kernel bundled without trace.py still runs
_TRACE_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace.py")
tr = None
if os.path.exists(_TRACE_PY):
    _spec = importlib.util.spec_from_file_location("arc2_trace", _TRACE_PY)
    tr = importlib.util.module_from_spec(_spec)
    sys.modules["arc2_trace"] = tr
    _spec.loader.exec_module(tr)

SYSTEM_PROMPT = (
    "You are an expert at ARC puzzles. Every puzzle's examples follow one hidden rule that maps an input grid "
    "to an output grid. Grids are lists of integers 0-9 (colors), usually with 0 as background. You work by "
    "writing Python programs."
)

NUMPY = rp.numpy_available()  # the REPL worker and the submit runner both use this interpreter
SUBMIT_ENV = ("submit runs your program in a separate plain Python process, not the REPL sandbox: the standard "
              "library" + (" and numpy are" if NUMPY else " is") + " importable there without the REPL's restrictions.")

TOOLS = [
    {"type": "function", "function": {
        "name": "python",
        "description": ("Run Python to inspect the puzzle in a restricted sandbox. State persists between calls. "
                        + rp.namespace_doc(NUMPY) + " Print compact summaries, never whole large grids."),
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code; "
                       "print() what you need."}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": ("Submit the full source of a program defining transform(grid). It is run on every example "
                        "input; the reply says which examples it reproduces and, for failures, which cells are "
                        "wrong. The attempt ends when every example is reproduced. " + SUBMIT_ENV),
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Full Python "
                       "source defining transform(grid)."}}, "required": ["code"]}}},
]

INSTRUCTIONS = (
    "Work like a scientist. Use the python tool to measure the examples: objects, colors, sizes, positions, and "
    "how outputs differ from inputs. Then call submit with a complete `def transform(grid):` (standard library"
    + (" and numpy" if NUMPY else "") + " only; it takes and returns a list of lists of ints). Submitting is free "
    "and a failed submit shows exactly which cells are wrong, so submit early and often instead of waiting for certainty. Fix the rule "
    "rather than special-casing the examples."
)

PROSE_HINT = "Use the python tool to explore, or call submit with your transform program."
CUT_HINT = "Your reply was cut off. Be more concise: test your best hypothesis with python or call submit."
CAP_HINT = ("Your per-turn thinking limit was hit. Stop analysing: call python to test your best hypothesis, "
            "or call submit.")
NUDGE = ("\n\n[{n} python calls without a submit. Submitting is free and returns a cell diff: submit your best "
         "transform now, then refine it.]")
REPEAT_HINT = "You have made the same call three times in a row with the same result. Change approach."
RESCUE_PROMPT = ("Out of thinking budget. Reply now, without further analysis, with your best program as a "
                 "submit(code) call.")


def dims(g):
    return f"{len(g)}x{len(g[0])}"


def task_message(task):
    parts = [INSTRUCTIONS, ""]
    for i, p in enumerate(task["train"], 1):
        parts.append(f"## Example {i}\nInput ({dims(p['input'])}):\n{ls.grid_text(p['input'])}\n"
                     f"Output ({dims(p['output'])}):\n{ls.grid_text(p['output'])}\n")
    for j, t in enumerate(task["test"], 1):
        parts.append(f"## Test input {j}\nInput ({dims(t['input'])}):\n{ls.grid_text(t['input'])}\n")
    return "\n".join(parts)


def diff_map(got, want):
    """Wrong-cell map: '.' for correct cells, the produced digit for wrong ones."""
    return "\n".join("".join("." if g == w else str(g) for g, w in zip(gr, wr)) for gr, wr in zip(got, want))


def diff_detail(got, want, cells=True):
    """Up to 10 wrong cells ('(r,c) got A expected B') and the 5 commonest got->expected colour pairs."""
    wrong = [(r, c, g, w) for r, (gr, wr) in enumerate(zip(got, want)) for c, (g, w) in enumerate(zip(gr, wr)) if g != w]
    pairs = ", ".join(f"{g}->{w} x{n}" for (g, w), n in Counter((g, w) for _, _, g, w in wrong).most_common(5))
    lines = [f"Wrong cells: {'; '.join(f'({r},{c}) got {g} expected {w}' for r, c, g, w in wrong[:10])}"] if cells else []
    return "\n".join(lines + [f"Colour pairs (got->expected): {pairs}"])


REPLY_CAP = 6000  # characters in one submit reply; the first failing example's grids (<= 30x30) always fit


def submit_reply(cand, task, max_examples=2):
    """Reply text for a submit; may re-run the failing examples to show their wrong-cell maps."""
    train = task["train"]
    n = len(train)
    if cand["verified"]:
        return f"Accepted: reproduces all {n} examples."
    if not cand["fails"]:  # demos pass but a test input produced no valid grid
        return ("Reproduces all examples but a test input gave no valid grid (error, timeout or bad output). "
                      "Make transform robust on other inputs and submit again.")
    idx = [int(f.split(":")[0].split()[1]) for f in cand["fails"]][:max_examples]
    wrong_cells = [i for i, f in zip(idx, cand["fails"]) if "cells wrong" in f]
    got = {}
    if wrong_cells:
        res = sy.run_program(cand["code"], [train[i - 1]["input"] for i in wrong_cells])
        got = {i: r["grid"] for i, r in zip(wrong_cells, res) if r["status"] == "ok"}
    out = [f"Reproduced {n - len(cand['fails'])} of {n} examples."]
    detailed = False  # cell/pair diagnostics for the first right-shaped failure only
    for i, f in zip(idx, cand["fails"]):
        want = train[i - 1]["output"]
        out.append(f"## Example {i} failed")
        if i in got:
            g = got[i]
            bad = sum(a != b for gr, wr in zip(g, want) for a, b in zip(gr, wr))
            total = len(want) * len(want[0])
            if total > 900:
                out.append(f"{bad} of {total} cells are wrong")
                if not detailed:
                    out.append(diff_detail(g, want, cells=False))
            else:
                out.append(f"{bad} of {total} cells are wrong. Expected output:\n{ls.grid_text(want)}\n"
                           f"Your output ('.' = correct cell, digit = wrong cell):\n{diff_map(g, want)}")
                if not detailed:
                    out.append(diff_detail(g, want))
            detailed = True
        else:
            out.append(f.split(": ", 1)[1])
            if "wrong shape" in f and len(want) * len(want[0]) <= 900:
                out.append(f"Expected output:\n{ls.grid_text(want)}")
    out.append("Fix the rule and submit again.")
    text = "\n".join(out)
    if len(text) > REPLY_CAP:
        keep = REPLY_CAP - 80
        text = f"{text[:keep]}\n[... reply truncated: {len(text) - keep} chars cut ...]"
    return text


class Hooks:
    """Extension points for control rules. Defaults change nothing."""

    def prepare(self, messages, state):
        """Return the message list to send this turn (context trimming goes here)."""
        return messages

    def tool_result(self, name, text, state):
        """Return the text of a tool reply (nudges go here)."""
        return text


def _chars(msgs):
    return sum(len(json.dumps(m)) for m in msgs)


def _fail_headline(text):
    # a "[task <id>] " tag may precede the reply's first line
    for ln in text.splitlines():
        ln = ln.split("] ", 1)[1] if ln.startswith("[task ") and "] " in ln else ln
        if ln.startswith("Reproduce"):
            return ln
    return ""


@dataclass
class Control(Hooks):
    """Control rules with the spec's defaults; a zero or False field switches its rule off."""
    nudge_every: int = 6        # python calls without a submit before the nudge suffix
    trim_budget: int = 24000    # estimated prompt tokens that trigger eviction
    trim_drain: int = 4000      # evict down to budget - drain, so the prefix stays fixed between evictions
    keep_recent: int = 6        # messages after the head that eviction never touches
    rescue: bool = True         # one no-think turn asking for submit(code) before an attempt ends
    rescue_tokens: int = 8192
    overflow_retry: bool = True  # HTTP 400 over context: shrink budget by a quarter, trim, retry once
    repeat_guard: bool = True   # three identical tool-call turns in a row: ask for a new approach once

    def tool_result(self, name, text, state):
        if name != "python" or not self.nudge_every:
            return text
        state["py_run"] = n = state.get("py_run", 0) + 1
        if n % self.nudge_every:
            return text
        state["attempt"].events["nudges"] = state["attempt"].events.get("nudges", 0) + 1
        return text + NUDGE.format(n=n)

    def prepare(self, messages, state):
        """Evict the oldest whole assistant+tool exchanges once over budget. Messages [0:2] (system, task) and
        the marker at [2] are the stable head; eviction rewrites the list in place, so between evictions every
        request extends the previous one byte for byte."""
        if not self.trim_budget:
            return messages
        budget = state.setdefault("trim_budget", self.trim_budget)
        pt = state.get("prompt_tokens", 0)
        sent = state.get("sent", 0)
        est = pt + _chars(messages[sent:]) / 3 if pt else _chars(messages) / 3
        if state.pop("force_trim", False):
            est = max(est, budget + 1)
        if est <= budget:
            return messages
        head = 3 if state.get("marker") else 2
        groups = []  # a group starts at an assistant message, so tool replies never lose their call
        for m in messages[head:]:
            if m["role"] == "assistant" or not groups:
                groups.append([m])
            else:
                groups[-1].append(m)
        remaining, drop = len(messages) - head, []
        while groups and est > budget - self.trim_drain and remaining - len(groups[0]) >= self.keep_recent:
            g = groups.pop(0)
            est -= _chars(g) / 3
            remaining -= len(g)
            drop += g
        if not drop:
            return messages
        d = state.setdefault("dropped", {"exchanges": 0, "python": 0, "failed": 0, "last": ""})
        names = {}
        for m in drop:
            for c in m.get("tool_calls") or []:
                names[c["id"]] = c["function"]["name"]
                d["python"] += c["function"]["name"] == "python"
            text = m["content"] if isinstance(m.get("content"), str) else ""
            if m["role"] == "assistant":
                d["exchanges"] += 1
            elif (m["role"] == "user" or names.get(m.get("tool_call_id")) == "submit") and _fail_headline(text):
                d["failed"] += 1
                d["last"] = _fail_headline(text)
        note = (f"[Older turns were removed to save context: {d['exchanges']} exchanges, {d['python']} python calls, "
                f"{d['failed']} failed submits." + (f" Last failed submit: {d['last']}" if d["last"] else "")
                + " Your python variables persist.]")
        messages[2:head + len(drop)] = [{"role": "user", "content": note}]
        state["marker"], state["prompt_tokens"] = True, 0
        ev = state["attempt"].events
        ev["trims"] = ev.get("trims", 0) + 1
        return messages


class Attempt:
    def __init__(self):
        self.messages, self.cands, self.turns = [], [], []
        self.events, self.control = {}, {}
        self.tokens = self.python_calls = self.submits = 0
        self.status = "running"
        self.verified = False
        self.persist = True

    def log(self):
        return {"status": self.status, "verified": self.verified, "tokens": self.tokens, "turns": len(self.turns),
                "python_calls": self.python_calls, "submits": self.submits, "events": self.events,
                "control": self.control, "persist": self.persist}


def _default_repl(task, persist=True):
    from repl import Repl
    return Repl(task, persist=persist)


def run_attempt(task, *, base_url, model, budget, deadline, tag="", max_turns=40, max_tokens=32768,
                attempt_tokens=120000, request_timeout=3600, temperature=0.6, effort="", thinking_budget=0,
                sandbox_timeout=10, repl_factory=None, hooks=None, on_cand=None, max_hints=1,
                tool_timeout=None, control=None, persist=1, trace=None, task_id="", att_idx=0):
    """One conversation. Returns an Attempt (cands feed ``synth.select_attempts``).
    ``trace``: optional ``trace.TraceWriter``; task_id/att_idx label its events."""
    pre = f"[task {tag}] " if tag else ""
    ctl = control or Control()
    hooks = hooks or ctl
    att = Attempt()
    att.control = asdict(ctl)
    att.persist = bool(persist)
    att.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pre + task_message(task)}]
    kwargs = {"reasoning_effort": effort} if effort else None
    seen = {}  # code -> (cand, reply, submit number), to flag identical resubmissions without re-running them
    repl, hints, think = None, 0, thinking_budget
    state = {"attempt": att}
    sigs, overflowed = [], False
    t_att = sy.clock()

    def emit(ev, **f):
        if trace:
            trace.emit(ev, task=task_id, att=att_idx, **f)

    def trace_submit(cand, code):
        train = task["train"]
        res = sy.run_program(code, [p["input"] for p in train])  # trace-only re-run for the per-example outputs
        outs = [r["grid"] if r["status"] == "ok" else None for r in res]
        wrong = []
        for i, (o, p) in enumerate(zip(outs, train)):  # `ex` is a 0-based index into train, like `outputs`
            want = p["output"]
            if o == want:
                continue
            same = o is not None and len(o) == len(want) and len(o[0]) == len(want[0])
            cells = [[r, c, g, w] for r, (gr, wr) in enumerate(zip(o, want)) for c, (g, w) in enumerate(zip(gr, wr))
                     if g != w] if same else []
            wrong.append({"ex": i, "total": len(cells) if same else len(want) * len(want[0]), "cells": cells[:10]})
        flags = []
        if cand["verified"]:
            flags = list(dict.fromkeys(w for _, ws in sy.plausibility(task, cand) for w in ws))
        emit("submit", turn=len(att.turns), program=code, reproduced=len(train) - len(cand["fails"]),
             n_train=len(train), verified=cand["verified"], outputs=outs, wrong=wrong, flags=flags,
             test_outputs=cand["test_outputs"])

    def submit(code):
        att.submits += 1
        state["py_run"] = 0
        key = code.strip()
        if key in seen:
            k, reply = seen[key]
            return (f"WARNING: identical to your submission #{k}; it was not re-run, the result is the same. "
                    "Change the approach.\n" + reply)
        cand = sy.evaluate(code, task, sandbox_timeout)
        if cand["verified"] and any(o is None for o in cand["test_outputs"]):
            cand["verified"] = False  # a test input must produce a valid grid too
        cand.update(turn=len(att.turns))
        att.cands.append(cand)
        if on_cand:
            on_cand(cand)
        if trace:
            with contextlib.suppress(Exception):  # tracing must not change the attempt
                trace_submit(cand, code)
        if cand["verified"]:
            att.verified = True
        seen[key] = (att.submits, reply := submit_reply(cand, task))
        return reply

    def dispatch(call):
        nonlocal repl
        a = call["arguments"]
        code = a.get("code") if isinstance(a.get("code"), str) else ""
        if call["name"] == "python":
            att.python_calls += 1
            if not code.strip():
                return "Missing 'code' argument. Call python with a JSON object {\"code\": \"...\"}."
            try:
                repl = repl or (repl_factory or _default_repl)(task, persist=bool(persist))
                return repl.run(code, tool_timeout) if tool_timeout else repl.run(code)
            except Exception as e:  # noqa: BLE001 - a broken sandbox must not kill the attempt
                return f"[python tool error: {e!r}]"
        if call["name"] == "submit":
            if not code.strip():
                return "Missing 'code' argument. Call submit with a JSON object {\"code\": \"...\"}."
            return submit(code)
        return f"Unknown tool {call['name']!r}. Tools: python, submit."

    def emit_turn(r, dur, error=None):
        if not trace:
            return
        if r is None:
            emit("turn", turn=len(att.turns), prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
                 finish="error", dur_s=round(dur, 3), reasoning="", content=repr(error), tool_calls=[])
            return
        det = r["usage"].get("completion_tokens_details") or {}
        fin = r["finish_reason"] if r["finish_reason"] in ("stop", "length", "tool_calls") else "stop"
        emit("turn", turn=len(att.turns), prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
             reasoning_tokens=det.get("reasoning_tokens") or 0, finish=fin, dur_s=round(dur, 3),
             reasoning=r["reasoning"], content=r["content"],
             tool_calls=[{"id": c["id"], "name": c["name"], "args": c["arguments"]} for c in r["tool_calls"]])

    def rescue(why):
        """One last request with thinking off, asking for the best submit; a call or fenced code counts."""
        lim = budget.call_limits(ctl.rescue_tokens, request_timeout)
        if not ctl.rescue or att.events.get("rescue") or sy.clock() > deadline or lim is None:
            return
        att.events["rescue"] = why
        emit("rescue", turn=len(att.turns) + 1, reason=why)
        att.messages.append({"role": "user", "content": pre + RESCUE_PROMPT})
        info = {"turn": len(att.turns), "rescue": why}
        att.turns.append(info)
        t_r = sy.clock()
        try:
            r = ls.chat_full(base_url, model, hooks.prepare(att.messages, state), lim[0], lim[1], temperature,
                             {"enable_thinking": False}, tools=TOOLS, tool_choice="auto")
        except Exception as e:  # noqa: BLE001
            info["error"] = repr(e)
            emit_turn(None, 0.0, error=e)
            return
        emit_turn(r, sy.clock() - t_r)
        att.tokens += r["completion_tokens"]
        budget.add_tokens(r["completion_tokens"])
        code = next((c["arguments"]["code"] for c in r["tool_calls"] if c["name"] == "submit"
                     and isinstance(c["arguments"].get("code"), str)), None) or sy.extract_code(r["content"])
        info.update(tokens=r["completion_tokens"], submit=bool(code and code.strip()))
        if info["submit"]:
            submit(code)
            if att.verified:
                att.status = "verified"

    try:
        for _ in range(max_turns):
            if sy.clock() > deadline:
                att.status = "deadline"
                break
            left = attempt_tokens - att.tokens
            limits = budget.call_limits(min(max_tokens, left), request_timeout) if left >= 1024 else None
            if limits is None:
                att.status = "no_budget"
                if left < 1024:
                    rescue("tokens")
                break
            info = {"turn": len(att.turns)}
            att.turns.append(info)
            t0 = sy.clock()
            try:
                sent = hooks.prepare(att.messages, state)
                state["sent"] = len(sent)
                r = ls.chat_full(base_url, model, sent, limits[0], limits[1], temperature,
                                 kwargs, tools=TOOLS, tool_choice="auto", thinking_token_budget=think)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace").lower() if e.fp else ""
                if think and e.code == 400 and any(w in body for w in ("thinking", "budget", "reasoning")):
                    think = 0  # the server cannot take the field: drop it and retry this turn
                    att.turns.pop()
                    continue
                if (e.code == 400 and ctl.overflow_retry and not overflowed and ctl.trim_budget
                        and any(w in body for w in ("context", "length", "max_tokens", "max_model_len"))):
                    overflowed = True  # shrink by a quarter (floor 8000), force a trim, retry this turn once
                    cur = state.get("trim_budget", ctl.trim_budget)
                    state["trim_budget"] = att.events["trim_budget"] = max(int(cur * 0.75), min(cur, 8000))
                    state["force_trim"] = True
                    att.events["overflow_retries"] = att.events.get("overflow_retries", 0) + 1
                    att.turns.pop()
                    continue
                info["error"] = repr(e)
                att.status = "error"
                emit_turn(None, sy.clock() - t0, error=e)
                break
            except Exception as e:  # noqa: BLE001
                info["error"] = repr(e)
                att.status = "error"
                emit_turn(None, sy.clock() - t0, error=e)
                break
            overflowed = False
            state["prompt_tokens"] = r["prompt_tokens"]
            n = r["completion_tokens"]
            att.tokens += n
            budget.add_tokens(n)
            budget.observe(n, sy.clock() - t0)
            calls, content = r["tool_calls"], r["content"]
            info.update(tokens=n, finish=r["finish_reason"], calls=[c["name"] for c in calls])
            msg = {"role": "assistant", "content": content or None}
            if calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                    "name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]
            else:
                msg["content"] = content or "(no reply)"
            att.messages.append(msg)
            emit_turn(r, sy.clock() - t0)
            if calls:
                for c in calls:
                    t_c = sy.clock()
                    text = pre + hooks.tool_result(c["name"], dispatch(c), state)
                    if c["name"] in ("python", "submit"):
                        emit("tool_result", turn=len(att.turns), name=c["name"], text=text,
                             dur_s=round(sy.clock() - t_c, 3))
                    att.messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
                if att.verified:
                    att.status = "verified"
                    break
                sigs.append(json.dumps([[c["name"], c["arguments"]] for c in calls], sort_keys=True))
                if ctl.repeat_guard and not att.events.get("repeat_guard") and len(sigs) >= 3 \
                        and len(set(sigs[-3:])) == 1:
                    att.events["repeat_guard"] = 1
                    att.messages.append({"role": "user", "content": pre + REPEAT_HINT})
                continue
            code = sy.extract_code(content) if content else None
            if code:  # a fenced program in prose counts as a submit, answered as a user message
                info["implicit_submit"] = True
                att.messages.append({"role": "user", "content": pre + submit(code)})
                if att.verified:
                    att.status = "verified"
                    break
                continue
            cut = r["finish_reason"] == "length"
            if hints >= max_hints:
                att.status = "prose"
                if cut:
                    rescue("cut")
                break
            hints += 1
            att.messages.append({"role": "user", "content": pre + (
                (CAP_HINT if think else CUT_HINT) if cut else PROSE_HINT)})
        else:
            att.status = "max_turns"
    finally:
        emit("attempt_end", status=att.status if att.status in tr.STATUSES else "error", verified=att.verified,
             tokens=att.tokens, turns=len(att.turns), python_calls=att.python_calls, submits=att.submits,
             wall_s=round(sy.clock() - t_att, 3))
        if repl is not None:
            try:
                repl.close()
            except Exception as e:  # noqa: BLE001
                att.turns.append({"close_error": repr(e)})
    return att


NO_ANSWERS = object()  # solve_task default: no solutions known, task_start carries no `answers`


def trace_task_end(trace, tid, rows, verified, answers):
    solved = [a in (r["attempt_1"], r["attempt_2"]) for r, a in zip(rows, answers)] if answers else None
    trace.emit("task_end", task=tid, rows=rows, verified=verified, solved=solved)


def solve_task(task, args, deadline, on_update=None, fallback_rows=None, tag="", budget=None, repl_factory=None,
               hooks=None, trace=None, task_id="", answers=NO_ANSWERS):
    """Up to args.attempts sequential attempts (stops once args.want_verified programs verified); returns (rows, log).
    ``answers`` (public solutions, list or None) is only for the trace; the model never sees it."""
    budget = budget or sy.Budget(deadline, 1, 1, args.reserve_sec, args.min_call_tokens)
    fallback_rows = fallback_rows or ls.load_fallback()(task)
    cands, logs = [], []
    if trace:
        trace.emit("task_start", task=task_id, train=task["train"], test=[{"input": t["input"]} for t in task["test"]],
                   **({} if answers is NO_ANSWERS else {"answers": answers}))

    def update():
        rows = sy.select_attempts(task, cands, fallback_rows)
        if on_update:
            on_update(rows)
        return rows

    for idx in range(args.attempts):
        if sy.clock() > deadline or sum(c["verified"] for c in cands) >= args.want_verified:
            break
        att = run_attempt(
            task, base_url=args.base_url, model=args.model, budget=budget, deadline=deadline, tag=tag,
            max_turns=args.max_turns, max_tokens=args.max_tokens, attempt_tokens=args.task_budget,
            request_timeout=args.request_timeout, temperature=args.temperature, effort=args.effort,
            thinking_budget=args.thinking_budget, sandbox_timeout=args.sandbox_timeout,
            repl_factory=repl_factory, hooks=hooks, persist=getattr(args, "persist", 1),
            on_cand=lambda c: (cands.append(c), update()), trace=trace, task_id=task_id, att_idx=idx)
        logs.append(dict(att.log(), turns_detail=att.turns))
    rows = update()
    best = max(cands, key=lambda c: (c["verified"], c["frac"]), default=None)
    log = {"rounds": [t for lg in logs for t in lg["turns_detail"]], "attempts": [
               {k: v for k, v in lg.items() if k != "turns_detail"} for lg in logs],
           "tokens": sum(lg["tokens"] for lg in logs), "verified": sum(c["verified"] for c in cands),
           "candidates": len(cands), "final_program": best["code"] if best else None,
           "final_frac": round(best["frac"], 3) if best else None}
    if trace:
        trace_task_end(trace, task_id, rows, log["verified"], answers)
    return rows, log


def run(args):
    with open(args.challenges) as f:
        challenges = json.load(f)
    ids = sorted(challenges)[: args.limit] if args.limit else sorted(challenges)
    if args.order == "easy":
        ids.sort(key=lambda t: (sy.task_size(challenges[t]), t))
    fallback = ls.load_fallback()
    fb = {tid: fallback(challenges[tid]) for tid in challenges}
    sub = {tid: fb[tid] for tid in challenges}
    lock = threading.Lock()
    stats = {"done": 0, "verified": 0, "inflight": 0}
    t_start = sy.clock()
    deadline = t_start + args.deadline_min * 60
    budget = sy.Budget(deadline, len(ids), args.n_parallel, args.reserve_sec, args.min_call_tokens)
    os.makedirs(args.log_dir, exist_ok=True)
    tw = tr.TraceWriter(args.trace, cfg=getattr(args, "cfg", "default")) if getattr(args, "trace", None) else None  # needs trace.py beside this file
    sol = None
    if tw:
        if args.solutions:
            with open(args.solutions) as f:
                sol = json.load(f)
        tw.emit("run_start", run=tw.run_id, model=args.model, effort=args.effort, mode="agent",
                persist=args.persist, tasks=ids, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def flush():
        tmp = f"{args.out}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump(dict(sub), f)
        os.replace(tmp, args.out)

    def set_rows(tid, rows):
        with lock:
            sub[tid] = rows
            flush()

    def progress():
        el = max(sy.clock() - t_start, 1e-9)
        print(f"PROGRESS done={stats['done']}/{len(ids)} inflight={stats['inflight']} verified={stats['verified']} "
              f"tokens={budget.tokens} tok_s={budget.tokens / el:.1f} seq_rate={budget.rate():.1f} "
              f"left_s={max(0, int(deadline - sy.clock()))}", flush=True)

    def work(tid):
        if sy.clock() > deadline:
            return
        t0 = sy.clock()
        with lock:
            stats["inflight"] += 1
        try:
            rows, log = solve_task(challenges[tid], args, deadline, lambda r: set_rows(tid, r), fallback_rows=fb[tid],
                                   tag=tid if args.tag_prompts else "", budget=budget, trace=tw, task_id=tid,
                                   **({"answers": sol.get(tid)} if sol is not None else {}))
        except Exception as e:  # noqa: BLE001 - one task's failure must not stop the run
            rows, log = fb[tid], {"error": repr(e), "tokens": 0, "verified": 0}
            if tw:
                trace_task_end(tw, tid, rows, 0, sol.get(tid) if sol is not None else None)
        dt = sy.clock() - t0
        with lock:
            sub[tid] = rows
            stats["done"] += 1
            stats["inflight"] -= 1
            stats["verified"] += bool(log.get("verified"))
            budget.done += 1
            flush()
            print(f"[{stats['done']}/{len(ids)}] {tid} verified={log.get('verified')} tok={log.get('tokens')} "
                  f"turns={len(log.get('rounds', []))} {dt:.0f}s", flush=True)
        with open(os.path.join(args.log_dir, f"{tid}.json"), "w") as f:
            json.dump(dict(log, elapsed=dt), f)

    def on_term(signum, frame):
        flush()
        progress()
        os._exit(128 + signum)

    flush()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, on_term)
    stop = threading.Event()

    def ticker():
        while not stop.wait(sy.PROGRESS_SEC):
            progress()

    progress()
    threading.Thread(target=ticker, daemon=True).start()
    with ThreadPoolExecutor(args.n_parallel) as ex:
        list(ex.map(work, ids))
    stop.set()
    flush()
    progress()
    wall = sy.clock() - t_start
    summary = {"tasks": len(ids), "done": stats["done"], "verified": stats["verified"], "wall_s": round(wall),
               "completion_tokens": budget.tokens, "tok_per_s": round(budget.tokens / wall, 1) if wall else 0}
    if args.solutions:
        with open(args.solutions) as f:
            sol = json.load(f)
        hits = sum(1 for tid in ids for row, s in zip(sub[tid], sol[tid]) if s in (row["attempt_1"], row["attempt_2"]))
        summary["solved_test_inputs"] = f"{hits}/{sum(len(sol[tid]) for tid in ids)}"
    print("SUMMARY " + json.dumps(summary), flush=True)
    if tw:
        tw.emit("run_end", summary={
            "tasks": len(ids), "verified": stats["verified"], "solved_test_inputs": summary.get("solved_test_inputs"),
            "tokens": budget.tokens, "tok_s": summary["tok_per_s"], "wall_s": summary["wall_s"]})
        tw.close()
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--solutions")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--n-parallel", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32768, help="per model call")
    ap.add_argument("--task-budget", type=int, default=120000, help="completion tokens per attempt, all turns")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--attempts", type=int, default=1, help="sequential attempts per task")
    ap.add_argument("--want-verified", type=int, default=1, help="stop a task after this many verified programs")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--effort", default="medium", help="reasoning_effort template kwarg ('' to omit)")
    ap.add_argument("--thinking-budget", type=int, default=0, help="thinking_token_budget per turn (0 = omit)")
    ap.add_argument("--sandbox-timeout", type=float, default=10)
    ap.add_argument("--request-timeout", type=int, default=3600)
    ap.add_argument("--deadline-min", type=float, default=60)
    ap.add_argument("--min-call-tokens", type=int, default=2048)
    ap.add_argument("--reserve-sec", type=float, default=120)
    ap.add_argument("--order", choices=["easy", "given"], default="easy")
    ap.add_argument("--log-dir", default="agent_logs")
    ap.add_argument("--persist", type=int, choices=[0, 1], default=1, help="python variables survive between calls")
    ap.add_argument("--tag-prompts", type=int, default=0, help="prefix every user message with [task <id>]")
    ap.add_argument("--trace", default=None, help="append a JSONL event trace (docs/arc2-trace-schema.md) here")
    ap.add_argument("--cfg", default="default", help="config name written on every trace event (bench: one per config)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
