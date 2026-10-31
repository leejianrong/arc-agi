# MIT No Attribution (MIT-0) — see ../arc2_baseline/solve.py for the full notice.
"""Explore-then-submit agent for ARC: the model measures the examples with a `python` tool and
proposes programs with a `submit` tool; a submit that reproduces every demo ends the attempt.

Stdlib only; reuses ``synth`` (sandboxed evaluation, selection, budget) and ``llm_solve`` (chat client).
Control rules beyond the bare loop (nudges, context trimming, rescue turns) belong to a later story;
``Hooks`` marks where they plug in.

    python synth_agent.py --challenges C.json --out sub.json [--solutions S.json]
"""
import argparse
import json
import os
import signal
import sys
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_solve as ls
import synth as sy

SYSTEM_PROMPT = (
    "You are an expert at ARC puzzles. Every puzzle's examples follow one hidden rule that maps an input grid "
    "to an output grid. Grids are lists of integers 0-9 (colors), usually with 0 as background. You work by "
    "writing Python programs."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "python",
        "description": ("Run Python to inspect the puzzle. State persists between calls. Preloaded: np, train_inputs, "
                        "train_outputs, test_inputs (lists of int numpy arrays), train (list of (input, output)), "
                        "show(g), diff(a, b), objects(g, bg=None, diagonal=False, multicolor=False). "
                        "Print compact summaries, never whole large grids."),
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code; "
                       "print() what you need."}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": ("Submit the full source of a program defining transform(grid). It is run on every example "
                        "input; the reply says which examples it reproduces and, for failures, which cells are "
                        "wrong. The attempt ends when every example is reproduced."),
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Full Python "
                       "source defining transform(grid)."}}, "required": ["code"]}}},
]

INSTRUCTIONS = (
    "Work like a scientist. Use the python tool to measure the examples: objects, colors, sizes, positions, and "
    "how outputs differ from inputs. Then call submit with a complete `def transform(grid):` (standard library "
    "and numpy only; it takes and returns a list of lists of ints). Submitting is free and a failed submit shows "
    "exactly which cells are wrong, so submit early and often instead of waiting for certainty. Fix the rule "
    "rather than special-casing the examples."
)

PROSE_HINT = "Use the python tool to explore, or call submit with your transform program."
CUT_HINT = "Your reply was cut off. Be more concise: test your best hypothesis with python or call submit."


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


def submit_reply(cand, task, repeated=False, max_examples=2):
    """Reply text for a submit; may re-run the failing examples to show their wrong-cell maps."""
    train = task["train"]
    n = len(train)
    pre = ("WARNING: identical to a previous submission, so the result is the same. Change the approach.\n"
           if repeated else "")
    if cand["verified"]:
        return pre + f"Accepted: reproduces all {n} examples."
    if not cand["fails"]:  # demos pass but a test input produced no valid grid
        return pre + ("Reproduces all examples but a test input gave no valid grid (error, timeout or bad output). "
                      "Make transform robust on other inputs and submit again.")
    idx = [int(f.split(":")[0].split()[1]) for f in cand["fails"]][:max_examples]
    wrong_cells = [i for i, f in zip(idx, cand["fails"]) if "cells wrong" in f]
    got = {}
    if wrong_cells:
        res = sy.run_program(cand["code"], [train[i - 1]["input"] for i in wrong_cells])
        got = {i: r["grid"] for i, r in zip(wrong_cells, res) if r["status"] == "ok"}
    out = [pre + f"Reproduced {n - len(cand['fails'])} of {n} examples."]
    for i, f in zip(idx, cand["fails"]):
        want = train[i - 1]["output"]
        out.append(f"## Example {i} failed")
        if i in got:
            g = got[i]
            bad = sum(a != b for gr, wr in zip(g, want) for a, b in zip(gr, wr))
            total = len(want) * len(want[0])
            if total > 900:
                out.append(f"{bad} of {total} cells are wrong")
            else:
                out.append(f"{bad} of {total} cells are wrong. Expected output:\n{ls.grid_text(want)}\n"
                           f"Your output ('.' = correct cell, digit = wrong cell):\n{diff_map(g, want)}")
        else:
            out.append(f.split(": ", 1)[1])
            if "wrong shape" in f and len(want) * len(want[0]) <= 900:
                out.append(f"Expected output:\n{ls.grid_text(want)}")
    out.append("Fix the rule and submit again.")
    return "\n".join(out)


class Hooks:
    """Extension points for control rules (ARC-142). Defaults change nothing."""

    def prepare(self, messages, state):
        """Return the message list to send this turn (context trimming goes here)."""
        return messages

    def tool_result(self, name, text, state):
        """Return the text of a tool reply (nudges go here)."""
        return text


class Attempt:
    def __init__(self):
        self.messages, self.cands, self.turns = [], [], []
        self.tokens = self.python_calls = self.submits = 0
        self.status = "running"
        self.verified = False

    def log(self):
        return {"status": self.status, "verified": self.verified, "tokens": self.tokens, "turns": len(self.turns),
                "python_calls": self.python_calls, "submits": self.submits}


def _default_repl(task, persist=True):
    from repl import Repl
    return Repl(task, persist=persist)


def run_attempt(task, *, base_url, model, budget, deadline, tag="", max_turns=40, max_tokens=32768,
                attempt_tokens=120000, request_timeout=3600, temperature=0.6, effort="", thinking_budget=0,
                sandbox_timeout=10, repl_factory=None, hooks=None, on_cand=None, max_hints=1,
                tool_timeout=None):
    """One conversation. Returns an Attempt (cands feed ``synth.select_attempts``)."""
    pre = f"[task {tag}] " if tag else ""
    hooks = hooks or Hooks()
    att = Attempt()
    att.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pre + task_message(task)}]
    kwargs = {"reasoning_effort": effort} if effort else None
    seen = {}  # code -> cand, to flag identical resubmissions without re-running them
    repl, hints, think = None, 0, thinking_budget
    state = {"attempt": att}

    def submit(code):
        att.submits += 1
        key = code.strip()
        if key in seen:
            return submit_reply(seen[key], task, repeated=True)
        cand = sy.evaluate(code, task, sandbox_timeout)
        if cand["verified"] and any(o is None for o in cand["test_outputs"]):
            cand["verified"] = False  # a test input must produce a valid grid too
        cand.update(turn=len(att.turns))
        seen[key] = cand
        att.cands.append(cand)
        if on_cand:
            on_cand(cand)
        if cand["verified"]:
            att.verified = True
        return submit_reply(cand, task)

    def dispatch(call):
        nonlocal repl
        a = call["arguments"]
        code = a.get("code") if isinstance(a.get("code"), str) else ""
        if call["name"] == "python":
            att.python_calls += 1
            if not code.strip():
                return "Missing 'code' argument. Call python with a JSON object {\"code\": \"...\"}."
            try:
                repl = repl or (repl_factory or _default_repl)(task, persist=True)
                return repl.run(code, tool_timeout) if tool_timeout else repl.run(code)
            except Exception as e:  # noqa: BLE001 - a broken sandbox must not kill the attempt
                return f"[python tool error: {e!r}]"
        if call["name"] == "submit":
            if not code.strip():
                return "Missing 'code' argument. Call submit with a JSON object {\"code\": \"...\"}."
            return submit(code)
        return f"Unknown tool {call['name']!r}. Tools: python, submit."

    try:
        for _ in range(max_turns):
            if sy.clock() > deadline:
                att.status = "deadline"
                break
            left = attempt_tokens - att.tokens
            limits = budget.call_limits(min(max_tokens, left), request_timeout) if left >= 1024 else None
            if limits is None:
                att.status = "no_budget"
                break
            info = {"turn": len(att.turns)}
            att.turns.append(info)
            t0 = sy.clock()
            try:
                r = ls.chat_full(base_url, model, hooks.prepare(att.messages, state), limits[0], limits[1], temperature,
                         kwargs, tools=TOOLS, tool_choice="auto", thinking_token_budget=think)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace").lower() if e.fp else ""
                if think and e.code == 400 and any(w in body for w in ("thinking", "budget", "reasoning")):
                    think = 0  # the server cannot take the field: drop it and retry this turn
                    att.turns.pop()
                    continue
                info["error"] = repr(e)
                att.status = "error"
                break
            except Exception as e:  # noqa: BLE001
                info["error"] = repr(e)
                att.status = "error"
                break
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
            if calls:
                for c in calls:
                    text = hooks.tool_result(c["name"], dispatch(c), state)
                    att.messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
                if att.verified:
                    att.status = "verified"
                    break
                continue
            code = sy.extract_code(content) if content else None
            if code:  # a fenced program in prose counts as a submit, answered as a user message
                info["implicit_submit"] = True
                att.messages.append({"role": "user", "content": pre + submit(code)})
                if att.verified:
                    att.status = "verified"
                    break
                continue
            if hints >= max_hints:
                att.status = "prose"
                break
            hints += 1
            att.messages.append({"role": "user", "content": pre + (CUT_HINT if r["finish_reason"] == "length"
                                                                  else PROSE_HINT)})
        else:
            att.status = "max_turns"
    finally:
        if repl is not None:
            try:
                repl.close()
            except Exception as e:  # noqa: BLE001
                att.turns.append({"close_error": repr(e)})
    return att


def solve_task(task, args, deadline, on_update=None, fallback_rows=None, tag="", budget=None, repl_factory=None,
               hooks=None):
    """Up to args.attempts sequential attempts (stops once args.want_verified programs verified); returns (rows, log)."""
    budget = budget or sy.Budget(deadline, 1, 1, args.reserve_sec, args.min_call_tokens)
    fallback_rows = fallback_rows or ls.load_fallback()(task)
    cands, logs = [], []

    def update():
        rows = sy.select_attempts(task, cands, fallback_rows)
        if on_update:
            on_update(rows)
        return rows

    for _ in range(args.attempts):
        if sy.clock() > deadline or sum(c["verified"] for c in cands) >= args.want_verified:
            break
        att = run_attempt(
            task, base_url=args.base_url, model=args.model, budget=budget, deadline=deadline, tag=tag,
            max_turns=args.max_turns, max_tokens=args.max_tokens, attempt_tokens=args.task_budget,
            request_timeout=args.request_timeout, temperature=args.temperature, effort=args.effort,
            thinking_budget=args.thinking_budget, sandbox_timeout=args.sandbox_timeout,
            repl_factory=repl_factory, hooks=hooks, on_cand=lambda c: (cands.append(c), update()))
        logs.append(dict(att.log(), turns_detail=att.turns))
    rows = update()
    best = max(cands, key=lambda c: (c["verified"], c["frac"]), default=None)
    log = {"rounds": [t for lg in logs for t in lg["turns_detail"]], "attempts": [
               {k: v for k, v in lg.items() if k != "turns_detail"} for lg in logs],
           "tokens": sum(lg["tokens"] for lg in logs), "verified": sum(c["verified"] for c in cands),
           "candidates": len(cands), "final_program": best["code"] if best else None,
           "final_frac": round(best["frac"], 3) if best else None}
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
                                   tag=tid if args.tag_prompts else "", budget=budget)
        except Exception as e:  # noqa: BLE001 - one task's failure must not stop the run
            rows, log = fb[tid], {"error": repr(e), "tokens": 0, "verified": 0}
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
    ap.add_argument("--tag-prompts", type=int, default=0, help="prefix every user message with [task <id>]")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
