# MIT No Attribution (MIT-0) — see ../arc2_baseline/solve.py for the full notice.
"""Program-synthesis ARC solver: propose -> sandboxed execute -> verify -> diff feedback -> refine.

Stdlib only; reuses ``llm_solve`` (ships next to this file). The model writes
``def transform(grid)``; a program is VERIFIED when it reproduces every train
output. Verified programs are applied to the test inputs and the two most common
distinct outputs become the attempts; partial programs, a direct-answer result
and the baseline fill any remaining slot. Candidate code never runs in-process.

    python synth.py --challenges C.json --out sub.json [--solutions S.json]
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_solve as ls

# Runs in the child (`python -I -c`). Limits are set here, not via preexec_fn, which is
# unsafe from a threaded parent. Candidate prints go to /dev/null; only our JSON hits stdout.
_RUNNER = r"""
import copy, json, os, sys, traceback
req = json.load(sys.stdin)
out = sys.stdout
sys.stdout = open(os.devnull, "w")
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (req["mem"], req["mem"]))
    resource.setrlimit(resource.RLIMIT_CPU, (req["cpu"], req["cpu"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
except Exception:
    pass

def tail(): return "".join(traceback.format_exc().splitlines(True)[-4:])[-600:]

def norm(r):
    if hasattr(r, "tolist"): r = r.tolist()
    if not isinstance(r, (list, tuple)): raise TypeError("transform returned %s, not a grid" % type(r).__name__)
    g = [list(x.tolist() if hasattr(x, "tolist") else x) for x in r]
    if not (0 < len(g) <= 30 and all(0 < len(x) <= 30 and len(x) == len(g[0]) for x in g)):
        raise ValueError("invalid grid shape (need rectangular, 1..30 x 1..30)")
    if not all(type(c) is int and 0 <= c <= 9 for x in g for c in x):
        raise ValueError("grid cells must be ints 0..9")
    return g

res = []
try:
    ns = {"__name__": "__candidate__"}
    exec(compile(req["code"], "<candidate>", "exec"), ns)
    fn = ns["transform"]
except BaseException:
    res = [{"status": "exception", "error": tail()} for _ in req["grids"]]
else:
    for g in req["grids"]:
        try:
            res.append({"status": "ok", "grid": norm(fn(copy.deepcopy(g)))})
        except BaseException as e:
            res.append({"status": "invalid" if isinstance(e, (TypeError, ValueError)) and "grid" in str(e)
                        else "exception", "error": tail()})
out.write(json.dumps(res))
"""


def run_program(code, grids, timeout_s=10, mem_mb=3072):
    """Run ``transform`` over grids in a fresh `python -I` process.

    Returns one dict per grid: {"status": ok|exception|timeout|invalid, "grid"|"error"}.
    """
    n = len(grids)
    req = json.dumps({"code": code, "grids": grids, "mem": mem_mb << 20, "cpu": int(timeout_s) + 2})
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.Popen([sys.executable, "-I", "-c", _RUNNER], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, start_new_session=True)
    try:
        stdout, _ = proc.communicate(req.encode(), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        return [{"status": "timeout", "error": f"timed out after {timeout_s}s"}] * n
    try:
        res = json.loads(stdout[:4_000_000])
        if not (isinstance(res, list) and len(res) == n):
            raise ValueError("bad result list")
    except ValueError:
        return [{"status": "exception", "error": f"no valid output (exit code {proc.returncode}; "
                 "out of memory/CPU limit, crash, or the program wrote to stdout)"}] * n
    for r in res:  # never trust the child's output
        if r.get("status") == "ok" and not ls.valid_grid(r.get("grid")):
            r.update(status="invalid", error="invalid grid")
    return res


def score_pair(res, expected):
    """Return (n_correct_cells, total_cells, failure_text or None) for one train pair."""
    total = len(expected) * len(expected[0])
    if res["status"] != "ok":
        return 0, total, f"{res['status']}: {res.get('error', '').strip()}"
    got = res["grid"]
    if (len(got), len(got[0])) != (len(expected), len(expected[0])):
        return 0, total, f"wrong shape: got {len(got)}x{len(got[0])}, expected {len(expected)}x{len(expected[0])}"
    bad = [(r, c, got[r][c], expected[r][c]) for r in range(len(got)) for c in range(len(got[0]))
           if got[r][c] != expected[r][c]]
    if not bad:
        return total, total, None
    shown = ", ".join(f"({r},{c}) got {g} expected {e}" for r, c, g, e in bad[:6])
    pairs = Counter((g, e) for _, _, g, e in bad).most_common(5)
    by_color = ", ".join(f"{g}->{e} x{n}" for (g, e), n in pairs)
    return total - len(bad), total, (f"{len(bad)} of {total} cells wrong; first: {shown}; "
                                     f"most common (got->expected): {by_color}")


def evaluate(code, task, timeout_s=10):
    """Run code on train + test inputs. Returns a candidate dict."""
    train, tests = task["train"], task["test"]
    res = run_program(code, [p["input"] for p in train] + [t["input"] for t in tests], timeout_s)
    ok = tot = 0
    fails, detail = [], None
    for i, (p, r) in enumerate(zip(train, res), 1):
        c, t, why = score_pair(r, p["output"])
        ok, tot = ok + c, tot + t
        if why:
            fails.append(f"Example {i}: {why}")
            if detail is None and r["status"] == "ok" and "wrong shape" not in why:
                detail = (i, r["grid"], p["output"])
    return {"code": code, "verified": not fails, "frac": ok / tot if tot else 0.0, "fails": fails, "detail": detail,
            "test_outputs": [r["grid"] if r["status"] == "ok" else None for r in res[len(train):]]}


def feedback_text(cand, n_train, repeated=False):
    passed = n_train - len(cand["fails"])
    out = []
    if repeated:
        out.append("WARNING: your program is identical to your previous one, so it fails in exactly the same way. "
                   "Change the approach, do not resubmit it.\n")
    out.append(f"Your program passed {passed} of {n_train} training examples.\n" + "\n".join(cand["fails"]))
    if cand.get("detail"):
        i, got, want = cand["detail"]
        out.append(f"\nExample {i}, expected output:\n{ls.grid_text(want)}\nExample {i}, your output:\n{ls.grid_text(got)}")
    out.append("\nA program with the right output shape can still have the wrong rule; look at which cells differ and "
               "why. Prefer a simple, general rule over special cases for these examples. Think briefly, then reply "
               "with the full corrected program in ONE ```python block.")
    return "\n".join(out)


def build_prompt(task):
    parts = ["You are solving an ARC-AGI puzzle by writing a program. Each grid is rows of digits 0-9 (colors).\n"]
    for i, p in enumerate(task["train"], 1):
        parts.append(f"Training example {i}\nInput:\n{ls.grid_text(p['input'])}\nOutput:\n{ls.grid_text(p['output'])}\n")
    for i, t in enumerate(task["test"], 1):
        parts.append(f"Test input {i}:\n{ls.grid_text(t['input'])}\n")
    parts.append(
        "Infer the rule, then write a Python function `def transform(grid):` that takes the input grid "
        "(list of lists of ints) and returns the output grid (list of lists of ints). It must reproduce every "
        "training output exactly and generalize to the test input(s). You may `import numpy`, but convert the "
        "result back to a list of lists. Reason first, then end with ONE ```python block containing the "
        "complete program."
    )
    return "\n".join(parts)


def extract_code(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = [b for b in re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL) if "def transform" in b]
    if blocks:
        return blocks[-1].strip()
    m = re.search(r"```(?:python|py)?\s*\n(def transform.*)", text, re.DOTALL)  # unclosed (budget-forced) block
    return m.group(1).strip() if m else None


CODE_TAIL = (
    "\n\nTime is up. I must stop reasoning and write my best program now.\n</think>\n\n"
    "```python\ndef transform(grid):\n"
)


def force_code(base_url, model, messages, reasoning, max_tokens, timeout, temperature):
    """Continue a truncated reasoning trace straight into the code block (raw ChatML via /completions)."""
    raw = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    raw += f"<|im_start|>assistant\n<think>\n{reasoning}{CODE_TAIL}"
    body = json.dumps({"model": model, "prompt": raw, "max_tokens": max_tokens, "temperature": temperature,
                       "stop": ["```"]}).encode()
    req = ls.urllib.request.Request(base_url.rstrip("/") + "/completions", body, {"Content-Type": "application/json"})
    with ls.urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return "def transform(grid):\n" + d["choices"][0].get("text", ""), d.get("usage", {}).get("completion_tokens", 0)


def select_attempts(task, cands, fallback_rows, direct_rows=None):
    """Two attempts per test input: verified majority, best partials, direct answer, fallback.

    cands: candidate dicts in round order. direct_rows: optional llm_solve-style rows
    ({"attempt_1","attempt_2"} per test input, entries may be None).
    """
    verified = [c for c in cands if c["verified"]]
    partial = sorted((c for c in cands if not c["verified"]), key=lambda c: -c["frac"])  # stable: earlier first
    rows = []
    for i in range(len(task["test"])):
        votes = Counter()
        for c in verified:
            if c["test_outputs"][i] is not None:
                votes[json.dumps(c["test_outputs"][i])] += 1
        # most_common is stable on insertion order, so ties go to the earlier round
        picks = [json.loads(k) for k, _ in votes.most_common()]
        picks += [c["test_outputs"][i] for c in partial]
        if direct_rows:
            picks += [direct_rows[i].get("attempt_1"), direct_rows[i].get("attempt_2")]
        picks += [fallback_rows[i]["attempt_1"], fallback_rows[i]["attempt_2"]]
        chosen = []
        for g in picks:
            if g is not None and g not in chosen:
                chosen.append(g)
            if len(chosen) == 2:
                break
        rows.append({"attempt_1": chosen[0], "attempt_2": chosen[-1]})
    return rows


def synthesize(task, args, deadline, on_update=None, direct_rows=None, fallback_rows=None, tag=""):
    """Run up to args.chains refinement chains; returns (rows, log)."""
    fallback_rows = fallback_rows or ls.load_fallback()(task)
    n_train = len(task["train"])
    pre = f"[task {tag}] " if tag else ""  # lets semula answerers split tasks by the first 200 chars
    prompt = pre + build_prompt(task)
    kwargs = {"reasoning_effort": args.effort} if args.effort else None
    cands, rounds, tokens = [], [], 0

    def update():
        rows = select_attempts(task, cands, fallback_rows, direct_rows)
        if on_update:
            on_update(rows)
        return rows

    for chain in range(args.chains):
        messages = [{"role": "user", "content": prompt}]
        prev_code = None
        for rnd in range(args.rounds):
            left = args.task_budget - tokens
            if left < 1024 or time.time() > deadline or sum(c["verified"] for c in cands) >= args.want_verified:
                break
            info = {"chain": chain, "round": rnd}
            rounds.append(info)
            text, reasoning, err = "", "", None
            try:
                text, usage = ls.chat(args.base_url, args.model, messages, min(args.max_tokens, left),
                                      max(30, min(args.request_timeout, deadline - time.time())),
                                      args.temperature, kwargs)
                reasoning = usage.pop("reasoning_text", "")
                tokens += usage.get("completion_tokens", 0)
                info.update(tokens=usage.get("completion_tokens", 0), finish=usage.get("finish_reason"))
            except Exception as e:  # noqa: BLE001 - one task's failure must not stop the run
                err = repr(e)
            code = extract_code(text) if text else None
            if code is None and reasoning and args.force_answer and time.time() < deadline:
                try:
                    text2, t2 = force_code(args.base_url, args.model, messages, reasoning, args.force_max_tokens,
                                           max(30, min(900, deadline - time.time())), args.temperature)
                    tokens += t2
                    code = extract_code("```python\n" + text2)
                    info.update(forced=True, tokens=info.get("tokens", 0) + t2)
                except Exception as e:  # noqa: BLE001
                    err = (err or "") + " force:" + repr(e)
            if err:
                info["error"] = err
            if code is None:
                info["status"] = "no_code"
                if err:  # endpoint failure: retrying the same turn immediately rarely helps
                    break
                messages += [{"role": "assistant", "content": "(no program)"},
                             {"role": "user", "content": pre + "No ```python block with `def transform(grid):` was "
                              "found. Reply with the complete program in ONE ```python block."}]
                continue
            cand = evaluate(code, task, args.sandbox_timeout)
            cand.update(chain=chain, round=rnd)
            cands.append(cand)
            info.update(status="verified" if cand["verified"] else "failed", frac=round(cand["frac"], 3))
            update()
            if cand["verified"]:
                break
            messages += [{"role": "assistant", "content": f"```python\n{code}\n```"},
                         {"role": "user", "content": pre + feedback_text(cand, n_train, code == prev_code)}]
            messages = messages[:1] + messages[1:][-4:]  # keep prompt + last two code/feedback pairs
            prev_code = code
    rows = update()
    best = max((c for c in cands), key=lambda c: (c["verified"], c["frac"]), default=None)
    log = {"rounds": rounds, "tokens": tokens, "verified": sum(c["verified"] for c in cands),
           "candidates": len(cands), "final_program": best["code"] if best else None,
           "final_frac": round(best["frac"], 3) if best else None}
    return rows, log


def run(args):
    with open(args.challenges) as f:
        challenges = json.load(f)
    ids = sorted(challenges)[: args.limit] if args.limit else sorted(challenges)
    fallback = ls.load_fallback()
    fb = {tid: fallback(challenges[tid]) for tid in challenges}
    sub = {tid: fb[tid] for tid in challenges}
    lock = threading.Lock()
    stats = {"done": 0, "verified": 0, "tokens": 0}
    deadline = time.time() + args.deadline_min * 60
    os.makedirs(args.log_dir, exist_ok=True)

    def flush():
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sub, f)
        os.replace(tmp, args.out)

    def set_rows(tid, rows):
        with lock:
            sub[tid] = rows
            flush()

    def work(tid):
        if time.time() > deadline:
            return
        t0 = time.time()
        try:
            rows, log = synthesize(challenges[tid], args, deadline, lambda r: set_rows(tid, r), fallback_rows=fb[tid],
                                   tag=tid if args.tag_prompts else "")
        except Exception as e:  # noqa: BLE001
            rows, log = fb[tid], {"error": repr(e), "tokens": 0, "verified": 0}
        dt = time.time() - t0
        with lock:
            sub[tid] = rows
            stats["done"] += 1
            stats["verified"] += bool(log.get("verified"))
            stats["tokens"] += log.get("tokens", 0)
            flush()
            print(f"[{stats['done']}/{len(ids)}] {tid} verified={log.get('verified')} tok={log.get('tokens')} "
                  f"rounds={len(log.get('rounds', []))} {dt:.0f}s", flush=True)
        with open(os.path.join(args.log_dir, f"{tid}.json"), "w") as f:
            json.dump(dict(log, elapsed=dt), f)

    flush()
    t_start = time.time()
    with ThreadPoolExecutor(args.n_parallel) as ex:
        list(ex.map(work, ids))
    wall = time.time() - t_start
    summary = {"tasks": len(ids), "done": stats["done"], "verified": stats["verified"], "wall_s": round(wall),
               "completion_tokens": stats["tokens"], "tok_per_s": round(stats["tokens"] / wall, 1) if wall else 0}
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
    ap.add_argument("--task-budget", type=int, default=96000, help="completion tokens per task, all calls")
    ap.add_argument("--rounds", type=int, default=3, help="propose+refine calls per chain")
    ap.add_argument("--chains", type=int, default=2, help="independent chains per task")
    ap.add_argument("--want-verified", type=int, default=2, help="stop a task after this many verified programs")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--effort", default="medium", help="reasoning_effort template kwarg ('' to omit)")
    ap.add_argument("--force-answer", type=int, default=1)
    ap.add_argument("--force-max-tokens", type=int, default=4096)
    ap.add_argument("--sandbox-timeout", type=float, default=10)
    ap.add_argument("--request-timeout", type=int, default=3600)
    ap.add_argument("--deadline-min", type=float, default=60)
    ap.add_argument("--log-dir", default="synth_logs")
    ap.add_argument("--tag-prompts", type=int, default=0, help="prefix every user message with [task <id>]")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
