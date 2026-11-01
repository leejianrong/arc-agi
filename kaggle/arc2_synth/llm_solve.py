# MIT No Attribution (MIT-0) — see ../arc2_baseline/solve.py for the full notice.
"""Direct-answer ARC solver over an OpenAI-compatible endpoint (local vLLM).

Stdlib only. Asks the model for two answer grids per test input, parses the
last JSON block in the reply, and writes ``submission.json`` incrementally,
pre-filled with a fallback so a crash or deadline still leaves a complete file.

    python llm_solve.py --challenges C.json --out sub.json [--solutions S.json]
"""
import argparse
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def load_fallback():
    path = os.path.join(HERE, "..", "arc2_baseline", "solve.py")
    if not os.path.exists(path):
        path = os.path.join(HERE, "solve.py")
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("arc2_fallback", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.attempts
    return lambda task: [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]]


def grid_text(g):
    return "\n".join("".join(str(c) for c in row) for row in g)


def build_prompt(task):
    parts = [
        (
            "You are solving an ARC-AGI puzzle. Each grid is rows of digits 0-9 (colors). "
            "Infer the transformation rule from the training examples, then apply it to the test input(s).\n"
        )
    ]
    for i, p in enumerate(task["train"], 1):
        parts.append(f"Training example {i}\nInput:\n{grid_text(p['input'])}\nOutput:\n{grid_text(p['output'])}\n")
    for i, t in enumerate(task["test"], 1):
        parts.append(f"Test input {i}:\n{grid_text(t['input'])}\n")
    parts.append(
        f"Give two different candidate answers for each of the {len(task['test'])} test input(s) "
        "(attempt_1 is your best answer). After reasoning, end with ONLY a ```json block of the form "
        '{"outputs": [{"attempt_1": [[...]], "attempt_2": [[...]]}, ...]} with one entry per test input, '
        "in order, each grid a list of rows of integers 0-9."
    )
    return "\n".join(parts)


def valid_grid(g):
    return (
        isinstance(g, list) and 0 < len(g) <= 30
        and all(isinstance(r, list) and 0 < len(r) <= 30 and len(r) == len(g[0]) for r in g)
        and all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 9 for r in g for c in r)
    )


def parse_outputs(text, n_test):
    """Return a list of n_test {attempt_1, attempt_2} dicts (missing/invalid -> None), or None."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = blocks[::-1] or [m for m in re.findall(r"(\{\"outputs\".*\})", text, re.DOTALL)[::-1]]
    for raw in candidates:
        try:
            outs = json.loads(raw)["outputs"]
        except (ValueError, KeyError, TypeError):
            continue
        if not isinstance(outs, list):
            continue
        rows = []
        for i in range(n_test):
            o = outs[i] if i < len(outs) and isinstance(outs[i], dict) else {}
            rows.append({k: o.get(k) if valid_grid(o.get(k)) else None for k in ("attempt_1", "attempt_2")})
        return rows
    return None


def _json_args(a):
    """Tool-call arguments arrive as a JSON string (OpenAI) or already as an object (semula); always return a dict."""
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        try:
            v = json.loads(a) if a.strip() else {}
        except ValueError:
            return {"_raw": a}
        return v if isinstance(v, dict) else {"_raw": a}
    return {} if a is None else {"_raw": a}


_MARKUP = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)


def recover_markup_calls(text):
    """Tool calls a server failed to parse, written as <tool_call><function=NAME><parameter=K>V... markup."""
    calls = []
    for i, m in enumerate(_MARKUP.finditer(text or "")):
        params = {k: v.strip("\n") for k, v in re.findall(r"<parameter=([\w.-]+)>(.*?)</parameter>", m.group(2), re.DOTALL)}
        calls.append({"id": f"recovered_{i}", "name": m.group(1), "arguments": params})
    return calls, _MARKUP.sub("", text or "").strip()


def chat_full(base_url, model, prompt, max_tokens, timeout, temperature, template_kwargs=None, tools=None,
              tool_choice=None, thinking_token_budget=None):
    """One chat call. Returns {content, reasoning, tool_calls:[{id,name,arguments(dict)}], finish_reason,
    prompt_tokens, completion_tokens, usage}. Accepts the OpenAI body or a bare {"message": {...}}."""
    d = _post(base_url, model, prompt, max_tokens, timeout, temperature, template_kwargs, tools, tool_choice,
              thinking_token_budget)
    if isinstance(d.get("choices"), list) and d["choices"]:
        choice = d["choices"][0]
    else:
        choice = {"message": d["message"], "finish_reason": d.get("finish_reason")}
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or tc
        calls.append({"id": tc.get("id") or f"call_{i}", "name": fn.get("name", ""),
                      "arguments": _json_args(fn.get("arguments"))})
    if not calls and "<function=" in (content + reasoning):
        calls, _ = recover_markup_calls(content or reasoning)
        if calls and content:
            content = recover_markup_calls(content)[1]
    usage = dict(d.get("usage") or {})
    return {"content": content, "reasoning": reasoning, "tool_calls": calls, "usage": usage,
            "finish_reason": choice.get("finish_reason") or ("tool_calls" if calls else "stop"),
            "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}


def _post(base_url, model, prompt, max_tokens, timeout, temperature, template_kwargs, tools=None, tool_choice=None,
          thinking_token_budget=None):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    payload = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    if thinking_token_budget:
        payload["thinking_token_budget"] = int(thinking_token_budget)
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(base_url, model, prompt, max_tokens, timeout, temperature, template_kwargs=None, tools=None,
         tool_choice=None, thinking_token_budget=None):
    """prompt: a user string, or a full OpenAI-style messages list. Returns (text, usage); usage also carries
    finish_reason, reasoning_text and, when the model called tools, tool_calls."""
    r = chat_full(base_url, model, prompt, max_tokens, timeout, temperature, template_kwargs, tools, tool_choice,
                  thinking_token_budget)
    usage = dict(r["usage"])
    usage["finish_reason"] = r["finish_reason"]
    usage["reasoning_chars"] = len(r["reasoning"])
    usage["reasoning_text"] = r["reasoning"]
    if r["tool_calls"]:
        usage["tool_calls"] = r["tool_calls"]
    return r["content"], usage


FORCE_TAIL = (
    "\n\nTime is up. I must stop reasoning and commit to my best final answer now.\n</think>\n\n"
    '```json\n{"outputs": ['
)


def force_answer(base_url, model, prompt, reasoning, max_tokens, timeout, temperature):
    """Continue a truncated reasoning trace straight into the JSON answer (budget forcing).

    Uses raw ChatML via /completions, since the chat endpoint cannot prefill a reasoning trace.
    """
    raw = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n{reasoning}{FORCE_TAIL}"
    body = json.dumps({"model": model, "prompt": raw, "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    choice = d["choices"][0]
    return FORCE_TAIL.split("</think>")[1] + choice.get("text", ""), d.get("usage", {})


def merge(fallback_rows, parsed):
    out = []
    for fb, p in zip(fallback_rows, parsed or [{}] * len(fallback_rows)):
        a1 = p.get("attempt_1") or fb["attempt_1"]
        a2 = p.get("attempt_2") or fb["attempt_2"]
        if a2 == a1:
            a2 = fb["attempt_2"] if fb["attempt_2"] != a1 else a1
        out.append({"attempt_1": a1, "attempt_2": a2})
    return out


def run(args):
    with open(args.challenges) as f:
        challenges = json.load(f)
    ids = sorted(challenges)[: args.limit] if args.limit else sorted(challenges)
    fallback = load_fallback()
    sub = {tid: fallback(challenges[tid]) for tid in challenges}
    lock = threading.Lock()
    stats = {"done": 0, "parsed": 0, "errors": 0, "completion_tokens": 0, "elapsed": []}
    deadline = time.time() + args.deadline_min * 60
    os.makedirs(args.log_dir, exist_ok=True)

    def flush():
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sub, f)
        os.replace(tmp, args.out)

    seen = {"tok": 0.0, "sec": 0.0}  # observed per-stream decode rate, ratio of sums

    def limits():
        """(max_tokens, timeout) that let the call finish, and leave room for a forced answer, before the deadline.
        A non-streaming call that outlives the deadline returns nothing, so the cap must follow the time left."""
        with lock:
            remaining = max(1, len(ids) - stats["done"])
            rate = seen["tok"] / seen["sec"] if seen["sec"] else args.default_rate
        share = (deadline - time.time() - args.reserve_sec) / math.ceil(remaining / args.n_parallel)
        cap = max(512, min(args.max_tokens, int(share * rate)))
        timeout = max(30, min(args.request_timeout, share * 1.6 + 30, deadline - time.time() - args.reserve_sec / 2))
        return cap, timeout

    def work(tid):
        task = challenges[tid]
        if time.time() > deadline:
            return
        t0 = time.time()
        text, usage, err, retried = "", {}, None, False
        prompt = build_prompt(task)
        kwargs = {"reasoning_effort": args.effort} if args.effort else None
        try:
            cap, timeout = limits()
            text, usage = chat(args.base_url, args.model, prompt, cap, timeout, args.temperature, kwargs)
            with lock:
                seen["tok"] += usage.get("completion_tokens", 0)
                seen["sec"] += time.time() - t0
        except Exception as e:  # noqa: BLE001 - one task's failure must not stop the run
            err = repr(e)
        parsed = parse_outputs(text, len(task["test"])) if text else None
        reasoning = usage.pop("reasoning_text", "")
        if not parsed and args.force_answer and reasoning and time.time() < deadline:
            retried = True
            try:
                text2, usage2 = force_answer(args.base_url, args.model, prompt, reasoning, args.force_max_tokens,
                                             max(30, min(900, deadline - time.time())), args.temperature)
                usage["retry_completion_tokens"] = usage2.get("completion_tokens", 0)
                parsed = parse_outputs(text2, len(task["test"]))
                text = text + "\n=====FORCED=====\n" + text2
            except Exception as e:  # noqa: BLE001
                err = (err or "") + " retry:" + repr(e)
        dt = time.time() - t0
        usage["retried"] = retried
        with lock:
            if parsed:
                sub[tid] = merge(sub[tid], parsed)
                stats["parsed"] += 1
            if err:
                stats["errors"] += 1
            stats["done"] += 1
            stats["completion_tokens"] += usage.get("completion_tokens", 0) + usage.get("retry_completion_tokens", 0)
            stats["elapsed"].append(dt)
            flush()
            print(f"[{stats['done']}/{len(ids)}] {tid} parsed={bool(parsed)} err={err} "
                  f"tok={usage.get('completion_tokens')} fin={usage.get('finish_reason')} retry={retried} "
                  f"{dt:.0f}s", flush=True)
        with open(os.path.join(args.log_dir, f"{tid}.json"), "w") as f:
            json.dump({"usage": usage, "error": err, "elapsed": dt, "text": text, "parsed": bool(parsed),
                       "reasoning_tail": reasoning[-1500:]}, f)

    flush()
    t_start = time.time()
    with ThreadPoolExecutor(args.n_parallel) as ex:
        list(ex.map(work, ids))
    wall = time.time() - t_start
    summary = {"tasks": len(ids), "parsed": stats["parsed"], "errors": stats["errors"], "wall_s": round(wall),
               "completion_tokens": stats["completion_tokens"],
               "tok_per_s": round(stats["completion_tokens"] / wall, 1) if wall else 0}
    if args.solutions:
        with open(args.solutions) as f:
            sol = json.load(f)
        hits = sum(1 for tid in ids for row, s in zip(sub[tid], sol[tid]) if s in (row["attempt_1"], row["attempt_2"]))
        total = sum(len(sol[tid]) for tid in ids)
        summary["solved_test_inputs"] = f"{hits}/{total}"
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
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--effort", default="medium", help="reasoning_effort template kwarg ('' to omit)")
    ap.add_argument("--force-answer", type=int, default=1)
    ap.add_argument("--force-max-tokens", type=int, default=8192)
    ap.add_argument("--request-timeout", type=int, default=3600)
    ap.add_argument("--deadline-min", type=float, default=60)
    ap.add_argument("--reserve-sec", type=float, default=300, help="kept free before the deadline for forced answers")
    ap.add_argument("--default-rate", type=float, default=12.0, help="tokens/s per stream assumed until calls finish")
    ap.add_argument("--log-dir", default="llm_logs")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
