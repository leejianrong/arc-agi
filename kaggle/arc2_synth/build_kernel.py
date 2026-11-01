# MIT No Attribution (MIT-0). Build the single-file Kaggle kernel from launcher.py.tmpl.
"""python build_kernel.py OUT_DIR [--limit N] [--budget-min M] [--npar N] [--max-tokens T] [--mtp K] [--machine S] [--mode direct|synth|agent|hybrid|bench] [--effort E] [--max-model-len N]
[--thinking-budget N] [--attempts N] [--max-turns N] [--task-budget N] [--bench-config PATH] [--bench-tasks N] [--bench-budget-min M]
[--dev-tasks N] [--dev-budget-min M] [--hybrid-direct-frac F] [--agent-args "..."]"""
import argparse
import base64
import io
import json
import os
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--budget-min", type=int, default=600)
ap.add_argument("--npar", type=int, default=32)
ap.add_argument("--max-tokens", type=int, default=40000)
ap.add_argument("--mtp", type=int, default=2)
ap.add_argument("--machine", default="NvidiaL4", help="kernel machine_shape; the competition L4 pool uses NvidiaL4 (NvidiaL4X4 gave 2x T4)")
ap.add_argument("--slug", default="arc-agi-2-qwen-direct")
ap.add_argument("--mode", choices=["direct", "synth", "agent", "hybrid", "bench"], default="direct")
ap.add_argument("--synth-args", default="", help="extra synth.py flags, space separated")
ap.add_argument("--effort", default=None, help="reasoning_effort; default xhigh for agent mode, medium otherwise")
ap.add_argument("--max-model-len", type=int, default=None, help="vLLM context; default 98304 for agent mode, 65536 otherwise")
ap.add_argument("--thinking-budget", type=int, default=0, help="agent: thinking_token_budget per turn (0 = omit)")
ap.add_argument("--attempts", type=int, default=1, help="agent: sequential attempts per task")
ap.add_argument("--max-turns", type=int, default=40, help="agent: model turns per attempt")
ap.add_argument("--task-budget", type=int, default=120000, help="agent: completion tokens per attempt")
ap.add_argument("--bench-config", default=None, help="bench: JSON list of {name, mode, effort, ...extra solver args}")
ap.add_argument("--bench-tasks", type=int, default=12, help="bench: public-eval tasks in the fixed sample")
ap.add_argument("--bench-budget-min", type=int, default=150, help="bench: minutes for all configs after startup")
ap.add_argument("--dev-tasks", type=int, default=4, help="non-rerun runs: first N public-eval tasks solved into dev_submission.json")
ap.add_argument("--dev-budget-min", type=int, default=20, help="non-rerun runs: minutes after startup for the dev slice")
ap.add_argument("--hybrid-direct-frac", type=float, default=0.45,
                help="hybrid: share of the post-startup budget for the direct pass over all tasks; the agent gets the rest")
ap.add_argument("--agent-args", default="", help="extra synth_agent.py flags, space separated (agent/hybrid), e.g. '--reserve-sec 60'")
a = ap.parse_args()
DEFAULT_BENCH = [{"name": "agent_xhigh_persist", "mode": "agent", "effort": "xhigh", "persist": 1},
                 {"name": "agent_xhigh_fresh", "mode": "agent", "effort": "xhigh", "persist": 0},
                 {"name": "direct_xhigh", "mode": "direct", "effort": "xhigh"}]
bench = []
if a.mode == "bench":
    if a.bench_config:
        with open(a.bench_config) as f:
            bench = json.load(f)
    bench = bench or DEFAULT_BENCH
agent = a.mode in ("agent", "hybrid") or any(c.get("mode") == "agent" for c in bench)
effort = a.effort if a.effort is not None else ("xhigh" if agent else "medium")
max_len = a.max_model_len or (98304 if agent else 65536)
agent_args = ["--thinking-budget", a.thinking_budget, "--attempts", a.attempts, "--max-turns", a.max_turns,
              "--task-budget", a.task_budget, *a.agent_args.split()]

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    tf.add(os.path.join(HERE, "llm_solve.py"), arcname="llm_solve.py")
    tf.add(os.path.join(HERE, "synth.py"), arcname="synth.py")
    tf.add(os.path.join(HERE, "repl.py"), arcname="repl.py")
    tf.add(os.path.join(HERE, "synth_agent.py"), arcname="synth_agent.py")
    tf.add(os.path.join(HERE, "trace.py"), arcname="trace.py")
    tf.add(os.path.join(HERE, "..", "arc2_baseline", "solve.py"), arcname="solve.py")
with open(os.path.join(HERE, "launcher.py.tmpl")) as f:
    src = f.read()
for k, v in {"BUNDLE": base64.b64encode(buf.getvalue()).decode(), "LIMIT": a.limit, "BUDGET": a.budget_min,
             "NPAR": a.npar, "MAX_TOKENS": a.max_tokens, "MTP": a.mtp, "MODE": a.mode,
             "SYNTH_ARGS": repr(a.synth_args.split()), "AGENT_ARGS": repr([str(x) for x in agent_args]),
             "EFFORT": effort, "MAX_LEN": max_len, "BENCH_CONFIGS": repr(bench), "BENCH_TASKS": a.bench_tasks,
             "BENCH_BUDGET": a.bench_budget_min, "DEV_TASKS": a.dev_tasks, "DEV_BUDGET": a.dev_budget_min,
             "HYBRID_FRAC": a.hybrid_direct_frac}.items():
    src = src.replace(f"@@{k}@@", str(v))
os.makedirs(a.out_dir, exist_ok=True)
with open(os.path.join(a.out_dir, "main.py"), "w") as f:
    f.write(src)
meta = {
    "id": f"leejianrong/{a.slug}", "title": a.slug, "code_file": "main.py", "language": "python",
    "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
    "enable_internet": "false", "dataset_sources": ["mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"],
    "competition_sources": ["arc-prize-2026-arc-agi-2"], "kernel_sources": [],
    "model_sources": ["mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1"],
}
if a.machine:
    meta["machine_shape"] = a.machine
with open(os.path.join(a.out_dir, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("built", os.path.join(a.out_dir, "main.py"), len(src), "bytes")
