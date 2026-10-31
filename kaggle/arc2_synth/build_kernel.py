# MIT No Attribution (MIT-0). Build the single-file Kaggle kernel from launcher.py.tmpl.
"""python build_kernel.py OUT_DIR [--limit N] [--budget-min M] [--npar N] [--max-tokens T] [--mtp K] [--machine S] [--mode direct|synth|agent] [--effort E] [--max-model-len N]
[--thinking-budget N] [--attempts N] [--max-turns N] [--task-budget N]"""
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
ap.add_argument("--mode", choices=["direct", "synth", "agent"], default="direct")
ap.add_argument("--synth-args", default="", help="extra synth.py flags, space separated")
ap.add_argument("--effort", default=None, help="reasoning_effort; default xhigh for agent mode, medium otherwise")
ap.add_argument("--max-model-len", type=int, default=None, help="vLLM context; default 98304 for agent mode, 65536 otherwise")
ap.add_argument("--thinking-budget", type=int, default=0, help="agent: thinking_token_budget per turn (0 = omit)")
ap.add_argument("--attempts", type=int, default=1, help="agent: sequential attempts per task")
ap.add_argument("--max-turns", type=int, default=40, help="agent: model turns per attempt")
ap.add_argument("--task-budget", type=int, default=120000, help="agent: completion tokens per attempt")
a = ap.parse_args()
agent = a.mode == "agent"
effort = a.effort if a.effort is not None else ("xhigh" if agent else "medium")
max_len = a.max_model_len or (98304 if agent else 65536)
agent_args = ["--thinking-budget", a.thinking_budget, "--attempts", a.attempts, "--max-turns", a.max_turns,
              "--task-budget", a.task_budget]

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    tf.add(os.path.join(HERE, "llm_solve.py"), arcname="llm_solve.py")
    tf.add(os.path.join(HERE, "synth.py"), arcname="synth.py")
    tf.add(os.path.join(HERE, "repl.py"), arcname="repl.py")
    tf.add(os.path.join(HERE, "synth_agent.py"), arcname="synth_agent.py")
    tf.add(os.path.join(HERE, "..", "arc2_baseline", "solve.py"), arcname="solve.py")
with open(os.path.join(HERE, "launcher.py.tmpl")) as f:
    src = f.read()
for k, v in {"BUNDLE": base64.b64encode(buf.getvalue()).decode(), "LIMIT": a.limit, "BUDGET": a.budget_min,
             "NPAR": a.npar, "MAX_TOKENS": a.max_tokens, "MTP": a.mtp, "MODE": a.mode,
             "SYNTH_ARGS": repr(a.synth_args.split()), "AGENT_ARGS": repr([str(x) for x in agent_args]),
             "EFFORT": effort, "MAX_LEN": max_len}.items():
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
