# MIT No Attribution (MIT-0). Build the single-file Kaggle kernel from launcher.py.tmpl.
"""python build_kernel.py OUT_DIR [--limit N] [--budget-min M] [--npar N] [--max-tokens T] [--mtp K] [--machine S]"""
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
ap.add_argument("--machine", default="")
ap.add_argument("--slug", default="arc-agi-2-qwen-direct")
a = ap.parse_args()

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    tf.add(os.path.join(HERE, "llm_solve.py"), arcname="llm_solve.py")
    tf.add(os.path.join(HERE, "..", "arc2_baseline", "solve.py"), arcname="solve.py")
src = open(os.path.join(HERE, "launcher.py.tmpl")).read()
for k, v in {"BUNDLE": base64.b64encode(buf.getvalue()).decode(), "LIMIT": a.limit, "BUDGET": a.budget_min,
             "NPAR": a.npar, "MAX_TOKENS": a.max_tokens, "MTP": a.mtp}.items():
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
