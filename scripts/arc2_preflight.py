"""Go/no-go gate before pushing an ARC-AGI-2 notebook to Kaggle.

python scripts/arc2_preflight.py [--fast] [--modes direct,synth,agent,hybrid,bench]
Stops at the first failing step and exits non-zero.
"""
import argparse
import base64
import importlib.util
import io
import json
import py_compile
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kaggle/arc2_synth/build_kernel.py"
MODES = ["direct", "synth", "agent", "hybrid", "bench"]
MAX_MAIN_BYTES = 1_000_000
BASE_MODULES = {"llm_solve.py", "synth.py", "solve.py", "trace.py"}
AGENT_MODULES = {"repl.py", "synth_agent.py"}


class PreflightError(Exception):
    pass


def check_metadata(meta):
    """Return a list of problems with a kernel-metadata.json dict."""
    bad = []
    if meta.get("machine_shape") != "NvidiaL4":
        bad.append(f"machine_shape is {meta.get('machine_shape')!r}, want 'NvidiaL4'")
    if str(meta.get("enable_internet")).lower() != "false":
        bad.append(f"enable_internet is {meta.get('enable_internet')!r}, want 'false'")
    if str(meta.get("is_private")).lower() != "true":
        bad.append(f"is_private is {meta.get('is_private')!r}, want 'true'")
    if str(meta.get("enable_gpu")).lower() not in ("true", "1"):
        bad.append(f"enable_gpu is {meta.get('enable_gpu')!r}, want truthy")
    if meta.get("competition_sources") != ["arc-prize-2026-arc-agi-2"]:
        bad.append(f"competition_sources is {meta.get('competition_sources')!r}")
    if not any("wheelhouse" in s for s in meta.get("dataset_sources") or []):
        bad.append("wheelhouse dataset source missing")
    if not any("qwen" in s.lower() for s in meta.get("model_sources") or []):
        bad.append("Qwen model source missing")
    return bad


def find_placeholders(src):
    return sorted(set(re.findall(r"@@[A-Z_]+@@", src)))


def bundle_modules(src):
    m = re.search(r'^BUNDLE_B64 = "([^"]*)"', src, re.MULTILINE)
    if not m:
        raise PreflightError("BUNDLE_B64 not found in main.py")
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(m.group(1))), mode="r:gz") as tf:
        return set(tf.getnames())


def build(out_dir, mode, flags=()):
    r = subprocess.run([sys.executable, str(BUILD), str(out_dir), "--mode", mode, *flags],
                       capture_output=True, text=True, check=False)
    if r.returncode:
        raise PreflightError(f"build_kernel failed for {mode}: {r.stderr[-400:]}")
    return Path(out_dir) / "main.py", Path(out_dir) / "kernel-metadata.json"


def step_ruff():
    r = subprocess.run(["uv", "run", "ruff", "check", "kaggle", "tests"], cwd=ROOT, capture_output=True, text=True, check=False)
    if r.returncode:
        raise PreflightError("ruff check failed:\n" + (r.stdout + r.stderr)[-1500:])


def step_tests(fast):
    files = ["tests/test_kaggle_launcher.py", "tests/test_kaggle_agent.py", "tests/test_kaggle_scripted_player.py"] if fast else sorted(
        str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_kaggle_*.py"))
    r = subprocess.run(["uv", "run", "pytest", "-q", *files], cwd=ROOT, capture_output=True, text=True, check=False)
    if r.returncode:
        raise PreflightError("kaggle tests failed:\n" + r.stdout[-1500:])


def step_build(mode):
    with tempfile.TemporaryDirectory() as d:
        main, metaf = build(d, mode)
        src = main.read_text()
        try:
            py_compile.compile(str(main), doraise=True)
        except py_compile.PyCompileError as e:
            raise PreflightError(f"{mode}: main.py does not compile: {e}") from e
        bad = check_metadata(json.loads(metaf.read_text()))
        if left := find_placeholders(src):
            bad.append(f"unreplaced placeholders: {left}")
        if bad:
            raise PreflightError(f"{mode}: " + "; ".join(bad))


def step_size(mode):
    with tempfile.TemporaryDirectory() as d:
        main, _ = build(d, mode)
        src = main.read_text()
        if len(src.encode()) >= MAX_MAIN_BYTES:
            raise PreflightError(f"{mode}: main.py is {len(src)} bytes, limit {MAX_MAIN_BYTES}")
        want = BASE_MODULES | (AGENT_MODULES if mode in ("agent", "hybrid", "bench") else set())
        if missing := want - bundle_modules(src):
            raise PreflightError(f"{mode}: bundle is missing {sorted(missing)}")


def _launcher_helpers():
    spec = importlib.util.spec_from_file_location("_launcher_tests", ROOT / "tests/test_kaggle_launcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step_dry_run(mode, rerun=True):
    h = _launcher_helpers()
    srv = h._server()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            extra, files = [], None
            if mode == "bench":
                cfg = tmp / "bench.json"
                cfg.write_text(json.dumps([{"name": "ag", "mode": "agent", "effort": "xhigh", "persist": 0},
                                           {"name": "dir", "mode": "direct", "effort": "xhigh"}]))
                extra = ["--npar", "2", "--bench-config", str(cfg), "--bench-tasks", "3"]
                files = {"arc-agi_evaluation_challenges.json": h._tasks(),
                         "arc-agi_evaluation_solutions.json": {t: [[[9, 8, 7], [3, 2, 1]]] for t in h._tasks()}}
            if not rerun:  # dev slice of the public eval set; submission.json must stay the valid baseline
                extra = ["--npar", "2", "--dev-tasks", "2"]
                files = {"arc-agi_evaluation_challenges.json": h._tasks(),
                         "arc-agi_evaluation_solutions.json": {t: [[[9, 8, 7], [3, 2, 1]]] for t in h._tasks()}}
            r, sub = h._run(tmp, srv, mode, extra, files, env_extra=None if rerun else {"ARC_FORCE_RERUN": "0"})
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        raise PreflightError(f"{mode}: dry run did not produce submission.json: {e}") from e
    finally:
        srv.shutdown()
    if r.returncode:
        raise PreflightError(f"{mode}: dry run exit {r.returncode}: {r.stderr[-600:]}")
    if "submission validated" not in r.stdout:
        raise PreflightError(f"{mode}: 'submission validated' line missing")
    if not rerun and "DEV score: solved 2 of 2" not in r.stdout:
        raise PreflightError(f"{mode}: dev run did not log 'DEV score: solved 2 of 2'")
    if set(sub) != set(h._tasks()):
        raise PreflightError(f"{mode}: submission.json has tasks {sorted(sub)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true", help="run only the launcher, agent and scripted-player tests")
    ap.add_argument("--modes", default=",".join(MODES))
    args = ap.parse_args(argv)
    modes = [m for m in args.modes.split(",") if m]
    if unknown := [m for m in modes if m not in MODES]:
        ap.error(f"unknown modes {unknown}")
    steps = [("ruff", step_ruff), ("tests", lambda: step_tests(args.fast))]
    for m in modes:
        steps += [(f"build:{m}", lambda m=m: step_build(m)), (f"dry-run:{m}", lambda m=m: step_dry_run(m)),
                  (f"size:{m}", lambda m=m: step_size(m))]
    if dev_mode := next((m for m in modes if m != "bench"), None):  # the other kernel run type, once
        steps.append((f"dry-run-dev:{dev_mode}", lambda: step_dry_run(dev_mode, rerun=False)))
    results, failed = [], None
    for name, fn in steps:
        try:
            fn()
            results.append((name, "PASS"))
        except PreflightError as e:
            results.append((name, "FAIL"))
            print(f"{name}: {e}", file=sys.stderr)
            failed = name
            break
    for name, status in results:
        print(f"{name:<16} {status}")
    print("PREFLIGHT OK" if not failed else f"PREFLIGHT FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
