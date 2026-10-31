#!/usr/bin/env python3
# MIT No Attribution (MIT-0) — see ../kaggle/arc2_baseline/solve.py for the full notice.
"""Bundle ARC Agent Watch traces into one self-contained HTML file and open it.

    python scripts/arc_watch.py bundle PATH [PATH ...] -o OUT.html
    python scripts/arc_watch.py open PATH [PATH ...] [--out OUT.html] [--no-browser]
    python scripts/arc_watch.py kaggle-output SLUG_OR_DIR [--out OUT.html] [--no-browser]

PATH is a trace ``.jsonl``, a ``bench_results.json``, or a directory (``trace*.jsonl`` and
``bench_results.json`` are collected inside it and one directory level below, so a downloaded
Kaggle output, or an unpacked RunPod ``result.tar.gz`` with its ``out/`` folder, works as is).
The viewer (tools/agent_watch) is inlined; the files travel as JSON in
``<script type="application/json" id="arc-watch-data">`` and a small injected shim feeds them to
the viewer's file input, so viewer.js needs no change. Stdlib only.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "tools" / "agent_watch"

SHIM = """(function () {
  var el = document.getElementById('arc-watch-data'), inp = document.getElementById('file');
  if (!el || !inp || typeof DataTransfer === 'undefined') return;
  var data; try { data = JSON.parse(el.textContent); } catch (e) { return; }
  var dt = new DataTransfer();
  (data.files || []).forEach(function (f) { dt.items.add(new File([f.text], f.name, { type: 'text/plain' })); });
  if (!dt.files.length) return;
  inp.files = dt.files;
  inp.dispatchEvent(new Event('change', { bubbles: true }));
})();"""


def _load_trace_mod():
    spec = importlib.util.spec_from_file_location("arc2_trace", ROOT / "kaggle" / "arc2_synth" / "trace.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wanted(p):
    return p.is_file() and (p.name == "bench_results.json" or (p.name.startswith("trace") and p.suffix == ".jsonl"))


def collect(paths):
    """Expand PATHs into a sorted, de-duplicated list of files; raises FileNotFoundError on a bad PATH."""
    out = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for pat in ("*", "*/*"):
                out += sorted(c for c in p.glob(pat) if _wanted(c))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(raw)
    seen, uniq = set(), []
    for f in out:
        k = f.resolve()
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq


def _label(f, taken):
    name = f.name
    if name in taken:  # same basename from different dirs: keep the parent so cfg/file stays readable
        name = f"{f.parent.name}_{f.name}"
    taken.add(name)
    return name


def _safe_script(text, tag):
    return text.replace(f"</{tag}", f"<\\/{tag}").replace("<!--", "<\\!--")


def _json_blob(obj):
    s = json.dumps(obj, ensure_ascii=False)
    return (s.replace("</", "<\\/").replace("<!--", "<\\u0021--")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def build_html(files):
    """files: list of {"name", "text"} dicts -> the self-contained HTML string."""
    html = (VIEWER / "index.html").read_text(encoding="utf-8")
    css = (VIEWER / "viewer.css").read_text(encoding="utf-8")
    js = (VIEWER / "viewer.js").read_text(encoding="utf-8")
    link, script = '<link rel="stylesheet" href="viewer.css">', '<script src="viewer.js"></script>'
    if link not in html or script not in html:
        raise RuntimeError("tools/agent_watch/index.html no longer has the expected viewer.css/viewer.js tags")
    html = html.replace(link, f"<style>\n{_safe_script(css, 'style')}\n</style>")
    inject = (f'<script>\n{_safe_script(js, "script")}\n</script>\n'
              f'<script type="application/json" id="arc-watch-data">{_json_blob({"files": files})}</script>\n'
              f"<script>\n{SHIM}\n</script>")
    return html.replace(script, inject)


def bundle(paths, out):
    """Write the bundle; returns (exit_code, message). 2 when nothing usable was found."""
    try:
        found = collect(paths)
    except FileNotFoundError as e:
        return 2, f"arc_watch: no such path: {e}"
    if not found:
        return 2, "arc_watch: no trace*.jsonl or bench_results.json found in " + ", ".join(map(str, paths))
    trace = _load_trace_mod()
    files, taken = [], set()
    for f in found:
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"warning: skipping {f}: {e}", file=sys.stderr)
            continue
        if f.suffix == ".jsonl":
            try:
                _n, errs = trace.validate_file(str(f))
            except Exception as e:  # noqa: BLE001 - validator problems must not block bundling
                errs = [f"validator failed: {e}"]
            for m in errs[:5]:
                print(f"warning: {f}: {m}", file=sys.stderr)
            if len(errs) > 5:
                print(f"warning: {f}: ... and {len(errs) - 5} more", file=sys.stderr)
        files.append({"name": _label(f, taken), "text": text})
    if not files:
        return 2, "arc_watch: nothing readable to bundle"
    out = Path(out)
    out.write_text(build_html(files), encoding="utf-8")
    return 0, str(out)


def download_kaggle_output(slug, dest):
    """Fetch a kernel's output with the kaggle CLI into dest (tests monkeypatch this)."""
    subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(dest)], check=True)


def _default_out(label):
    return Path(tempfile.gettempdir()) / f"arc_watch_{label}.html"


def _finish(code, msg, open_browser):
    print(msg, file=sys.stderr if code else sys.stdout)
    if code == 0 and open_browser:
        webbrowser.open(Path(msg).resolve().as_uri())
    return code


def _missing_traces_message(d):
    names = [p.name for p in Path(d).rglob("*") if p.is_file() and p.name in ("submission.json", "bench_results.json")]
    return ("arc_watch: no trace*.jsonl in the output. Found: " + (", ".join(sorted(set(names))) or "nothing relevant")
            + ". Re-run with agent tracing enabled (trace.jsonl / trace_<name>.jsonl).")


def cmd_kaggle(arg, out, open_browser):
    p = Path(arg)
    if p.is_dir():
        d = p
        tmp = None
    else:
        tmp = tempfile.TemporaryDirectory()
        d = Path(tmp.name)
        try:
            download_kaggle_output(arg, d)
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"arc_watch: kaggle download failed: {e}", file=sys.stderr)
            return 2
    try:
        traces = [f for f in collect([d]) if f.suffix == ".jsonl"]
        if not traces:
            print(_missing_traces_message(d), file=sys.stderr)
            return 2
        out = out or _default_out(p.name or "kaggle")
        return _finish(*bundle([d], out), open_browser)
    finally:
        if tmp:
            tmp.cleanup()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle", help="write one self-contained HTML file")
    b.add_argument("paths", nargs="+")
    b.add_argument("-o", "--out", required=True)
    o = sub.add_parser("open", help="bundle, then open in the browser")
    o.add_argument("paths", nargs="+")
    o.add_argument("--out")
    o.add_argument("--no-browser", action="store_true")
    k = sub.add_parser("kaggle-output", help="a downloaded kernel output dir, or a kernel slug to download")
    k.add_argument("target")
    k.add_argument("--out")
    k.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "bundle":
        return _finish(*bundle(a.paths, a.out), False)
    if a.cmd == "open":
        return _finish(*bundle(a.paths, a.out or _default_out("open")), not a.no_browser)
    return cmd_kaggle(a.target, a.out, not a.no_browser)


if __name__ == "__main__":
    sys.exit(main())
