"""scripts/arc_watch.py: bundling traces into one self-contained viewer HTML."""
import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("arc_watch", ROOT / "scripts/arc_watch.py")
aw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aw)

SAMPLES = ROOT / "tools/agent_watch/sample"
A, B = SAMPLES / "sample_trace.jsonl", SAMPLES / "sample_trace_b.jsonl"


def _data(html):
    m = re.search(r'<script type="application/json" id="arc-watch-data">(.*?)</script>', html, re.DOTALL)
    return json.loads(m.group(1))


def test_bundle_samples(tmp_path):
    out = tmp_path / "o.html"
    assert aw.main(["bundle", str(A), str(B), "-o", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "<script src=" not in html and 'href="viewer.css"' not in html
    assert (ROOT / "tools/agent_watch/viewer.css").read_text().strip()[:40] in html
    assert "ARC Agent Watch" in html and "getElementById('file')" in html
    files = _data(html)["files"]
    assert [f["name"] for f in files] == ["sample_trace.jsonl", "sample_trace_b.jsonl"]
    assert files[0]["text"] == A.read_text(encoding="utf-8")


def test_hostile_text_cannot_break_out(tmp_path):
    ev = {"ev": "run_start", "t": 0, "note": "</script><b>x</b> <!-- \u2028 \u2029"}
    t = tmp_path / "trace.jsonl"
    t.write_text(json.dumps(ev) + "\n", encoding="utf-8")
    out = tmp_path / "o.html"
    assert aw.main(["bundle", str(t), "-o", str(out)]) == 0  # validation errors only warn
    html = out.read_text(encoding="utf-8")
    blob = html.split('id="arc-watch-data">', 1)[1].split("</script>", 1)[0]
    assert "\u2028" not in blob and "\u2029" not in blob and "</" not in blob and "<!--" not in blob
    assert json.loads(blob)["files"][0]["text"] == t.read_text(encoding="utf-8")


def test_refuses_empty_dir(tmp_path, capsys):
    assert aw.main(["bundle", str(tmp_path), "-o", str(tmp_path / "o.html")]) == 2
    assert "no trace" in capsys.readouterr().err
    assert not (tmp_path / "o.html").exists()
    assert aw.main(["bundle", str(tmp_path / "missing"), "-o", str(tmp_path / "o.html")]) == 2


def test_directory_collection_one_level(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out/deep").mkdir()
    (tmp_path / "trace_a.jsonl").write_bytes(A.read_bytes())
    (tmp_path / "out/trace_b.jsonl").write_bytes(B.read_bytes())
    (tmp_path / "out/bench_results.json").write_text("{}")
    (tmp_path / "out/submission.json").write_text("{}")
    (tmp_path / "out/deep/trace_c.jsonl").write_bytes(B.read_bytes())  # two levels down: ignored
    got = sorted(f.name for f in aw.collect([tmp_path]))
    assert got == ["bench_results.json", "trace_a.jsonl", "trace_b.jsonl"]


def test_open_no_browser(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(aw.webbrowser, "open", lambda *_: (_ for _ in ()).throw(AssertionError("opened")))
    out = tmp_path / "o.html"
    assert aw.main(["open", str(A), "--out", str(out), "--no-browser"]) == 0
    assert capsys.readouterr().out.strip() == str(out)


def test_open_calls_browser(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(aw.webbrowser, "open", seen.append)
    assert aw.main(["open", str(A), "--out", str(tmp_path / "o.html")]) == 0
    assert seen and seen[0].startswith("file://")


def test_kaggle_output_saved_dir(tmp_path):
    d = tmp_path / "k"
    d.mkdir()
    (d / "trace.jsonl").write_bytes(A.read_bytes())
    out = tmp_path / "o.html"
    assert aw.main(["kaggle-output", str(d), "--out", str(out), "--no-browser"]) == 0
    assert _data(out.read_text(encoding="utf-8"))["files"][0]["name"] == "trace.jsonl"


def test_kaggle_output_without_traces(tmp_path, capsys):
    d = tmp_path / "k"
    d.mkdir()
    (d / "submission.json").write_text("{}")
    (d / "bench_results.json").write_text("{}")
    assert aw.main(["kaggle-output", str(d), "--out", str(tmp_path / "o.html"), "--no-browser"]) == 2
    err = capsys.readouterr().err
    assert "submission.json" in err and "bench_results.json" in err


def test_kaggle_output_downloader_patched(tmp_path, monkeypatch):
    calls = []

    def fake(slug, dest):
        calls.append(slug)
        (Path(dest) / "trace_x.jsonl").write_bytes(B.read_bytes())

    monkeypatch.setattr(aw, "download_kaggle_output", fake)
    monkeypatch.setattr(aw.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real cli")))
    out = tmp_path / "o.html"
    assert aw.main(["kaggle-output", "me/slug", "--out", str(out), "--no-browser"]) == 0
    assert calls == ["me/slug"] and _data(out.read_text(encoding="utf-8"))["files"][0]["name"] == "trace_x.jsonl"


def test_make_target(tmp_path):
    r = subprocess.run(["make", "-n", "arc-watch", "ARGS=open some/path"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "scripts/arc_watch.py open some/path" in r.stdout
    h = subprocess.run(["make", "help"], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    assert "arc-watch" in h and 'make arc-watch ARGS=' in h
