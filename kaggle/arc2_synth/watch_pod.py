# MIT No Attribution (MIT-0).
"""Driver-side watcher for one RunPod job: prints only what changes or looks wrong.

    python watch_pod.py ENV_FILE LABEL [--interval 60] [--expect-min 120]

ENV_FILE is a shell file that exports RUNPOD_API_KEY, RP_STATE_FILE and RP_POD_NAME (the `rp` wrapper's
config). Each output line is one event, so it can feed a Monitor/notification stream:
  ALERT   a failure signature in the log (traceback, VLLM_DIED, SMOKE_FAILED, 500s, OOM, STALL?, ...)
  FEED    the log feed stopped advancing (pod may be wedged or logs lagging)
  GONE    the pod no longer exists (finished, reaped or crashed) - collect the artifact now
  PHASE / DONE / task progress / heartbeat digests / cost every few minutes.
"""
import json
import re
import subprocess
import sys
import time

ALERTS = re.compile(
    r"Traceback|VLLM_DIED|SMOKE_FAILED|STALL\?|HTTPError|Internal Server Error|out of memory|OutOfMemory|"
    r"Killed|FATAL|Connection refused|CLONE_FAILED|send-failed|No space left|illegal memory", re.IGNORECASE)
MARKS = re.compile(r"^(PHASE=|TRYING|CHOSEN_CONFIG|SMOKE|SUMMARY|RP_ARTIFACT_CODE|gpu memory used)")
PROGRESS = re.compile(r"^\[\d+/\d+\]")


def sh(cmd, env_file):
    r = subprocess.run(["bash", "-c", f". {env_file}; {cmd}"], capture_output=True, text=True, timeout=90, check=False)
    return r.stdout


def main():
    env_file, label = sys.argv[1], sys.argv[2]
    interval = int(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 60
    expect_min = int(sys.argv[sys.argv.index("--expect-min") + 1]) if "--expect-min" in sys.argv else 120
    seen, last_new, last_cost, t0, missing = set(), time.time(), 0.0, time.time(), 0
    last_digest = 0.0  # HB and throughput lines are throttled to one per 5 min so they stay readable
    done = 0

    def emit(kind, msg):
        print(f"{time.strftime('%H:%M:%S')} [{label}] {kind} {msg}", flush=True)

    emit("START", f"watching (interval {interval}s, expect <= {expect_min} min)")
    while True:
        try:
            raw = sh('rp logs 2>&1 | grep -v "$RUNPOD_API_KEY"', env_file)
            pods = json.loads(sh("runpodctl pod list", env_file) or "[]")
            name = sh('echo "$RP_POD_NAME"', env_file).strip()
        except Exception as e:  # noqa: BLE001 - the watcher must outlive transient CLI failures
            emit("WARN", f"poll failed: {e!r}")
            time.sleep(interval)
            continue
        mine = [p for p in pods if p.get("name") == name]
        new = 0
        for line in raw.splitlines():
            try:
                text = json.loads(line)["line"]
            except (ValueError, KeyError):  # a non-JSON line (e.g. an rp notice): skip it
                text = None
            if text is None:
                continue
            key = hash(line)
            if key in seen:
                continue
            seen.add(key)
            new += 1
            if ALERTS.search(text) and "POST /v1" not in text:
                emit("ALERT", text[:230])
            elif PROGRESS.match(text):
                done += 1
                emit("TASK", text[:200])
            elif MARKS.match(text):
                emit("PHASE", text[:200])
            elif (text.startswith("HB ") or "generation throughput" in text) and time.time() - last_digest > 300:
                last_digest = time.time()
                if text.startswith("HB "):
                    emit("HB", text[3:200])
                else:
                    emit("VLLM", text[text.find("Avg prompt"):][:200])
        if new:
            last_new = time.time()
        elif time.time() - last_new > 300:
            emit("FEED", f"no new log lines for {int(time.time() - last_new)}s (pod {'up' if mine else 'GONE'})")
            last_new = time.time()
        if not mine:
            missing += 1
            if missing >= 2:
                emit("GONE", f"pod '{name}' not in pod list; {done} tasks reported; fetch artifact if RP_ARTIFACT_CODE seen")
                return
        else:
            missing = 0
            up = mine[0].get("uptimeSeconds") or 0
            if time.time() - last_cost > 600:
                last_cost = time.time()
                emit("COST", f"~${mine[0].get('costPerHr', 0) * (time.time() - t0) / 3600:.2f} since watch start, "
                             f"${mine[0].get('costPerHr')}/h, uptime {up // 60} min")
            if time.time() - t0 > expect_min * 60:
                emit("ALERT", f"pod still running after {expect_min} min (expected to finish sooner)")
                expect_min += 30
        time.sleep(interval)


if __name__ == "__main__":
    main()
