# ARC Agent Watch

A static viewer for the agent `trace.jsonl` files described in `docs/arc2-trace-schema.md`.
No build step, no network, no libraries: it runs from `file://`.

## Open it

1. Open `tools/agent_watch/index.html` in a browser (double-click it, or `xdg-open tools/agent_watch/index.html`).
2. Click **Open trace.jsonl**, or drag a trace file onto the page. Try `sample/sample_trace.jsonl` first.
3. **Reload** re-reads the same file (use it on a trace that is still being written). If the browser refuses because
   the file changed, pick it again.

The loader reads JSONL line by line, ignores a truncated last line, bad lines and unknown events or fields. In bench
mode (several `cfg` values) a config selector appears in the header.

## What you see

- Header: run and config chip, elapsed (last event `t`), tokens (sum of `completion_tokens` over turns), tok/s (latest
  heartbeat), verified tasks.
- Right: run health from heartbeats, and the task list (verified, unverified, running, queued).
- Left: the selected task's examples and test (Test shows the public answer when the trace has one), and the latest
  submit: expected, program output, wrong cells, `(r,c) got A expected B`, colour pairs and plausibility flags. The
  example shown is the first failing one (`wrong[0].ex`, read as 0-based), else Example 1.
- Centre: turn timeline of THINKING, PYTHON and SUBMIT cards, a THINKING NOW card when the attempt has no reply yet.
- Footer scrubber: one tick per event of the selected task. Everything shows the state as of the chosen event.

## Keyboard

Left and right arrows step one event, Home and End jump, space plays or pauses (when a button is not focused).
Playback waits the event `t` gap divided by the speed (1x, 2x, 4x, 8x, 32x), capped at 1.5 s per step.

## Tests and sample

- `node --test tools/agent_watch/test` (also run by `tests/test_agent_watch_js.py` when `node` is installed).
- `python3 tools/agent_watch/sample/make_sample.py` regenerates the synthetic sample trace.
