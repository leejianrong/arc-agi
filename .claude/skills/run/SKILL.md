---
name: run
description: Launch this project's local visualizer (frontend build + backend server) and, when a real screenshot is needed, drive it with a headless browser to capture one - e.g. confirming the amber selection-overlay renders correctly after an arc_env/viz change. Use when asked to run/start/demo the app, to verify a UI change is actually working (not just passing tests), or when the task mentions a screenshot, headless browser, Playwright, chromium-cli, or checking the selection overlay. This project-level skill covers the launch step CLAUDE.md documents (`make viz`) plus the headless-browser step neither CLAUDE.md nor the global `run` skill's generic fallback covers - use it instead of guessing at either.
---

# Running and verifying the ARC-AGI visualizer

Fragile, exact-command sequence (see "Why these exact steps" below for the
friction each step avoids) - follow it in order, don't improvise around it.

## Checklist

- [ ] 1. `cd viz/frontend && npm ci` if `viz/frontend/node_modules` doesn't exist yet
- [ ] 2. `make viz` (or `make viz PORT=8001` if 8000 is taken) from repo root; wait for
      the `Serving on http://127.0.0.1:<port>` line
- [ ] 3. Make sure `runs/` has at least one run to look at (existing data, or generate one
      per "Getting something to look at" below)
- [ ] 4. `command -v chromium-cli` - if found, use it directly; if not (the common case),
      set up Playwright per "Headless browser" below
- [ ] 5. Capture the screenshot(s) you need, verify the image actually shows what you expect
- [ ] 6. Clean up: kill the backend process, delete any `runs/` dir and screenshots you
      created for this check - none of that is a deliverable

## 1-2. Build + start

```
cd viz/frontend && npm ci   # only if node_modules is missing - fast (~1-2s) if npm's
                             # cache already has the packages
cd <repo root>
make viz                    # builds the frontend, then starts the backend on :8000
```

`make viz` alone, with no `node_modules`, fails loudly and immediately:

```
cd viz/frontend && npm run build
> tsc -b && vite build
sh: 1: tsc: not found
make: *** [Makefile:41: build-frontend] Error 127
```

Run `npm ci` first. The very first `uv run` in a fresh checkout/worktree also
prints `Creating virtual environment...` / `Installed N packages` before the
server starts - that's `uv` syncing the Python venv, not an error; it only
happens once.

Port conflicts fail loudly too, they do **not** silently rebind to a free
port:

```
OSError: [Errno 98] Address already in use
```

If 8000 is already listening, use `make viz PORT=8001` (or any other free
port) - don't assume the default just works on every machine/session.

## 3. Getting something to look at

The dashboard auto-selects the most recently created run and its first
episode on load - as long as `runs/` has at least one run, no manual
selection is needed in the browser.

If `runs/` is empty, the fastest way to get a real episode showing the
selection overlay is `scripts/build_verify_episode.py` in this skill's own
`scripts/` dir (run it, don't just read it) - it hand-scripts a curated
task's known-correct solver program instead of relying on a random policy to
stumble into a selection action:

```
uv run python .claude/skills/run/scripts/build_verify_episode.py
# wrote <repo>/runs/verify
```

Default task `1f85a75f` runs `[select_largest, commit_selection]` - the
first action leaves a 12-cell object selected (amber outline visible), the
second crops to it (episode ends, SOLVED). Pass `--task_id <id>` for a
different curated task+program (see `arc_env/task_loader.py`'s
`CURATED_TASK_IDS` dict for what's available), or `--run_id`/`--episode_id`
to avoid colliding with an existing `runs/verify/`.

`make rollout` / `make train` also produce real runs, but a random policy
rarely lands on `select_*` + `commit_selection` back to back, so they're not
a reliable way to see the selection overlay specifically.

## 4. Headless browser

Check `chromium-cli` first - it is commonly *not* installed in this kind of
sandbox (confirmed absent, `command not found`, in more than one session
running this check):

```
command -v chromium-cli   # exit 1 if absent
```

If absent, fall back to Playwright. It is **not** a dependency of this
project (no root `package.json`; `viz/frontend/package.json` has never
included it) - don't add it there. Install it in a scratch directory outside
the repo instead, since it's a verification tool, not a shipped dependency:

```
mkdir -p /tmp/pw-verify && cd /tmp/pw-verify
npm init -y
npm install -D playwright
npx playwright install chromium
```

On a machine that's already run Playwright for any other project, the
browser binary is typically already cached at
`~/.cache/ms-playwright/chromium-<build>`, so `npx playwright install
chromium` is near-instant (no download) - it only downloads if that build
isn't cached yet. Don't skip running it though: the exact chromium build
your installed `playwright` version needs can differ from what happens to
be cached.

## 5. Capture the screenshot

Copy this skill's `scripts/screenshot.js` into your scratch Playwright
directory (it needs Playwright resolvable from its CWD) and run it:

```
cp <repo>/.claude/skills/run/scripts/screenshot.js /tmp/pw-verify/
cd /tmp/pw-verify
node screenshot.js http://127.0.0.1:8000/ /tmp/pw-verify/dashboard.png
```

To check the selection overlay specifically, step Panel A forward first -
step 1 selects (amber outline), step 2 commits (crops, SOLVED):

```
node screenshot.js http://127.0.0.1:8000/ /tmp/pw-verify/step1.png --forward 1
node screenshot.js http://127.0.0.1:8000/ /tmp/pw-verify/step2.png --forward 2
```

`--forward N` clicks Panel A's "step ▶" button (`.btn-forward` class) N
times from the initial state, waiting 200ms after each click for the canvas
to redraw. Use `--forward`, not the "▶ play" button - play auto-advances
continuously, which makes it hard to time a screenshot at an exact step.

Then actually look at the resulting PNG (e.g. via the Read tool) and confirm
it shows what you expected - don't report success from the command exiting
0 alone.

## 6. Clean up

```
kill <backend pid>                 # or pkill -f viz.backend.server
rm -rf <repo>/runs/verify          # or whatever run_id you used
rm -rf /tmp/pw-verify              # optional - safe to leave for next time,
                                    # since step 4's install becomes a no-op
```

`runs/` is already gitignored, but a verification run left behind still
clutters local `runs/` listings and `make prune-runs` runs - clean it up
rather than leaving it for the next session to wonder about.

## Why these exact steps

This sequence was re-derived hands-on (not copied from an old ticket
comment) to confirm every claim above: `make viz` really does fail with
`tsc: not found` on a clean `node_modules`; a second server on the same port
really does throw `OSError: Address already in use` rather than silently
picking another port; `chromium-cli` really is absent in this kind of
sandbox; Playwright really does find its Chromium build already cached
here and skips the download; and `.btn-forward` really is the right
selector to step Panel A's replay forward one action at a time. Each of
those was wrong-until-checked, which is why they're written as exact
commands/outputs here instead of paraphrased.
