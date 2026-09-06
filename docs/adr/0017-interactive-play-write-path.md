# ADR-0017: An interactive "play" write path into the visualizer (F13 Stage 0)

- Status: Accepted
- Date: 2026-09-06
- Deciders: repo owner (delegated implementation, per F13's own Stage 0
  framing), via conversation 2026-09-06

## Context

`docs/QUESTIONS.md` F13 asked whether to build an interactive editor so a
human can solve ARC-AGI-1 tasks by hand, logged as a demonstration
trajectory for warm-start/imitation learning (ADR-0009). The user decided
this staged, with only Stage 0 scoped as a buildable slice: a minimal UI
over the curated action space (`arc_env.actions`) plus a live `ArcEnv`
instance, logging through the *unmodified* `EpisodeWriter` schema (`arc_env/
episode_log.py`) so a human solve is automatically warm-start-compatible.

The catch, named explicitly in F13's own framing: `viz/backend/server.py`'s
module docstring said "read-only, no write path" (ADR-0006/ADR-0007) - both
prior ADRs designed the visualizer as a pure log-and-replay reader,
deliberately avoiding any trainer-to-visualizer or visualizer-to-disk write
coupling. Stage 0 needs a real write path (a human's step-by-step play
session has to end up as a `runs/<run_id>/` directory `server.py`'s existing
read-only endpoints can then browse). This ADR records that reversal
explicitly, and scopes it as narrowly as possible: a new module
(`viz/backend/play.py`) holding all the new logic, `server.py` only gaining
a thin HTTP-dispatch layer into it, with none of the existing read-only
run-browsing/dashboard routes touched or weakened.

## Decision

**Add `viz/backend/play.py`**, an in-memory session/game-logic module, kept
separate from `server.py`'s read-only endpoints:

- `start_session(task_id, pair_index)` creates a fresh `ArcEnv` (`max_steps
  = MAX_STEPS_PLAY = 200` - a human isn't bound by an RL rollout's step
  budget the same way a trainer's episodes are, `arc_env.env.
  DEFAULT_MAX_STEPS` is 25), resets it against the requested task/pair, and
  returns the same-shaped state dict every other call returns:
  `{session_id, task_id, pair_index, grid, target_grid, selected,
  terminated, truncated, valid_action, exact_match}`.
- `step_session(session_id, primitive, args)` resolves `primitive` via
  `arc_env.actions.ACTION_BY_NAME`, pads `args` with 0s up to `actions.
  MAX_ARITY`, drives `env.step()`, buffers the step (grid before/after,
  decoded action name/args, reward, `terminated`/`truncated`/`valid_action`/
  `exact_match`, the post-step selection), and returns the same state dict
  shape plus `reward`.
- `save_session(session_id, runs_dir, run_id=None)` writes the buffered
  steps out via the **unmodified** `arc_env.episode_log.EpisodeWriter`/
  `write_run_meta` - `algo="human"` is the only new value written to
  `run_meta.json`'s existing `algo` field, not a schema change. Defaults
  `run_id` to `f"human-{task_id}-{timestamp}"` when the caller doesn't
  supply one, and returns `{run_id, episode_id}`.
- `list_actions()` returns `arc_env.actions.ACTIONS`, generically described
  (`{name, kind, args: [{name, kind}, ...]}`), for the frontend's action
  picker to build itself from rather than hardcoding a menu that goes stale
  every time `arc_env/actions.py` grows (41 actions as of ADR-0016, already
   3 ADRs past this repo's last hardcoded menu size).

**Session model**: an in-memory `dict[session_id, PlaySession]` at module
scope, `session_id = uuid.uuid4().hex` (not guessable/sequential), guarded
by one `threading.Lock()` - `server.py` runs on `ThreadingHTTPServer`
already (see its own `run_server`), so concurrent requests against the same
or different sessions could otherwise race on the shared dict or on a
session's own mutable state (its `ArcEnv`, buffered step list, running
totals). **No persistence until an explicit `save_session` call** - this is
a local, single-user tool (Q1/Q9), so an in-flight, unsaved session being
lost across a backend process restart is an accepted simplicity/durability
tradeoff, not an oversight: adding real persistence (e.g. pickling sessions
to disk) would be meaningfully more machinery for a problem this project's
actual usage pattern (one person, one sitting, save when done) doesn't have.

**New HTTP routes**, dispatched from `server.py`'s `Handler` class (stdlib
`http.server.BaseHTTPRequestHandler` doesn't support mounting a second
handler, so the dispatch itself has to live in `server.py`, but every route
body is a one-line call into `play.py`):

- `GET  /api/actions` → `play.list_actions()`'s payload.
- `POST /api/play/start` → body `{task_id, pair_index}` → `play.
  start_session(...)`. 400 on an invalid/unknown `task_id` or out-of-range
  `pair_index`.
- `POST /api/play/<session_id>/step` → body `{primitive, args}` → `play.
  step_session(...)`. 404 for an unknown `session_id`, 400 for an unknown
  `primitive` name or non-list `args`.
- `POST /api/play/<session_id>/save` → body `{run_id?}` → `play.
  save_session(...)`. 404 for an unknown `session_id`, 400 for an invalid
  (path-traversal-shaped) `run_id`.

`server.py`'s new `do_POST` reads and JSON-decodes the request body, matches
the path, and translates a raised `play.PlayError(status, message)` into the
matching HTTP error - the same `_json`/`_error` helpers the read-only `do_GET`
routes already use, so error responses look identical either way.

**Path-traversal validation.** `task_id` is a new network-facing input that
flows into `arc_env.task_loader.load_task`, which does `open
(TRAINING_DATA_DIR / f"{task_id}.json")` - an unvalidated value (e.g.
`"../../../etc/passwd"`) is a path-traversal vulnerability. `play.py` defines
its own `_SAFE_ID_RE`/`_safe_id`, mirroring `server.py`'s existing
`_RUN_ID_RE`/`_safe_id` allowlist pattern exactly (`^[A-Za-z0-9_.-]+$`,
rejecting `.`/`..`), and checks `task_id` against it before any file access,
returning a 400 on failure rather than letting `load_task` attempt the
`open()` at all. The same check is applied to a client-supplied `run_id` on
save, for the same reason: `run_id` also becomes a path component
(`runs_dir / run_id`) that this module itself writes to, and F13's brief
only called out `task_id` explicitly - `run_id` is the same category of risk
and was validated the same way rather than left as a gap. A test
(`tests/test_viz_play.py::test_start_rejects_path_traversal_task_id`,
`::test_http_start_rejects_path_traversal_task_id`,
`::test_save_rejects_path_traversal_run_id`) proves both.

**Any training task_id is valid, not just curated ones.** `ArcEnv.reset`/
`load_task` already work for any of the 400 `third_party/ARC-AGI/data/
training/*.json` tasks, not only `arc_env.task_loader.CURATED_TASK_IDS` (38
as of ADR-0016). F13's own Stage 1 plan is specifically to run Stage 0 on
the *excluded* 269 tasks next, to observe which mechanism gaps a human
naturally reaches for - restricting Stage 0 to the curated subset would
defeat that plan before it starts, so `task_id` is validated only for
path-traversal shape and actual file existence, never against
`CURATED_TASK_IDS`.

## Mechanism

`play.py` never touches `arc_env`/`trainers` internals - it only calls
`ArcEnv`'s existing public surface (`__init__(max_steps=...)`, `reset(...)`,
`step(...)`, `get_grid()`, `get_selected()`) and `arc_env.actions`'s existing
`ACTIONS`/`ACTION_BY_NAME`/`MAX_ARITY`, and only calls `arc_env.episode_log`'s
existing `EpisodeWriter`/`write_run_meta`/`RunMeta`/`grid_to_list` - zero
changes to any of those modules. A human step and a trained-policy step are
logged through the exact same `EpisodeWriter.step(...)` call shape
(`grid_before`, `action_name`, `action_args`, `grid_after`, `reward`,
`terminated`, `truncated`, `valid_action`, `exact_match`, `selected`) - the
only new value anywhere in the schema is `run_meta.json`'s `algo` field
seeing `"human"` for the first time, alongside the existing `"ppo"`/`"gp"`.
This is what makes a saved human-play run automatically warm-start-
compatible with `trainers/ppo/warm_start.py`'s `--warm_start_from
<gp_run_dir>` (ADR-0009) with zero code changes needed there either - that
loader only cares about `episodes/<id>.jsonl`'s shape, not which `algo`
produced it. (Whether `warm_start.py`'s hardcoded `best-program` episode-ID
lookup should also learn to accept a human episode ID is a Stage 1/2
question, out of scope here - Stage 0 only needs the schema, not the
warm-start loader, to already work.)

Checked whether `algo="human"` breaks anything reading `run_meta.json`'s
`algo` field: the frontend's run picker (`viz/frontend/src/picker.ts`)
only ever displays it as a label (`` `${run.run_id} (${run.algo})` ``) and
groups runs by `created_at` date (`runs.ts`'s `groupRunsByDate`), never
switching behavior on the literal value - so a `"human"` run renders and
groups identically to a `"ppo"`/`"gp"` one, with no code change needed.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Open `EpisodeWriter` incrementally at session-start and write each step as it happens, rather than buffering in memory until `save_session` | Rejected for Stage 0: an open file handle held across a long-lived, possibly-abandoned session (a human closes the tab mid-task) is more state to manage correctly (crash safety, cleanup) for no real benefit at this scale (a human episode is at most `MAX_STEPS_PLAY=200` steps, trivial to hold in memory) - buffering is simpler to get right and matches this ADR's "no persistence until an explicit save" session model anyway. |
| A dedicated write-capable server process/port, separate from the read-only visualizer backend | Rejected: `make viz` already starts one process; splitting into two would be real new operational surface (two ports, two lifecycles) for a problem that a lock and a narrowly-scoped new module already solve without it. |
| Restrict Stage 0 to `CURATED_TASK_IDS` only | Rejected per F13's Stage 1 plan - the whole point of collecting human-play data is to see which of the *excluded* 269 tasks reveal missing mechanisms; restricting to the 38 already-curated tasks would answer nothing new. |
| Validate `task_id` against `CURATED_TASK_IDS` and treat everything else as a 400 | Same rejection as above - would silently defeat Stage 1 before it starts. `task_id` validation is deliberately just "well-formed path component that resolves to a real training task file", nothing narrower. |

## Consequences

- `viz/backend/server.py`'s module docstring is corrected: it no longer
  claims "read-only, no write path" unconditionally. The run-browsing/
  dashboard routes are still exactly as read-only as ADR-0006/0007 designed
  them - this ADR adds a second, narrow, explicitly-named write path
  alongside them, not a weakening of the first.
- `viz/backend/play.py` is new, ~250 lines, with its own test file
  (`tests/test_viz_play.py`) covering: a full start→step→save flow for a
  real curated task (`67a3c6ac`, solved by a single `vmirror`) with the
  saved run confirmed readable back through the existing read-only
  `list_runs`/`list_episode_ids`/`read_episode` functions and `algo ==
  "human"`; an unknown-session 404; an unknown-primitive-name 400; and the
  path-traversal-rejection case for both `task_id` and `run_id`.
- The frontend gains a new, independent "Play" section
  (`viz/frontend/src/play.ts`, wired from `main.ts` the same way `dashboard`/
  `picker`/`player` already are): a free-text `task_id` input (any of the
  400 training tasks, not just curated ones), an action picker built
  generically from `GET /api/actions` (so it doesn't need updating the next
  time `arc_env/actions.py` grows), a step button, and a save-as-run button.
  Reuses `grid.ts`'s `drawGrid`/`cellSizeFor` for rendering, the same way
  `PlayerPanel` already does for replay.
- Sessions are entirely in-memory and process-local - restarting the
  backend (e.g. `make viz` again) loses any unsaved play session. Acceptable
  for a local, single-user tool; would need real persistence (out of scope)
  if this were ever multi-user or long-running.
- This is **Stage 0 only**. Stage 1 (running this UI on the excluded-269
  tasks from F11's coverage audit, to see what tools a human naturally
  reaches for) and Stage 2 (turning Stage 1's findings into new ADR'd
  mechanisms) are both still not started - `docs/QUESTIONS.md`'s F13 entry
  is updated to point at this ADR for Stage 0 and reconfirm Stages 1/2 are
  still contingent, unstarted future work.
