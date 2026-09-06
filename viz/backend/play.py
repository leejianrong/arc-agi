"""Live human-play session logic (F13 Stage 0, ADR-0017).

This is the one real write path into an otherwise read-only visualizer
backend - `viz/backend/server.py`'s own module docstring used to claim
"read-only, no write path" unconditionally; ADR-0017 records this as a
deliberate, narrow reversal of that design, not a rewrite of it. Kept in its
own module so `server.py`'s existing run-browsing/dashboard endpoints stay
untouched: `server.py` only parses the HTTP request and dispatches into the
functions below.

A human plays one curated action (`arc_env.actions`) at a time against a
live `ArcEnv` instance, same as a trained policy would - `start_session`
creates the env, `step_session` drives it, and `save_session` writes the
accumulated steps out through the *unmodified* `arc_env.episode_log.
EpisodeWriter`/`write_run_meta`, with `algo="human"` as the only new value
(no schema change), so a human solve is automatically warm-start-compatible
(ADR-0009).

Session model: an in-memory `dict[session_id, PlaySession]`, guarded by one
`threading.Lock()` since `server.py` runs on `ThreadingHTTPServer` and
concurrent requests could otherwise race on the same session (or on the
session dict itself). No persistence until an explicit `save_session` call -
this is a local, single-user tool, so losing an in-flight (unsaved) session
across a process restart is an accepted simplicity/durability tradeoff, not
an oversight (see ADR-0017).

Security: `task_id` and (optionally client-supplied) `run_id` are new
network-facing inputs. `task_id` flows into `arc_env.task_loader.load_task`,
which does `open(TRAINING_DATA_DIR / f"{task_id}.json")`, and `run_id` flows
into a `runs_dir / run_id` path this module itself writes to - both are path
components, so both are checked against `_SAFE_ID_RE` (mirroring `server.
py`'s own `_RUN_ID_RE`/`_safe_id` allowlist) before touching the filesystem.
Any training task_id (not just the curated subset) is accepted - `ArcEnv`/
`load_task` already work for any of the 400 `third_party/ARC-AGI/data/
training/*.json` tasks, and F13's own Stage 1 plan is to run this on the
excluded 269 later.
"""

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from arc_env import actions
from arc_env.env import ArcEnv
from arc_env.episode_log import EpisodeWriter, RunMeta, grid_to_list, write_run_meta
from arc_env.task_loader import load_task

# A human isn't bound by an RL rollout budget the same way a trainer's
# episodes are (`arc_env.env.DEFAULT_MAX_STEPS` is 25) - generous, named
# constant rather than a magic number sprinkled through this module.
MAX_STEPS_PLAY = 200

# Mirrors `viz/backend/server.py`'s `_RUN_ID_RE`/`_safe_id` - both `task_id`
# and (optionally client-supplied) `run_id` are path components here too.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _safe_id(value: str) -> bool:
    return bool(_SAFE_ID_RE.match(value)) and value not in (".", "..")


class PlayError(Exception):
    """A client error that should map to a specific HTTP status code."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class _StepRecord:
    """One buffered step, holding everything `EpisodeWriter.step` needs -
    buffered in memory and only written out at `save_session` time, rather
    than opening the writer incrementally at session-start (simpler: no file
    handle to keep open/guard across a long-lived, possibly-abandoned
    session)."""

    step: int
    grid_before: tuple
    action_name: str | None
    action_args: dict
    grid_after: tuple
    reward: float
    terminated: bool
    truncated: bool
    valid_action: bool
    exact_match: bool
    selected: list | None


@dataclass
class PlaySession:
    env: ArcEnv
    task_id: str
    pair_index: int
    input_grid: tuple
    target_grid: tuple
    steps: list = field(default_factory=list)  # list[_StepRecord]
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    exact_match: bool = False


_sessions: dict[str, PlaySession] = {}
_lock = threading.Lock()


def list_actions() -> list:
    """`GET /api/actions`'s payload: every curated action, generically
    derived from `arc_env.actions.ACTIONS` so the frontend's action picker
    never goes stale as that list grows."""

    return [
        {
            "name": a.name,
            "kind": a.kind,
            "args": [{"name": spec.name, "kind": spec.kind} for spec in a.args],
        }
        for a in actions.ACTIONS
    ]


def start_session(task_id, pair_index=0) -> dict:
    if not isinstance(task_id, str) or not _safe_id(task_id):
        raise PlayError(400, "invalid task_id")
    try:
        pair_index = int(pair_index)
    except (TypeError, ValueError):
        raise PlayError(400, "invalid pair_index") from None

    try:
        task = load_task(task_id)
    except FileNotFoundError:
        raise PlayError(400, f"unknown task_id: {task_id}") from None

    if not (0 <= pair_index < len(task.train)):
        raise PlayError(400, f"invalid pair_index: {pair_index}") from None

    env = ArcEnv(max_steps=MAX_STEPS_PLAY)
    env.reset(task_id=task_id, pair_index=pair_index, task=task)
    pair = task.train[pair_index]

    session_id = uuid.uuid4().hex
    session = PlaySession(
        env=env,
        task_id=task_id,
        pair_index=pair_index,
        input_grid=pair.input,
        target_grid=pair.output,
    )
    with _lock:
        _sessions[session_id] = session

    return {
        "session_id": session_id,
        "task_id": task_id,
        "pair_index": pair_index,
        "grid": grid_to_list(session.input_grid),
        "target_grid": grid_to_list(session.target_grid),
        "selected": env.get_selected(),
        "terminated": False,
        "truncated": False,
        "valid_action": True,
        "exact_match": session.input_grid == session.target_grid,
    }


def _get_session(session_id: str) -> PlaySession:
    with _lock:
        session = _sessions.get(session_id)
    if session is None:
        raise PlayError(404, "unknown session_id")
    return session


def step_session(session_id: str, primitive, raw_args) -> dict:
    session = _get_session(session_id)

    if not isinstance(primitive, str) or primitive not in actions.ACTION_BY_NAME:
        raise PlayError(400, f"unknown primitive: {primitive!r}")
    if not isinstance(raw_args, list):
        raise PlayError(400, "args must be a list of ints")

    primitive_index = actions.ACTION_BY_NAME[primitive]
    padded = list(raw_args) + [0] * actions.MAX_ARITY
    try:
        action = {"primitive": primitive_index}
        for i in range(actions.MAX_ARITY):
            action[f"arg{i + 1}"] = int(padded[i])
    except (TypeError, ValueError):
        raise PlayError(400, "args must be ints") from None

    with _lock:
        grid_before = session.env.get_grid()
        _, reward, terminated, truncated, info = session.env.step(action)
        grid_after = session.env.get_grid()
        selected = session.env.get_selected()

        session.steps.append(
            _StepRecord(
                step=len(session.steps),
                grid_before=grid_before,
                action_name=info["action_name"],
                action_args=info["action_args"],
                grid_after=grid_after,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                valid_action=info["valid_action"],
                exact_match=info["exact_match"],
                selected=selected,
            )
        )
        session.total_reward += reward
        session.terminated = terminated
        session.truncated = truncated
        session.exact_match = info["exact_match"]

    return {
        "session_id": session_id,
        "task_id": session.task_id,
        "pair_index": session.pair_index,
        "grid": grid_to_list(grid_after),
        "target_grid": grid_to_list(session.target_grid),
        "selected": selected,
        "terminated": terminated,
        "truncated": truncated,
        "valid_action": info["valid_action"],
        "exact_match": info["exact_match"],
        "reward": reward,
    }


def save_session(session_id: str, runs_dir: Path, run_id=None) -> dict:
    session = _get_session(session_id)

    if run_id is not None:
        if not isinstance(run_id, str) or not _safe_id(run_id):
            raise PlayError(400, "invalid run_id")
    else:
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_id = f"human-{session.task_id}-{timestamp}"

    episode_id = f"{session.task_id}-p{session.pair_index}"
    run_dir = runs_dir / run_id

    with _lock:
        write_run_meta(
            run_dir,
            RunMeta(
                run_id=run_id,
                algo="human",
                task_ids=[session.task_id],
                config={"pair_index": session.pair_index, "max_steps": MAX_STEPS_PLAY},
            ),
        )
        with EpisodeWriter(run_dir, episode_id) as writer:
            writer.start(
                session.task_id, session.pair_index, session.input_grid, session.target_grid, MAX_STEPS_PLAY
            )
            for rec in session.steps:
                writer.step(
                    rec.step,
                    rec.grid_before,
                    rec.action_name,
                    rec.action_args,
                    rec.grid_after,
                    rec.reward,
                    rec.terminated,
                    rec.truncated,
                    rec.valid_action,
                    exact_match=rec.exact_match,
                    selected=rec.selected,
                )
            writer.end(len(session.steps), session.exact_match, session.total_reward)

    return {"run_id": run_id, "episode_id": episode_id}
