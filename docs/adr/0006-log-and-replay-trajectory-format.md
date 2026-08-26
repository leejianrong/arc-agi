# ADR-0006: Log-and-replay via local JSONL/CSV, not live streaming from the trainer

- Status: Accepted
- Date: 2026-08-27
- Deciders: repo owner

## Context

The visualizer needs to show both training-run metrics and step-by-step
episode replay ("watch it play like a game"). The trainer and visualizer
could be coupled live (a socket/IPC channel streaming state as training
happens) or decoupled via files the trainer writes and the visualizer reads.

## Decision

The trainer writes to `runs/<run_id>/` continuously during training:
`run_meta.json` (config, algorithm, `schema_version`), `metrics.jsonl` (one
row per episode/update: reward, success rate, loss), and
`episodes/<episode_id>.jsonl` (one line per step: grid before/after, action,
args, reward, done) for periodically-saved evaluation rollouts. The
visualizer reads these files directly — tailing/polling `metrics.jsonl` for
the dashboard, loading a full `episodes/*.jsonl` file for replay — with no
socket or IPC channel between trainer and visualizer process.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Live streaming (websocket/IPC from the trainer to the visualizer) | Requires the trainer to run a server or push to one, coupling training-process lifecycle to visualizer availability, for a "live" feel that log-tailing already delivers at second-level latency — not worth the added architecture for this milestone. |
| Database (SQLite/Postgres) instead of flat files | No concurrent-writer problem exists to justify it (`docs/QUESTIONS.md` Q5), and flat JSONL/CSV keeps runs directly inspectable/diffable by the user without a client, which a database would not. |

## Consequences

- The visualizer is a read-only consumer of `runs/` and can be started,
  stopped, or crash independently of any training process — no lifecycle
  coupling.
- "Live" means "as fresh as the last periodic write," not instant — acceptable
  per the user's confirmed choice (`docs/QUESTIONS.md` F5), but means the
  trainer must be designed to flush metrics/checkpoints periodically (e.g.
  every N episodes) rather than only at the end of a run.
- This log format is also what makes ADR-0003's genetic-programming ↔ RL
  interoperation possible later — both trainers write the same shape of file.
