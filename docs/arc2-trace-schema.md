# Agent trace schema (`trace.jsonl`)

One JSON object per line, append-only, flushed after every event, so a crashed run leaves a usable file (a truncated last
line is ignored). Written by the agent loop and solvers, read by the replay viewer (`tools/agent_watch/`). Every event has:

- `ev` (string): the event type below
- `t` (number): seconds since `run_start`, monotonic, 3 decimals
- `cfg` (string): config name, `"default"` outside bench mode

Task events also carry `task` (task id string). Attempt events also carry `att` (integer, 0-based). Grids are lists of
lists of integers 0-9, at most 30 by 30. Text fields are capped as stated; a capped field ends with `"…[cut N]"`.

| `ev` | Required fields | Meaning |
|---|---|---|
| `run_start` | `run` (string id), `model`, `effort`, `mode` (`agent`\|`direct`\|`synth`), `persist` (0/1 or null), `tasks` (list of ids), `started_at` (RFC 3339) | first event of a config's run |
| `task_start` | `task`, `train` (list of `{input, output}`), `test` (list of `{input}`), `answers` (OPTIONAL: list of grids or null, public eval solutions when known, for the viewer's "expected" and scoring; omitted when no solutions file; never given to the model) | task begins |
| `turn` | `task`, `att`, `turn` (1-based), `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `finish` (`stop`\|`length`\|`tool_calls`\|`error`), `dur_s`, `reasoning` (string, cap 4000 chars: head and tail), `content` (string, cap 2000), `tool_calls` (list of `{id, name, args}`; `args` for `python`/`submit` is `{code}`, cap 6000 chars each) | one model reply |
| `tool_result` | `task`, `att`, `turn`, `name` (`python`\|`submit`), `text` (cap 4000), `dur_s` | what the tool returned to the model |
| `submit` | `task`, `att`, `turn`, `program` (string, cap 6000), `reproduced` (int), `n_train` (int), `verified` (bool), `outputs` (list per train example: grid or null), `wrong` (list per failing train example: `{ex, total, cells: [[r,c,got,expected], ...] up to 10}`; `ex` is the 0-based index into `train`, like `outputs`), `flags` (list of strings, plausibility reasons), `test_outputs` (list per test input: grid or null) | one `submit` evaluated on the demos |
| `rescue` | `task`, `att`, `turn`, `reason` | no-think rescue turn issued (optional) |
| `attempt_end` | `task`, `att`, `status` (`verified`\|`deadline`\|`no_budget`\|`error`\|`prose`\|`max_turns`), `verified` (bool), `tokens` (int), `turns`, `python_calls`, `submits`, `wall_s` | attempt finished |
| `task_end` | `task`, `rows` (list per test input: `{attempt_1, attempt_2}` grids), `verified` (int), `solved` (list of bool per test input, or null when no answers) | final answers for the task |
| `heartbeat` | `tok_s` (aggregate), `running`, `waiting`, `kv_pct`, `gpu_mem_mib` (any may be null) | server health, roughly every 30-120 s |
| `run_end` | `summary` (`{tasks, verified, solved_test_inputs, tokens, tok_s, wall_s}`, same fields as the BENCH line) | last event |

Ordering rules: `run_start` first, `run_end` last (absent in a crashed run); a task's events appear in time order, tasks
interleave when run in parallel; a `tool_result` follows the `turn` that requested it; `submit` precedes its `tool_result`.
The validator (`kaggle/arc2_synth/trace.py`) checks required fields and types, ignores unknown extra fields (forward
compatible), rejects unknown `ev` values, and tolerates a truncated final line.
