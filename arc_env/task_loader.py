"""Loads the curated-action-space task subset.

`CURATED_TASK_IDS` was derived, not hand-picked: a task qualifies iff
`third_party/arc-dsl/solvers.py`'s known-correct solver for that task calls
only primitives in `arc_env.actions`'s action groups (`ZERO_ARG`/`ONE_ARG`/
`TWO_ARG`/`THREE_ARG`/`FOUR_ARG`/`SELECT`/`ACT_ON_SELECTION`). See
`arc_env/actions.py`'s module docstring for the full reasoning.

V1 additionally required same-shape input/output pairs, as the smallest
possible first slice - not because variable-shape outputs needed a
different action space (`trim`/`tophalf`/`upscale`/`downscale`/... already
change shape). V3 lifts that restriction and adds the 5 tasks that need
`canvas`/`commit` (ADR-0002) or a variable-shape chain of the existing
actions: 11 same-shape (V1) + 5 variable-shape (V3) = 16 total. ADR-0010
Phase 1 (2026-08-29) adds 8 more tasks reachable via the new
self-concatenation actions (`hconcat_self`/etc., see `arc_env/actions.py`)
or an already-expressible-but-previously-untranscribed `commit` call: 7 of
those 8 are variable-shape, but `f25ffba3` nets back to the same shape
(`bottomhalf` then `vconcat_self_hmirror_top` undoes its own height change),
so it's a same-shape task despite going through an intermediate shape
change - 12 same-shape + 12 variable-shape = 24 total. ADR-0011 (ADR-0010
Phase 2 Slice 1, 2026-08-31) adds 2 more variable-shape tasks reachable via
the new object-selection mechanism (`select_largest`/`select_smallest` +
`commit_selection`, see `arc_env/actions.py`) - 12 same-shape + 14
variable-shape = 26 total.

`d10ecb37`'s solver is `crop(I, ORIGIN, TWO_BY_TWO)` - a single `crop` call
- which is exactly what `commit(row=0, col=0, height=2, width=2)` does
(`arc_env.actions`'s `commit` fuses `crop` with ending the episode; see that
module's docstring), so its entry below uses `"commit"` as the primitive
name, not `"crop"`. `5bd6f4ac`'s solver is `crop(I, tojvec(SIX), THREE_BY_THREE)`
= `crop(I, (0, 6), (3, 3))` - `tojvec`/`THREE_BY_THREE` are arc-dsl constant-
building sugar, not grid-dependent, so this is exactly `commit(0, 6, 3, 3)`,
the same pattern as `d10ecb37`.

This is also exactly the regression-test fixture set
(`tests/test_dsl_regression.py`): each task's solver program, replayed
through `arc_env.actions.execute`, must reproduce the task's expected output
exactly.
"""

import json
from dataclasses import dataclass
from pathlib import Path

TRAINING_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "third_party" / "ARC-AGI" / "data" / "training"
)

# task_id -> the known-correct solver's call sequence, as
# (primitive_name, real_args) pairs transcribed from
# `third_party/arc-dsl/solvers.py`'s `solve_<task_id>`. This is both the
# curated task subset and the regression-test fixture table
# (`tests/test_dsl_regression.py` replays each sequence through
# `arc_env.actions` and checks it reproduces the task's exact output).
CURATED_TASK_IDS = {
    # V1: same-shape.
    "6150a2bd": [("rot180", ())],
    "b1948b0a": [("replace", (6, 2))],
    "3c9b0459": [("rot180", ())],
    "9dfd6313": [("dmirror", ())],
    "c8f0f002": [("replace", (7, 5))],
    "ed36ccf7": [("rot270", ())],
    "74dd1130": [("dmirror", ())],
    "d511f180": [("switch", (5, 8))],
    "67a3c6ac": [("vmirror", ())],
    "68b16354": [("hmirror", ())],
    "0d3d703e": [
        ("switch", (3, 4)),
        ("switch", (8, 9)),
        ("switch", (2, 6)),
        ("switch", (1, 5)),
    ],
    # V3: variable-shape.
    "d10ecb37": [("commit", (0, 0, 2, 2))],  # solver: crop(I, ORIGIN, TWO_BY_TWO)
    "c59eb873": [("upscale", (2,))],
    "9172f3a0": [("upscale", (3,))],
    "5614dbcf": [("replace", (5, 0)), ("downscale", (3,))],
    "46f33fce": [("rot180", ()), ("downscale", (2,)), ("rot180", ()), ("upscale", (4,))],
    # ADR-0010 Phase 1: variable-shape, via the new self-concatenation
    # actions or an already-expressible `commit` call (see module docstring).
    "a416b8f3": [("hconcat_self", ())],
    "6d0aefbc": [("hconcat_self_vmirror", ())],
    "c9e6f938": [("hconcat_self_vmirror", ())],
    "4c4377d9": [("vconcat_self_hmirror_top", ())],
    "6fa7a44f": [("vconcat_self_hmirror_bottom", ())],
    "8be77c9e": [("vconcat_self_hmirror_bottom", ())],
    "5bd6f4ac": [("commit", (0, 6, 3, 3))],  # solver: crop(I, tojvec(SIX), THREE_BY_THREE)
    # ADR-0010 Phase 1, same-shape: bottomhalf then vconcat_self_hmirror_top
    # nets back to the original shape (halves height, then doubles it back).
    "f25ffba3": [("bottomhalf", ()), ("vconcat_self_hmirror_top", ())],
    # ADR-0011 (ADR-0010 Phase 2 Slice 1): select-then-extract, via the new
    # object-selection mechanism. Solvers: `subgrid(argmax(objects(I,T,T,T),
    # size), I)` / `subgrid(argmin(objects(I,T,T,T), size), I)`.
    "1f85a75f": [("select_largest", ()), ("commit_selection", ())],
    "23b5c85d": [("select_smallest", ()), ("commit_selection", ())],
}

# task_id -> whether every train/test pair is same-shape (V1) or not (V3 /
# ADR-0010 Phase 1). Not used by the env/trainers - just documents the split
# for anyone auditing coverage (e.g. `tests/test_task_loader.py`).
VARIABLE_SHAPE_TASK_IDS = {
    "d10ecb37", "c59eb873", "9172f3a0", "5614dbcf", "46f33fce",
    "a416b8f3", "6d0aefbc", "c9e6f938", "4c4377d9", "6fa7a44f", "8be77c9e", "5bd6f4ac",
    "1f85a75f", "23b5c85d",
}


@dataclass(frozen=True)
class Pair:
    input: tuple
    output: tuple


@dataclass(frozen=True)
class Task:
    task_id: str
    train: tuple  # tuple[Pair, ...]
    test: tuple  # tuple[Pair, ...]


def _to_grid(rows: list) -> tuple:
    return tuple(tuple(row) for row in rows)


def load_task(task_id: str) -> Task:
    with open(TRAINING_DATA_DIR / f"{task_id}.json") as f:
        raw = json.load(f)
    train = tuple(Pair(_to_grid(p["input"]), _to_grid(p["output"])) for p in raw["train"])
    test = tuple(Pair(_to_grid(p["input"]), _to_grid(p["output"])) for p in raw["test"])
    return Task(task_id=task_id, train=train, test=test)


def load_curated_tasks() -> dict:
    """Returns `{task_id: Task}` for every task in `CURATED_TASK_IDS`."""

    return {task_id: load_task(task_id) for task_id in CURATED_TASK_IDS}


def iter_curated_pairs():
    """Yields `(task_id, pair_index, Pair)` for every train pair of every
    curated task - the unit of one V1 episode."""

    for task_id, task in load_curated_tasks().items():
        for i, pair in enumerate(task.train):
            yield task_id, i, pair
