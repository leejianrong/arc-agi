#!/usr/bin/env python3
"""Prints an ARC-AGI-1 task's train/test grids straight to the terminal.

Built so an agentic coding session (this repo's own use case: Claude Code
referring to "task 928ad970") and the human working alongside it can look at
the exact same rendering without opening the visualizer or reading raw JSON.

Renders each grid as a block of 2-char-wide truecolor cells, colored with
ARC-AGI's own canonical 10-color palette (the same one `viz/frontend/src/
palette.ts` transcribes from `third_party/ARC-AGI/apps/css/common.css`, per
ADR-0007's "follow apps/js's palette conventions" decision - kept in sync by
hand, same as that TS file already is with the CSS). Falls back to a plain
bracketed-digit rendering when stdout isn't a TTY, `--no-color` is passed, or
`NO_COLOR` is set (https://no-color.org) - piping to a file/log or a
non-truecolor terminal still gets a readable grid. Pass `--color` to force
truecolor escapes on regardless of TTY detection (e.g. when stdout is
captured/piped by a harness that still renders the bytes in a real
terminal, like this agent's own tool-output pane) - takes precedence over
the TTY check but not over `--no-color`/`NO_COLOR`.

    uv run python scripts/print_task.py 928ad970
    uv run python scripts/print_task.py 928ad970 --pair train 0
    uv run python scripts/print_task.py 928ad970 --no-color
    uv run python scripts/print_task.py 928ad970 --color
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = (
    REPO_ROOT / "third_party" / "ARC-AGI" / "data" / "training",
    REPO_ROOT / "third_party" / "ARC-AGI" / "data" / "evaluation",
)

# Same 10 colors as `viz/frontend/src/palette.ts` (RGB, not hex, since ANSI
# truecolor escapes want the components separately).
PALETTE = (
    (0, 0, 0),  # 0 black
    (0, 116, 217),  # 1 blue
    (255, 65, 54),  # 2 red
    (46, 204, 64),  # 3 green
    (255, 220, 0),  # 4 yellow
    (170, 170, 170),  # 5 grey
    (240, 18, 190),  # 6 fuschia
    (255, 133, 27),  # 7 orange
    (127, 219, 255),  # 8 teal
    (135, 12, 37),  # 9 brown
)

# ARC task ids are 8 lowercase-hex-digit filename stems - reject anything
# else before it ever reaches the filesystem (same convention `viz/backend/
# play.py` uses for its own task_id/run_id path-traversal allowlist).
_TASK_ID_RE = re.compile(r"^[0-9a-f]{8}$")


class TaskNotFoundError(Exception):
    pass


def find_task_path(task_id: str) -> Path:
    if not _TASK_ID_RE.match(task_id):
        raise ValueError(f"{task_id!r} doesn't look like an ARC-AGI-1 task id (expected 8 lowercase hex digits)")
    for data_dir in DATA_DIRS:
        candidate = data_dir / f"{task_id}.json"
        if candidate.is_file():
            return candidate
    raise TaskNotFoundError(f"{task_id!r} not found under third_party/ARC-AGI/data/{{training,evaluation}}")


def load_task(task_id: str) -> dict:
    with open(find_task_path(task_id)) as f:
        return json.load(f)


def _use_color(no_color_flag: bool, color_flag: bool) -> bool:
    if no_color_flag or os.environ.get("NO_COLOR"):
        return False
    return color_flag or sys.stdout.isatty()


def _render_grid_lines(grid: list, color: bool) -> list:
    """One rendered line per grid row, each cell 2 characters wide."""

    lines = []
    for row in grid:
        if color:
            cells = []
            for value in row:
                r, g, b = PALETTE[value] if 0 <= value < len(PALETTE) else (34, 34, 34)
                cells.append(f"\x1b[48;2;{r};{g};{b}m  \x1b[0m")
            lines.append("".join(cells))
        else:
            lines.append("".join(f"[{value}]" for value in row))
    return lines


def render_pair(input_grid: list, output_grid: list, color: bool) -> str:
    """Input grid and output grid, side by side, joined by a "->" on the
    vertically-centered row (or the first row of an even-height pair)."""

    in_lines = _render_grid_lines(input_grid, color)
    out_lines = _render_grid_lines(output_grid, color)
    cell_width = 2 if color else 3
    in_visual_width = len(input_grid[0]) * cell_width

    height = max(len(in_lines), len(out_lines))
    arrow_row = (height - 1) // 2
    rows = []
    for i in range(height):
        left = in_lines[i] if i < len(in_lines) else ""
        left_visual_len = len(input_grid[0]) * cell_width if i < len(in_lines) else 0
        pad = " " * max(0, in_visual_width - left_visual_len)
        gap = " -> " if i == arrow_row else "    "
        right = out_lines[i] if i < len(out_lines) else ""
        rows.append(f"{left}{pad}{gap}{right}")
    return "\n".join(rows)


def print_task(task_id: str, split: str | None, pair_index: int | None, color: bool) -> None:
    task = load_task(task_id)
    splits = [split] if split else ["train", "test"]
    for s in splits:
        pairs = task.get(s, [])
        indices = [pair_index] if pair_index is not None else range(len(pairs))
        for i in indices:
            pair = pairs[i]
            in_grid, out_grid = pair["input"], pair["output"]
            print(f"\n{task_id} {s}[{i}]  in {len(in_grid)}x{len(in_grid[0])} -> out {len(out_grid)}x{len(out_grid[0])}")
            print(render_pair(in_grid, out_grid, color))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_id", help="8-hex-digit ARC-AGI-1 task id (filename stem, e.g. 928ad970)")
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("SPLIT", "INDEX"),
        help="print just one pair, e.g. --pair train 0 (default: every train and test pair)",
    )
    parser.add_argument("--no-color", action="store_true", help="force plain bracketed-digit rendering")
    parser.add_argument("--color", action="store_true", help="force truecolor rendering regardless of TTY detection")
    args = parser.parse_args()

    split, pair_index = (None, None)
    if args.pair:
        split, pair_index = args.pair[0], int(args.pair[1])
        if split not in ("train", "test"):
            parser.error("--pair's SPLIT must be 'train' or 'test'")

    try:
        print_task(args.task_id, split, pair_index, color=_use_color(args.no_color, args.color))
    except (ValueError, TaskNotFoundError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
