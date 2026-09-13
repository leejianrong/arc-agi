"""Unit tests for `scripts/print_task.py`."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from print_task import (
    PALETTE,
    TaskNotFoundError,
    find_task_path,
    load_task,
    render_pair,
)


def test_find_task_path_rejects_non_hex_ids():
    with pytest.raises(ValueError):
        find_task_path("../../etc/passwd")
    with pytest.raises(ValueError):
        find_task_path("not-a-task-id")
    with pytest.raises(ValueError):
        find_task_path("0123456")  # 7 chars, one short


def test_find_task_path_rejects_unknown_task_id():
    with pytest.raises(TaskNotFoundError):
        find_task_path("ffffffff")


def test_find_task_path_and_load_task_find_a_real_training_task():
    path = find_task_path("6150a2bd")
    assert path.name == "6150a2bd.json"
    task = load_task("6150a2bd")
    assert "train" in task and "test" in task
    assert len(task["train"]) > 0


def test_render_pair_same_shape_aligns_arrow_and_colors_every_cell():
    input_grid = [[0, 1], [2, 3]]
    output_grid = [[3, 2], [1, 0]]
    plain = render_pair(input_grid, output_grid, color=False)
    lines = plain.split("\n")
    assert len(lines) == 2
    assert "->" in lines[0] or "->" in lines[1]
    assert lines[0].startswith("[0][1]")
    assert lines[0].rstrip().endswith("[3][2]")

    colored = render_pair(input_grid, output_grid, color=True)
    # every color-mode cell renders as a truecolor background escape + 2
    # spaces + reset - just check the exact RGB triples for corner colors 0/3
    # both appear (0,0,0 and 46,204,64).
    assert "48;2;0;0;0" in colored
    assert "48;2;46;204;64" in colored


def test_render_pair_handles_different_shapes_without_index_error():
    input_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    output_grid = [[1, 1]]
    # Should not raise, and should still show every input row.
    plain = render_pair(input_grid, output_grid, color=False)
    assert len(plain.split("\n")) == 3


def test_palette_has_ten_colors_matching_viz_frontend():
    assert len(PALETTE) == 10
    assert PALETTE[0] == (0, 0, 0)
    assert PALETTE[1] == (0, 116, 217)
