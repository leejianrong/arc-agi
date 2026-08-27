"""Tests for `arc_env.task_loader` - specifically that
`VARIABLE_SHAPE_TASK_IDS` accurately reflects each task's actual train/test
pairs (V1's same-shape tasks vs. V3's variable-shape ones), since nothing
else in the codebase re-derives this classification."""

from arc_env.task_loader import CURATED_TASK_IDS, VARIABLE_SHAPE_TASK_IDS, load_task


def _shape(grid: tuple) -> tuple:
    return (len(grid), len(grid[0]))


def test_variable_shape_task_ids_is_a_subset_of_curated_task_ids():
    assert VARIABLE_SHAPE_TASK_IDS <= set(CURATED_TASK_IDS)


def test_v1_same_shape_tasks_are_genuinely_same_shape():
    for task_id in set(CURATED_TASK_IDS) - VARIABLE_SHAPE_TASK_IDS:
        task = load_task(task_id)
        for pair in (*task.train, *task.test):
            assert _shape(pair.input) == _shape(pair.output), task_id


def test_v3_variable_shape_tasks_have_at_least_one_shape_changing_pair():
    for task_id in VARIABLE_SHAPE_TASK_IDS:
        task = load_task(task_id)
        pairs = (*task.train, *task.test)
        assert any(_shape(p.input) != _shape(p.output) for p in pairs), task_id
