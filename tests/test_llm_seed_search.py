"""Tests for `scripts/llm_seed_search.py` (F12's "LLM-seeded search"
refinement, ADR-0018): the harness's `describe_task`/`verify`/
`sequence_to_program`/`write_seed_episode` functions, independent of the
one-off experiment run itself (which isn't a pytest fixture - see that
module's docstring).

`scripts/` isn't part of the installed editable package (unlike `arc_env`/
`trainers`/`viz` - see `pyproject.toml`'s `[tool.setuptools.packages.find]`),
so it needs the same repo-root `sys.path` insertion `scripts/
llm_seed_search.py` itself already does for standalone CLI invocation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc_env import actions
from arc_env.task_loader import CURATED_TASK_IDS, load_task
from scripts.llm_seed_search import (
    PROPOSED_SEQUENCES,
    describe_task,
    sequence_to_program,
    verify,
    write_seed_episode,
)
from trainers.gp.evolve import GPConfig, run_gp
from viz.backend.server import read_episode


def test_describe_task_returns_train_test_pairs_and_the_full_action_menu():
    described = describe_task("67a3c6ac")

    assert described["task"].task_id == "67a3c6ac"
    assert len(described["task"].train) > 0
    assert len(described["menu"]) == len(actions.ACTIONS)
    assert {"name", "kind", "args"} <= described["menu"][0].keys()


def test_verify_passes_a_known_correct_sequence():
    report = verify("67a3c6ac", [("vmirror", ())])

    assert report["all_passed"] is True
    assert all(p["passed"] for p in report["pairs"])
    # Every train pair AND the test pair are checked, not just train.
    task = load_task("67a3c6ac")
    assert len(report["pairs"]) == len(task.train) + len(task.test)


def test_verify_fails_a_wrong_sequence_with_a_reason():
    report = verify("67a3c6ac", [("hmirror", ())])  # 67a3c6ac needs vmirror, not hmirror

    assert report["all_passed"] is False
    assert any(not p["passed"] and p["reason"] for p in report["pairs"])


def test_verify_fails_gracefully_on_an_unknown_primitive_name():
    report = verify("67a3c6ac", [("not_a_real_action", ())])

    assert report["all_passed"] is False
    assert "unknown primitive" in report["pairs"][0]["reason"]


def test_verify_fails_on_an_invalid_step_rather_than_crashing():
    # select_smallest with nothing selected yet is fine (a "select" action),
    # but commit_selection with nothing selected is invalid - act_on_selection
    # requires a prior successful select.
    report = verify("1f85a75f", [("commit_selection", ())])

    assert report["all_passed"] is False
    assert "was invalid" in report["pairs"][0]["reason"]


def test_all_three_proposed_sequences_verify_against_their_task():
    """The 3 sequences this pass's experiment actually uses
    (`docs/research/llm-seeded-search-experiment.md`) must each verify
    cleanly - a regression guard against a future action-space change
    silently invalidating one of them."""

    for task_id, sequence in PROPOSED_SEQUENCES.items():
        report = verify(task_id, sequence)
        assert report["all_passed"], (task_id, report)


def test_sequence_to_program_round_trips_through_actions_execute():
    """A converted `Program`, replayed through `actions.execute` directly
    (the same interface GP's own `fitness.run_program` uses), must reproduce
    exactly what `verify`'s higher-level check already confirmed."""

    sequence = PROPOSED_SEQUENCES["ea32f347"]
    program = sequence_to_program(sequence)

    assert len(program) == len(sequence)
    for (primitive_index, raw_args), (name, _args) in zip(program, sequence):
        assert primitive_index == actions.ACTION_BY_NAME[name]
        assert len(raw_args) == actions.MAX_ARITY

    task = load_task("ea32f347")
    for pair in task.train:
        grid = pair.input
        selected = None
        for primitive_index, raw_args in program:
            grid, selected, _decoded, valid = actions.execute(primitive_index, raw_args, grid, selected)
            assert valid
        assert grid == pair.output


def test_sequence_to_program_matches_the_curated_entry_for_matching_tasks():
    """This pass's independently-derived proposals happen to match
    `CURATED_TASK_IDS`'s existing entries exactly (see the research doc) -
    proven here at the `Program` level, not just "both verify"."""

    for task_id, sequence in PROPOSED_SEQUENCES.items():
        assert sequence == CURATED_TASK_IDS[task_id]
        assert sequence_to_program(sequence) == sequence_to_program(CURATED_TASK_IDS[task_id])


def test_write_seed_episode_produces_a_readable_llm_seed_run(tmp_path):
    run_dir = tmp_path / "llm-seed-run"
    sequence = PROPOSED_SEQUENCES["5bd6f4ac"]

    result = write_seed_episode("5bd6f4ac", sequence, run_dir)

    assert result["success"] is True
    episode = read_episode(tmp_path, "llm-seed-run", "best-program")
    assert episode["steps"][-1]["exact_match"] is True

    import json
    with open(run_dir / "run_meta.json") as f:
        meta = json.load(f)
    assert meta["algo"] == "llm-seed"
    assert meta["task_ids"] == ["5bd6f4ac"]


def test_write_seed_episode_output_is_usable_as_a_gp_seed_program():
    """The whole point of `sequence_to_program`: its output should be a
    valid `run_gp(..., seed_programs=[...])` entry that GP recognizes as an
    immediate perfect-fitness solution."""

    sequence = PROPOSED_SEQUENCES["5bd6f4ac"]
    program = sequence_to_program(sequence)
    task = load_task("5bd6f4ac")
    config = GPConfig(population_size=10, n_generations=3, max_program_length=6, seed=0)

    result = run_gp(task, config, seed_programs=[program])

    assert result.best_fitness[0] == 1.0
    assert result.n_generations_run == 1  # solved at generation 0, straight from the seed
