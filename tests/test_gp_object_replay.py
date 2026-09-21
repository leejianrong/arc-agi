"""Tests for `trainers.gp_object.replay` - a GP-found object-grammar genome
replayed step by step produces the same step-trace shape
`trainers/gp/replay.py`'s `program_to_episode_trace` does, so it logs
through the same `EpisodeWriter` path with no special-casing, and - the
KAN-1546 regression flat GP already guards - never stops early on a
transient intermediate target match."""

from arc_env.tasks import Pair
from object_env.actions import ACTIONS
from object_env.grammar import legal_steps
from trainers.gp_object.replay import genome_to_episode_trace


def _gene_for(action_name, args, filled=frozenset({"grid"})):
    menu = legal_steps(filled, ACTIONS)
    for i, step in enumerate(menu):
        if step.action.name == action_name and step.args == args:
            return i
    raise AssertionError(f"{action_name}{args} is not legal given filled={sorted(filled)}")


def test_genome_to_episode_trace_matches_a_known_solving_genome():
    genome = [_gene_for("replace_color", {"from_color": 6, "to_color": 2})]
    pair = Pair(input=((6, 0), (0, 6)), output=((2, 0), (0, 2)))

    result = genome_to_episode_trace(genome, pair)

    assert result["success"] is True
    assert len(result["steps"]) == 1
    assert result["steps"][0]["action_name"] == "replace_color"
    assert result["steps"][0]["exact_match"] is True
    assert result["steps"][0]["grid_after"] == pair.output
    assert result["steps"][0]["truncated"] is True
    assert result["steps"][0]["terminated"] is False


def test_empty_genome_scores_its_static_identity_endpoint():
    grid = ((1, 2), (3, 4))
    pair = Pair(input=grid, output=grid)

    result = genome_to_episode_trace([], pair)

    assert result["steps"] == []
    assert result["success"] is True


def test_trace_does_not_stop_at_an_intermediate_target_match():
    # KAN-1546 regression, object-track analogue: step 1 already matches the
    # target, but the static genome has a second, destructive gene after it -
    # the trace must keep running and score the FINAL grid, not the
    # transient match.
    genome = [
        _gene_for("replace_color", {"from_color": 1, "to_color": 2}),
        _gene_for("replace_color", {"from_color": 2, "to_color": 9}),
    ]
    pair = Pair(input=((1, 2), (3, 4)), output=((2, 2), (3, 4)))  # == replace(1->2)(input)

    result = genome_to_episode_trace(genome, pair)

    assert len(result["steps"]) == 2
    assert result["steps"][0]["exact_match"] is True
    assert result["steps"][0]["terminated"] is False
    assert result["steps"][0]["truncated"] is False  # not the last step
    assert result["steps"][1]["exact_match"] is False
    assert result["steps"][1]["truncated"] is True
    assert result["success"] is False
