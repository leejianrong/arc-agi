"""Integration + unit + discovery-smoke for the object grammar (SLICES.md V5).

- Integration: a type-invalid composition is rejected at *construction* time and
  a type-valid one executes; the enumerator only ever emits type-valid steps.
- Unit: segmentation + attribute extraction on hand-built grids; derived colors.
- Discovery smoke (fast, seeded, bounded budget): the grammar-directed search
  finds an exact program for a non-set-op task from scratch AND a set-op task via
  its grammar-derived skeleton — the object-track analogue of V4's `vmirror`
  test, kept in the default (non-slow) layer.
"""

import random

import pytest

from arc_env.task_loader import load_task
from object_env import objects
from object_env.actions import ACTION_BY_NAME, ACTIONS
from object_env.colors import COLOR_DOMAIN, DerivedColor, resolve
from object_env.grammar import (
    GrammarError,
    Program,
    Step,
    enumerate_skeletons,
    legal_steps,
    sample_program,
    skeleton_of,
)
from object_env.programs import build
from object_env.search import arg_search, discover, evaluate
from object_env.state import ObjState
from object_env.types import Obj


# ---------------- integration: the grammar constrains construction ----------------
def _step(name, **args):
    return Step(ACTION_BY_NAME[name], args)


def test_combine_before_select_is_rejected_at_construction():
    with pytest.raises(GrammarError):
        Program([_step("combine", op=0)])  # reads set_a/set_b, both unfilled


def test_paint_before_split_is_rejected_at_construction():
    with pytest.raises(GrammarError):
        Program([_step("paint_canvas", bg=0, fill=3)])


def test_move_before_select_is_rejected_at_construction():
    with pytest.raises(GrammarError):
        Program([_step("move_object", direction=0)])  # reads obj, unfilled


def test_missing_value_arg_is_rejected():
    with pytest.raises(GrammarError):
        Program([_step("split")])  # axis arg missing


def test_valid_composition_constructs_and_runs():
    prog = Program([_step("split", axis=0), _step("select_color", slot=0, region=0, color=0)])
    assert len(prog) == 2
    # runs without error on a 2x2 grid; select on region reachable
    prog.run(((1, 0), (0, 1)))


def test_enumerator_only_yields_type_valid_steps():
    initial = frozenset({"grid"})
    names = {s.action.name for s in legal_steps(initial, ACTIONS)}
    # nothing that reads a not-yet-filled slot is offered from a bare grid
    assert "combine" not in names and "paint_canvas" not in names and "move_object" not in names
    assert {"split", "select_largest", "replace_color"} <= names


def test_sampled_programs_are_all_type_valid():
    rng = random.Random(0)
    for _ in range(1000):
        sample_program(rng, ACTIONS, 6)  # constructs -> type_check passes, or raises


def test_skeletons_are_grid_reachable_and_finite():
    skels = list(enumerate_skeletons(ACTIONS, 2))
    assert all(sk[0][0].reads(sk[0][1]) == ("grid",) for sk in skels)  # depth-1 reads grid
    assert len(skels) == 9 + 100  # counts checked empirically


def test_skeleton_of_strips_value_args_but_stays_valid():
    skeleton = skeleton_of(build("6430c8c4"))
    assert [a.name for a, _ in skeleton] == [
        "split", "select_color", "select_color", "combine", "paint_canvas",
    ]
    # structural args kept, value args (color/op/axis) dropped
    assert skeleton[1][1] == {"slot": 0, "region": 0}


# ---------------- unit: segmentation + attributes ----------------
def test_segment_finds_expected_objects():
    grid = ((1, 1, 0), (0, 0, 0), (0, 2, 2))
    objs = objects.segment(grid, univalued=True, diagonal=True, without_bg=True)
    by_color = {o.color: o for o in objs}
    assert set(by_color) == {1, 2}
    assert by_color[1].cells == frozenset({(0, 0), (0, 1)})
    assert by_color[2].size == 2


def test_obj_attributes():
    o = Obj(color=3, cells=frozenset({(1, 1), (1, 2), (3, 1)}))
    assert o.bbox == (1, 1, 3, 2)
    assert o.height == 3 and o.width == 2 and o.size == 3
    # shape signature is translation-invariant
    shifted = Obj(color=3, cells=frozenset({(6, 4), (6, 5), (8, 4)}))
    assert o.shape_signature == shifted.shape_signature


def test_select_largest_and_smallest():
    grid = ((1, 1, 0), (1, 0, 0), (0, 0, 2))
    assert objects.select_largest(grid).color == 1
    assert objects.select_smallest(grid).color == 2
    assert objects.select_largest(((0, 0), (0, 0))) is None  # empty -> no object


# ---------------- unit: derived colors ----------------
def test_derived_color_resolves_against_grid():
    grid = ((5, 5, 5), (5, 2, 5), (5, 5, 5))  # most=5, least=2
    state = ObjState(grid=grid)
    assert resolve(DerivedColor("most"), state) == 5
    assert resolve(DerivedColor("least"), state) == 2
    assert resolve(7, state) == 7  # a literal is itself


def test_color_domain_includes_literals_and_derived():
    assert set(range(10)) <= set(COLOR_DOMAIN)
    assert any(isinstance(c, DerivedColor) for c in COLOR_DOMAIN)


# ---------------- discovery smoke (fast, deterministic) ----------------
def test_discover_non_setop_from_scratch():
    task = load_task("b1948b0a")  # depth-1 transform, found in well under budget
    solved, _evals, _best, prog = discover(task, budget=5_000, seed=0)
    assert solved and evaluate(prog, task)[0] >= 1.0


def test_arg_search_solves_setop_via_derived_skeleton():
    task = load_task("f2829549")  # fastest set-op at seed 0 (~362 evals)
    skeleton = skeleton_of(build("f2829549"))
    solved, _evals, best = arg_search(task, skeleton, budget=5_000, seed=0)
    assert solved and best[0] >= 1.0
