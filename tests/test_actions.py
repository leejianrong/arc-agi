"""Unit tests (SLICES.md V1): each curated DSL primitive wrapped as an
action produces the same grid as calling the vendored `arc-dsl` function
directly, for representative inputs."""

import pytest

from arc_env import actions
from arc_env._dsl import constants, dsl
from arc_env.actions import _left_third, _middle_third, _right_third

GRIDS = [
    ((1, 2, 3), (4, 5, 6), (7, 8, 9)),
    ((0, 0), (0, 1)),
    ((3,),),
    ((2, 2, 2, 2),),
]

# ADR-0010 Phase 1's self-concatenation actions (plus ADR-0019's
# `swap_two_least_colors`) are derived (composed from more than one `dsl`
# call, like `fill_cell`/`canvas`/`commit`), not a 1:1 `dsl.<name>` call -
# excluded from the generic 1:1 check below, covered by their own dedicated
# tests instead.
DERIVED_ZERO_ARG_NAMES = {
    "hconcat_self", "hconcat_self_vmirror", "vconcat_self_hmirror_top", "vconcat_self_hmirror_bottom",
    "swap_two_least_colors", "switch_least_most_colors",
    # ADR-0023 (Bucket C).
    "repeat_mirror_tile",
    # ADR-0024: fixed geometric tilings, plus `left_third` (not a bare
    # `dsl.left_third` - there is no such `dsl` function at all).
    "quad_rotate_tile", "quad_mirror_tile", "stack3_vmirror_tile", "left_third",
    # ADR-0025: 9 more derived zero-arg actions - none is a bare 1:1 `dsl`
    # call (there's no `dsl.fill_frontiers`/etc. either).
    "fill_frontiers", "switch_palette_then_zero_five", "tile_alternating_column_mirror",
    "dedupe_grid_both_axes", "fill_holes_in_object_bbox", "tile_by_mostcolor",
    "paint_vmirrored_righthalf_onto_lefthalf", "fill_nonsingleton_foreground",
    "recolor_objects_by_size",
    # ADR-0026: 13 more derived zero-arg actions - none is a bare 1:1 `dsl`
    # call either.
    "canvas_by_symmetry", "tophalf_or_lefthalf_by_equality",
    "mirror_crop_by_rectangle_marker", "diagonal_canvas_by_object_count",
    "canvas_row_by_foreground_count", "bar_chart_canvas_by_size4_count",
    "upscale_by_numcolors_minus_one", "fill_outbox_of_band_intersection",
    "shoot_diagonals_from_objects", "stamp_shape_at_singleton_echoes",
    "crop_to_leastcommon_quadrant", "fill_backdrop_and_box_by_rarity",
    "mirror_border_decoration",
}
DIRECT_ZERO_ARG = [a for a in actions.ZERO_ARG if a.name not in DERIVED_ZERO_ARG_NAMES]

# ADR-0019's `canvas_mostcolor` is likewise derived (`dsl.canvas` plus
# `dsl.mostcolor`, not a bare `dsl.canvas_mostcolor`) - excluded from the
# generic TWO_ARG 1:1 check below for the same reason. ADR-0025's
# `fill_delta_by_color` is also derived (`dsl.fill`/`dsl.delta`/
# `dsl.ofcolor`, not a bare `dsl.fill_delta_by_color`).
DERIVED_TWO_ARG_NAMES = {"canvas_mostcolor", "fill_delta_by_color"}
DIRECT_TWO_ARG = [a for a in actions.TWO_ARG if a.name not in DERIVED_TWO_ARG_NAMES]

# ADR-0021's `fractal_expand_cellwise` is likewise derived (`dsl.cellwise`
# plus `dsl.upscale` plus the `_tile_factor` helper, not a bare
# `dsl.fractal_expand_cellwise`) - excluded from the generic ONE_ARG 1:1
# check below for the same reason. ADR-0023's `fill_inbox_by_dot_color` is
# also derived (`dsl.ofcolor`/`dsl.subgrid`/`dsl.leastcolor`/`dsl.trim`/
# `dsl.inbox`/`dsl.fill`, not a bare `dsl.fill_inbox_by_dot_color`) and its
# one `ArgSpec` is a `COLOR_ARG`, not this group's usual `FACTOR_ARG` - the
# generic check's `dsl_fn(grid, factor)` call wouldn't apply here either
# way, so it's excluded too.
DERIVED_ONE_ARG_NAMES = {"fractal_expand_cellwise", "fill_inbox_by_dot_color"}
DIRECT_ONE_ARG = [a for a in actions.ONE_ARG if a.name not in DERIVED_ONE_ARG_NAMES]


@pytest.mark.parametrize("action", DIRECT_ZERO_ARG, ids=lambda a: a.name)
@pytest.mark.parametrize("grid", GRIDS)
def test_zero_arg_action_matches_direct_dsl_call(action, grid):
    dsl_fn = getattr(dsl, action.name)
    try:
        expected = dsl_fn(grid)
    except Exception:  # noqa: BLE001 - any DSL error means "not applicable here", skip
        pytest.skip(f"{action.name} not well-defined for this grid shape")
    assert action.fn(grid) == expected


@pytest.mark.parametrize("grid", GRIDS)
def test_hconcat_self_matches_dsl_hconcat_with_itself(grid):
    action = actions.ACTIONS[actions.ACTION_BY_NAME["hconcat_self"]]
    assert action.fn(grid) == dsl.hconcat(grid, grid)


@pytest.mark.parametrize("grid", GRIDS)
def test_hconcat_self_vmirror_matches_dsl_hconcat_with_vmirror(grid):
    action = actions.ACTIONS[actions.ACTION_BY_NAME["hconcat_self_vmirror"]]
    assert action.fn(grid) == dsl.hconcat(grid, dsl.vmirror(grid))


@pytest.mark.parametrize("grid", GRIDS)
def test_vconcat_self_hmirror_top_matches_dsl_vconcat_with_hmirror_first(grid):
    action = actions.ACTIONS[actions.ACTION_BY_NAME["vconcat_self_hmirror_top"]]
    assert action.fn(grid) == dsl.vconcat(dsl.hmirror(grid), grid)


@pytest.mark.parametrize("grid", GRIDS)
def test_vconcat_self_hmirror_bottom_matches_dsl_vconcat_with_hmirror_second(grid):
    action = actions.ACTIONS[actions.ACTION_BY_NAME["vconcat_self_hmirror_bottom"]]
    assert action.fn(grid) == dsl.vconcat(grid, dsl.hmirror(grid))


@pytest.mark.parametrize("action", DIRECT_ONE_ARG, ids=lambda a: a.name)
def test_one_arg_action_matches_direct_dsl_call(action):
    grid = ((1, 2, 3, 4), (5, 6, 7, 8))
    dsl_fn = getattr(dsl, action.name)
    for factor in (2, 3):
        assert action.fn(grid, factor) == dsl_fn(grid, factor)


@pytest.mark.parametrize("action", DIRECT_TWO_ARG, ids=lambda a: a.name)
def test_two_arg_action_matches_direct_dsl_call(action):
    grid = ((1, 2, 3), (4, 1, 6))
    dsl_fn = getattr(dsl, action.name)
    assert action.fn(grid, 1, 9) == dsl_fn(grid, 1, 9)


def test_fill_cell_paints_exactly_one_cell():
    grid = ((0, 0), (0, 0))
    result = actions.ACTIONS[actions.ACTION_BY_NAME["fill_cell"]].fn(grid, 5, 1, 0)
    assert result == ((0, 0), (5, 0))


def test_canvas_replaces_the_grid_with_a_fresh_uniformly_colored_one():
    grid = ((1, 2, 3), (4, 5, 6))
    result = actions.ACTIONS[actions.ACTION_BY_NAME["canvas"]].fn(grid, 7, 2, 4)
    assert result == ((7, 7, 7, 7), (7, 7, 7, 7))


# SLICES.md V3 unit test: commit/crop produces the expected sub-grid for a
# range of hand-constructed painted regions (corners, full canvas, single
# cell).
GRID_4X4 = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
)


@pytest.mark.parametrize(
    "row,col,height,width,expected",
    [
        (0, 0, 2, 2, ((1, 2), (5, 6))),  # top-left corner
        (0, 2, 2, 2, ((3, 4), (7, 8))),  # top-right corner
        (2, 0, 2, 2, ((9, 10), (13, 14))),  # bottom-left corner
        (2, 2, 2, 2, ((11, 12), (15, 16))),  # bottom-right corner
        (0, 0, 4, 4, GRID_4X4),  # full canvas
        (1, 1, 1, 1, ((6,),)),  # single cell
    ],
)
def test_commit_crops_to_the_expected_sub_grid(row, col, height, width, expected):
    result = actions.ACTIONS[actions.ACTION_BY_NAME["commit"]].fn(GRID_4X4, row, col, height, width)
    assert result == expected


@pytest.mark.parametrize(
    "action_name,primitive_index_offset",
    [(a.name, i) for i, a in enumerate(actions.ACTIONS)],
)
def test_action_registry_index_is_consistent(action_name, primitive_index_offset):
    assert actions.ACTIONS[actions.ACTION_BY_NAME[action_name]].name == action_name


# ADR-0011: object selection - `select_largest`/`select_smallest` (kind
# "select") and `commit_selection` (kind "act_on_selection").
OBJECTS_GRID = (
    (0, 0, 0, 0),
    (0, 2, 0, 3),
    (0, 2, 0, 0),
    (0, 0, 0, 0),
)  # two 4-connected, single-colored objects (background 0): a 2-cell "2" and a 1-cell "3"


def test_select_largest_picks_the_biggest_object_by_size():
    selected = actions.ACTIONS[actions.ACTION_BY_NAME["select_largest"]].fn(OBJECTS_GRID)
    assert selected == frozenset({(1, 1), (2, 1)})  # the 2-cell "2" object


def test_select_smallest_picks_the_smallest_object_by_size():
    selected = actions.ACTIONS[actions.ACTION_BY_NAME["select_smallest"]].fn(OBJECTS_GRID)
    assert selected == frozenset({(1, 3)})  # the 1-cell "3" object


def test_select_returns_empty_for_a_grid_with_no_objects():
    blank = ((0, 0), (0, 0))
    assert actions.ACTIONS[actions.ACTION_BY_NAME["select_largest"]].fn(blank) == frozenset()


def test_commit_selection_crops_to_the_selected_patchs_bounding_box():
    selected = frozenset({(1, 1), (2, 1)})
    result = actions.ACTIONS[actions.ACTION_BY_NAME["commit_selection"]].fn(OBJECTS_GRID, selected)
    assert result == ((2,), (2,))


class TestExecuteSelectionThreading:
    """`execute`'s `selected`/`selected_region` parameters/returns
    (ADR-0011/ADR-0020), exercised through the full raw-args interface
    rather than `Action.fn` directly. `selected`/`selected_region` are
    2-key dicts (`0`="a"/`1`="b") - these tests only ever touch slot 0."""

    def test_select_action_updates_selection_without_touching_the_grid(self):
        idx = actions.ACTION_BY_NAME["select_largest"]
        new_grid, new_selected, new_selected_region, decoded, valid = actions.execute(
            idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID
        )
        assert valid
        assert new_grid == OBJECTS_GRID
        assert new_selected[0] == frozenset({(1, 1), (2, 1)})
        assert new_selected_region == {0: None, 1: None}
        assert decoded == {}

    def test_select_action_is_invalid_when_no_objects_found(self):
        blank = ((0, 0), (0, 0))
        idx = actions.ACTION_BY_NAME["select_largest"]
        new_grid, new_selected, _new_selected_region, _decoded, valid = actions.execute(
            idx, (0,) * actions.MAX_ARITY, blank
        )
        assert not valid
        assert new_grid == blank
        assert new_selected[0] is None  # unchanged - there was nothing selected before either

    def test_act_on_selection_is_invalid_with_no_current_selection(self):
        idx = actions.ACTION_BY_NAME["commit_selection"]
        new_grid, new_selected, _new_selected_region, _decoded, valid = actions.execute(
            idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID
        )
        assert not valid
        assert new_grid == OBJECTS_GRID
        assert new_selected[0] is None

    def test_act_on_selection_consumes_a_prior_selection(self):
        select_idx = actions.ACTION_BY_NAME["select_smallest"]
        commit_idx = actions.ACTION_BY_NAME["commit_selection"]
        grid, selected, selected_region, _, valid = actions.execute(select_idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID)
        assert valid
        grid, selected, selected_region, _, valid = actions.execute(
            commit_idx, (0,) * actions.MAX_ARITY, grid, selected, selected_region
        )
        assert valid
        assert grid == ((3,),)

    def test_a_successful_ordinary_transform_clears_a_stale_selection(self):
        select_idx = actions.ACTION_BY_NAME["select_largest"]
        vmirror_idx = actions.ACTION_BY_NAME["vmirror"]
        _, selected, selected_region, _, valid = actions.execute(select_idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID)
        assert valid and selected[0]

        _, selected_after, selected_region_after, _, valid = actions.execute(
            vmirror_idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID, selected, selected_region
        )
        assert valid
        assert selected_after == {0: None, 1: None}
        assert selected_region_after == {0: None, 1: None}

    def test_crop_to_selection_is_invalid_with_no_current_selection(self):
        # ADR-0015: same "act_on_selection" invalidity rule as commit_selection.
        idx = actions.ACTION_BY_NAME["crop_to_selection"]
        new_grid, new_selected, _new_selected_region, _decoded, valid = actions.execute(
            idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID
        )
        assert not valid
        assert new_grid == OBJECTS_GRID
        assert new_selected[0] is None

    def test_crop_to_selection_preserves_the_selection_unlike_an_ordinary_transform(self):
        # ADR-0015: `act_on_selection` actions don't clear the selection
        # (only a successful ordinary "transform" does) - a further
        # act_on_selection or the original selection stays usable afterward.
        select_idx = actions.ACTION_BY_NAME["select_smallest"]
        crop_idx = actions.ACTION_BY_NAME["crop_to_selection"]
        grid, selected, selected_region, _, valid = actions.execute(select_idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID)
        assert valid
        grid, selected_after, selected_region_after, _, valid = actions.execute(
            crop_idx, (0,) * actions.MAX_ARITY, grid, selected, selected_region
        )
        assert valid
        assert grid == ((3,),)
        assert selected_after == selected
        assert selected_region_after == selected_region

    def test_a_failed_ordinary_transform_leaves_the_selection_untouched(self):
        select_idx = actions.ACTION_BY_NAME["select_largest"]
        fill_cell_idx = actions.ACTION_BY_NAME["fill_cell"]
        _, selected, selected_region, _, valid = actions.execute(select_idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID)
        assert valid and selected[0]

        # fill_cell's row/col args decode to out-of-bounds coordinates for this grid.
        out_of_bounds_raw = (0, 29, 29, 0)
        _, selected_after, selected_region_after, _, valid = actions.execute(
            fill_cell_idx, out_of_bounds_raw, OBJECTS_GRID, selected, selected_region
        )
        assert not valid
        assert selected_after == selected
        assert selected_region_after == selected_region


# ADR-0020: region-scoped dual-selection mechanism - direct `execute()`-level
# coverage of the new kinds' invalid paths, which none of the 8 curated
# fixture tasks exercises (they're all "happy path", always valid).
REGION_GRID = (
    (1, 1, 1, 1),
    (0, 0, 0, 0),
    (1, 1, 1, 1),
    (0, 0, 0, 0),
)  # tophalf == bottomhalf's own content, so a color-1 select in each region
# lands on the same local coordinates - a real (non-empty) intersection.


def _select_region(slot, region, color, grid=REGION_GRID, selected=None, selected_region=None):
    idx = actions.ACTION_BY_NAME["select_by_color_in_region"]
    raw_args = (slot, region, color) + (0,) * (actions.MAX_ARITY - 3)
    return actions.execute(idx, raw_args, grid, selected, selected_region)


class TestRegionScopedDualSelection:
    """ADR-0020: `select_by_color_in_region`/`combine_slots`/
    `fill_new_canvas`/`fill_onto_region` and the 3 new `Action.kind`
    values' invalid paths."""

    def test_select_by_color_in_region_writes_into_the_named_slot_only(self):
        _, selected, selected_region, _, valid = _select_region(1, 0, 1)  # slot b, tophalf, color 1
        assert valid
        assert selected == {0: None, 1: frozenset({(0, 0), (0, 1), (0, 2), (0, 3)})}
        assert selected_region == {0: None, 1: 0}

    def test_select_by_color_in_region_is_invalid_when_nothing_found_in_that_region(self):
        _, selected, _selected_region, _, valid = _select_region(0, 0, 9)  # color 9 not present
        assert not valid
        assert selected == {0: None, 1: None}  # unchanged

    def test_combine_slots_intersects_both_regions_and_clears_slot_b(self):
        grid, selected, selected_region, _, valid = _select_region(0, 0, 1)  # slot a, tophalf
        assert valid
        grid, selected, selected_region, _, valid = _select_region(1, 1, 1, grid, selected, selected_region)  # slot b, bottomhalf
        assert valid

        combine_idx = actions.ACTION_BY_NAME["combine_slots"]
        raw_args = (0,) * actions.MAX_ARITY  # op=0 -> intersect
        _, combined, combined_region, _, valid = actions.execute(combine_idx, raw_args, grid, selected, selected_region)
        assert valid
        assert combined[0] == frozenset({(0, 0), (0, 1), (0, 2), (0, 3)})
        assert combined[1] is None
        assert combined_region == {0: 0, 1: None}  # slot a keeps its own region tag (tophalf)

    def test_combine_slots_is_invalid_with_an_unpopulated_slot(self):
        grid, selected, selected_region, _, valid = _select_region(0, 0, 1)  # only slot a populated
        assert valid
        combine_idx = actions.ACTION_BY_NAME["combine_slots"]
        _, _, _, _, valid = actions.execute(combine_idx, (0,) * actions.MAX_ARITY, grid, selected, selected_region)
        assert not valid

    def test_combine_slots_is_invalid_when_the_two_regions_shapes_dont_match(self):
        # tophalf (2x4) vs. lefthalf (4x2) of the same 4x4 grid - not
        # comparable positions, even though both selects individually
        # succeed.
        grid, selected, selected_region, _, valid = _select_region(0, 0, 1)  # slot a, tophalf
        assert valid
        grid, selected, selected_region, _, valid = _select_region(1, 2, 0, grid, selected, selected_region)  # slot b, lefthalf, color 0
        assert valid
        combine_idx = actions.ACTION_BY_NAME["combine_slots"]
        _, _, _, _, valid = actions.execute(combine_idx, (0,) * actions.MAX_ARITY, grid, selected, selected_region)
        assert not valid

    def test_fill_new_canvas_paints_slot_as_selection_onto_a_fresh_canvas(self):
        grid, selected, selected_region, _, valid = _select_region(0, 0, 1)  # slot a, tophalf, color 1
        assert valid
        fill_idx = actions.ACTION_BY_NAME["fill_new_canvas"]
        raw_args = (0, 5, 1, 3) + (0,) * (actions.MAX_ARITY - 4)  # bg=0, fill=5, h=2, w=4
        new_grid, selected_after, _, _, valid = actions.execute(fill_idx, raw_args, grid, selected, selected_region)
        assert valid
        assert new_grid == ((5, 5, 5, 5), (0, 0, 0, 0))
        assert selected_after == selected  # act_on_selection never clears/updates selection

    def test_fill_onto_region_paints_slot_as_selection_onto_its_tagged_region(self):
        grid, selected, selected_region, _, valid = _select_region(0, 0, 1)  # slot a, tophalf, color 1
        assert valid
        fill_idx = actions.ACTION_BY_NAME["fill_onto_region"]
        raw_args = (9,) + (0,) * (actions.MAX_ARITY - 1)  # fill=9
        new_grid, _, _, _, valid = actions.execute(fill_idx, raw_args, grid, selected, selected_region)
        assert valid
        assert new_grid == ((9, 9, 9, 9), (0, 0, 0, 0))  # tophalf, its color-1 cells recolored 9

    def test_fill_onto_region_is_invalid_with_no_current_selection(self):
        fill_idx = actions.ACTION_BY_NAME["fill_onto_region"]
        new_grid, selected, _, _, valid = actions.execute(fill_idx, (0,) * actions.MAX_ARITY, REGION_GRID)
        assert not valid
        assert new_grid == REGION_GRID
        assert selected == {0: None, 1: None}


# ADR-0022: the 3-way region split, `fill_slot_onto_region`, and
# `replace_region_and_fill`.
THIRDS_GRID = (
    (0, 1, 2, 3, 4, 5),
    (6, 7, 8, 9, 10, 11),
)  # a 2x6 grid, evenly divisible into 3 2x2 thirds - columns 0-1/2-3/4-5.


def test_left_third_picks_the_leftmost_columns():
    assert _left_third(THIRDS_GRID) == ((0, 1), (6, 7))


def test_middle_third_picks_the_middle_columns():
    assert _middle_third(THIRDS_GRID) == ((2, 3), (8, 9))


def test_right_third_picks_the_rightmost_columns():
    assert _right_third(THIRDS_GRID) == ((4, 5), (10, 11))


# tophalf (region 0) and bottomhalf (region 1) are both 2x4 crops of this
# 4x4 grid - comparable shapes, but distinct regions, so a slot tagged
# bottomhalf can be validly targeted at tophalf.
FILL_SLOT_GRID = (
    (1, 1, 1, 1),
    (0, 0, 0, 0),
    (1, 1, 1, 1),
    (0, 0, 0, 0),
)


def _select_region_raw(slot, region, color, grid=FILL_SLOT_GRID, selected=None, selected_region=None):
    idx = actions.ACTION_BY_NAME["select_by_color_in_region"]
    raw_args = (slot, region, color) + (0,) * (actions.MAX_ARITY - 3)
    return actions.execute(idx, raw_args, grid, selected, selected_region)


def _fill_slot_onto_region_raw(slot, target_region, fill_color, grid, selected, selected_region):
    idx = actions.ACTION_BY_NAME["fill_slot_onto_region"]
    raw_args = (slot, target_region, fill_color) + (0,) * (actions.MAX_ARITY - 3)
    return actions.execute(idx, raw_args, grid, selected, selected_region)


class TestFillSlotOntoRegion:
    """ADR-0022: `fill_slot_onto_region` - an explicit slot painted onto an
    explicit target region, independent of that slot's own tag."""

    def test_fills_slot_bs_selection_onto_an_explicitly_different_target_region(self):
        grid, selected, selected_region, _, valid = _select_region_raw(1, 1, 0)  # slot b, bottomhalf, color 0
        assert valid
        assert selected_region[1] == 1  # tagged bottomhalf

        new_grid, selected_after, selected_region_after, _, valid = _fill_slot_onto_region_raw(
            1, 0, 9, grid, selected, selected_region  # slot b, target=tophalf(0), fill=9
        )
        assert valid
        assert new_grid == ((1, 1, 1, 1), (9, 9, 9, 9))  # tophalf, local row 1 recolored 9
        # Selection state passes through unchanged on success.
        assert selected_after == selected
        assert selected_region_after == selected_region

    def test_invalid_when_the_named_slot_is_empty(self):
        grid, selected, selected_region, _, valid = _select_region_raw(1, 1, 0)  # only slot b populated
        assert valid
        new_grid, selected_after, _, _, valid = _fill_slot_onto_region_raw(
            0, 0, 9, grid, selected, selected_region  # slot a is empty
        )
        assert not valid
        assert new_grid == grid
        assert selected_after == selected

    def test_invalid_when_source_and_target_region_shapes_dont_match(self):
        # slot b tagged righthalf (4x2 crop of this 4x4 grid); tophalf is
        # 2x4 - not a comparable shape.
        grid, selected, selected_region, _, valid = _select_region_raw(1, 3, 0)  # slot b, righthalf, color 0
        assert valid
        assert selected_region[1] == 3
        new_grid, selected_after, selected_region_after, _, valid = _fill_slot_onto_region_raw(
            1, 0, 9, grid, selected, selected_region  # target=tophalf(0)
        )
        assert not valid
        assert new_grid == grid
        assert selected_after == selected
        assert selected_region_after == selected_region


class TestReplaceRegionAndFill:
    """ADR-0022: `replace_region_and_fill` - fuses a region-scoped `replace`
    with `fill` into one atomic `act_on_region_selection` action."""

    def test_replaces_then_fills_within_the_tagged_region(self):
        grid = (
            (1, 1, 1, 1),
            (9, 0, 9, 0),
            (1, 1, 1, 1),
            (0, 0, 0, 0),
        )
        selected_grid, selected, selected_region, _, valid = _select_region_raw(0, 0, 9, grid)  # slot a, tophalf, color 9
        assert valid
        assert selected[0] == frozenset({(1, 0), (1, 2)})

        idx = actions.ACTION_BY_NAME["replace_region_and_fill"]
        raw_args = (9, 0, 8) + (0,) * (actions.MAX_ARITY - 3)  # replacee=9, replacer=0, fill=8
        new_grid, selected_after, selected_region_after, _, valid = actions.execute(
            idx, raw_args, selected_grid, selected, selected_region
        )
        assert valid
        # tophalf's 9s -> 0, then the originally-selected (now-0) cells -> 8.
        assert new_grid == ((1, 1, 1, 1), (8, 0, 8, 0))
        # Selection/region-tag pass through unchanged (act_on_region_selection convention).
        assert selected_after == selected
        assert selected_region_after == selected_region


# ADR-0012: rest of the object-selection menu.
UNIQUE_COLOR_GRID = (
    (0, 0, 0, 0, 0),
    (0, 2, 0, 2, 0),  # two separate single-cell "2" objects - color 2 is NOT unique
    (0, 0, 0, 0, 0),
    (0, 0, 3, 0, 0),  # one "3" object - color 3 IS unique
    (0, 0, 0, 0, 0),
)


def test_select_by_color_picks_the_object_of_the_given_color():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_by_color"]]
    assert action.fn(OBJECTS_GRID, 2) == frozenset({(1, 1), (2, 1)})
    assert action.fn(OBJECTS_GRID, 3) == frozenset({(1, 3)})


def test_select_by_color_returns_empty_when_no_object_has_that_color():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_by_color"]]
    assert action.fn(OBJECTS_GRID, 5) == frozenset()


def test_select_unique_color_picks_only_the_singly_occurring_color():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_unique_color"]]
    assert action.fn(UNIQUE_COLOR_GRID) == frozenset({(3, 2)})


def test_delete_selected_covers_the_selection_with_the_background_color():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["delete_selected"]]
    result = action.fn(OBJECTS_GRID, frozenset({(1, 1), (2, 1)}))
    assert result == ((0, 0, 0, 0), (0, 0, 0, 3), (0, 0, 0, 0), (0, 0, 0, 0))


def test_recolor_selected_fills_the_selection_with_the_given_color():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["recolor_selected"]]
    result = action.fn(OBJECTS_GRID, frozenset({(1, 3)}), 5)
    assert result == ((0, 0, 0, 0), (0, 2, 0, 5), (0, 2, 0, 0), (0, 0, 0, 0))


def test_move_selected_shifts_the_selection_in_the_given_direction():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["move_selected"]]
    result = action.fn(OBJECTS_GRID, frozenset({(1, 3)}), 0)  # 0 = DOWN
    assert result == ((0, 0, 0, 0), (0, 2, 0, 0), (0, 2, 0, 3), (0, 0, 0, 0))


def test_paint_selected_at_stamps_the_selection_without_removing_the_original():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["paint_selected_at"]]
    result = action.fn(OBJECTS_GRID, frozenset({(1, 3)}), 3, 0)
    assert result == ((0, 0, 0, 0), (0, 2, 0, 3), (0, 2, 0, 0), (3, 0, 0, 0))


def test_execute_threads_decoded_args_into_a_select_action():
    idx = actions.ACTION_BY_NAME["select_by_color"]
    raw_args = (2,) + (0,) * (actions.MAX_ARITY - 1)
    _, selected, _selected_region, decoded, valid = actions.execute(idx, raw_args, OBJECTS_GRID)
    assert valid
    assert selected[0] == frozenset({(1, 1), (2, 1)})
    assert decoded == {"color": 2}


# ADR-0013: additional `dsl.objects(...)` connectivity variants, beyond the
# single (univalued=True, diagonal=True, without_bg=True) triple ADR-0011/
# 0012 curate above.

# Four cells of color 2 form a diagonal staircase - (0,0)-(1,1)-(2,2)-(3,3) -
# each pair only diagonally adjacent, never sharing an edge. Color 3's two
# cells ARE edge-adjacent, so they stay one object under either connectivity
# variant. Under diagonal=True (existing `select_largest`), the staircase
# merges into one 4-cell object, bigger than the 3-object's 2 cells - it
# wins. Under diagonal=False (`select_largest_no_diag`), the staircase
# splits into four separate 1-cell objects, so the 3-object's 2 cells win
# instead - a real, different segmentation, not just a different pick.
DIAGONAL_CHAIN_GRID = (
    (2, 0, 0, 0, 3),
    (0, 2, 0, 0, 3),
    (0, 0, 2, 0, 0),
    (0, 0, 0, 2, 0),
)


def test_select_largest_merges_diagonally_adjacent_cells_into_one_object():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_largest"]]
    selected = action.fn(DIAGONAL_CHAIN_GRID)
    assert selected == frozenset({(0, 0), (1, 1), (2, 2), (3, 3)})


def test_select_largest_no_diag_treats_diagonally_adjacent_cells_as_separate_objects():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_largest_no_diag"]]
    selected = action.fn(DIAGONAL_CHAIN_GRID)
    # The diagonal staircase splits into four 1-cell objects; the edge-
    # adjacent pair of "3"s (2 cells) is now the largest.
    assert selected == frozenset({(0, 4), (1, 4)})


# A short-but-wide object (color 4, size 4, height 1), a tall-but-narrow
# object (color 5, size 3, height 3), and the background (color 0) which -
# because `select_tallest` uses without_bg=False - is itself a segmented
# object spanning the full grid height (4). `select_largest` (size,
# without_bg=True) picks color 4 (the biggest of the two non-background
# objects); `select_tallest` (height, without_bg=False) picks the background
# patch instead, since it's taller than both and only visible at all because
# background isn't excluded.
TALLEST_GRID = (
    (4, 4, 4, 4, 5, 0),
    (0, 0, 0, 0, 5, 0),
    (0, 0, 0, 0, 5, 0),
    (0, 0, 0, 0, 0, 0),
)


def test_select_largest_picks_the_biggest_non_background_object_by_size():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_largest"]]
    selected = action.fn(TALLEST_GRID)
    assert selected == frozenset({(0, 0), (0, 1), (0, 2), (0, 3)})  # color 4, size 4


def test_select_tallest_picks_the_tallest_object_including_background():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_tallest"]]
    selected = action.fn(TALLEST_GRID)
    expected_background = frozenset(
        {
            (0, 5),
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 5),
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 5),
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
        }
    )
    assert selected == expected_background


def test_select_no_diag_variants_return_empty_for_a_zero_size_grid_edge_case():
    # Sanity check mirroring `test_select_returns_empty_for_a_grid_with_no_objects`:
    # a uniform (all-background) grid has no non-background objects at all
    # under `select_largest_no_diag`.
    blank = ((0, 0), (0, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_largest_no_diag"]]
    assert action.fn(blank) == frozenset()


# ADR-0015: `select_largest_multicolor` (`univalued=False`) merges edge-
# adjacent cells into one object regardless of color - colors 2 and 3 here
# are adjacent, so `univalued=True` (`select_largest`) would split them into
# two separate 1-cell objects, while `univalued=False` merges them into one
# 2-cell object.
MULTICOLOR_GRID = (
    (0, 0, 0, 0),
    (0, 2, 3, 0),
    (0, 0, 0, 0),
)


def test_select_largest_multicolor_merges_adjacent_different_colored_cells():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_largest_multicolor"]]
    selected = action.fn(MULTICOLOR_GRID)
    assert selected == frozenset({(1, 1), (1, 2)})


def test_select_largest_multicolor_returns_empty_for_a_grid_with_no_objects():
    blank = ((0, 0), (0, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_largest_multicolor"]]
    assert action.fn(blank) == frozenset()


# ADR-0015: `crop_to_selection` wraps the identical `dsl.subgrid` call
# `commit_selection` does - same crop, different action name (see that
# ADR and `tests/test_env.py`'s termination test for why the name matters).
def test_crop_to_selection_crops_to_the_selected_patchs_bounding_box():
    selected = frozenset({(1, 1), (2, 1)})
    result = actions.ACTIONS[actions.ACTION_BY_NAME["crop_to_selection"]].fn(OBJECTS_GRID, selected)
    assert result == ((2,), (2,))


# ADR-0016: `stamp_selected` - shifts the *selected indices* by a fixed
# offset and fills the grid with a color there, WITHOUT clearing the
# original selected cells (unlike `move_selected`).
def test_stamp_selected_fills_a_shifted_copy_without_clearing_the_original():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["stamp_selected"]]
    selected = frozenset({(1, 3)})  # the "3" cell
    result = action.fn(OBJECTS_GRID, selected, 5, 0)  # 0 = DOWN
    # Original (1, 3) still has its original color 3; (2, 3) is newly 5.
    assert result == ((0, 0, 0, 0), (0, 2, 0, 3), (0, 2, 0, 5), (0, 0, 0, 0))


def test_stamp_selected_supports_all_8_directions():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["stamp_selected"]]
    center = frozenset({(2, 2)})
    grid = tuple(tuple(0 for _ in range(5)) for _ in range(5))
    # direction_index -> expected shifted cell, per actions._STAMP_DIRECTIONS'
    # documented order: DOWN, UP, LEFT, RIGHT, UNITY, NEG_UNITY, UP_RIGHT, DOWN_LEFT.
    expected_cells = {
        0: (3, 2),  # DOWN
        1: (1, 2),  # UP
        2: (2, 1),  # LEFT
        3: (2, 3),  # RIGHT
        4: (3, 3),  # UNITY
        5: (1, 1),  # NEG_UNITY
        6: (1, 3),  # UP_RIGHT
        7: (3, 1),  # DOWN_LEFT
    }
    for direction_index, (row, col) in expected_cells.items():
        result = action.fn(grid, center, 9, direction_index)
        assert result[row][col] == 9, direction_index
        assert result[2][2] == 0, direction_index  # original cell untouched


def test_stamp_selected_reuses_the_selection_across_several_calls():
    # A prior selection survives an act_on_selection call (ADR-0015's rule),
    # so several stamp_selected calls in a row can reuse the same selection -
    # exactly how a9f96cdd/d364b489's curated sequences use it.
    select_idx = actions.ACTION_BY_NAME["select_by_color"]
    stamp_idx = actions.ACTION_BY_NAME["stamp_selected"]
    raw_select = (2,) + (0,) * (actions.MAX_ARITY - 1)
    grid, selected, selected_region, _, valid = actions.execute(select_idx, raw_select, OBJECTS_GRID)
    assert valid

    raw_stamp_down = (5, 0) + (0,) * (actions.MAX_ARITY - 2)
    grid, selected_after_first, selected_region_after_first, _, valid = actions.execute(
        stamp_idx, raw_stamp_down, grid, selected, selected_region
    )
    assert valid
    assert selected_after_first == selected  # still usable for a second stamp

    raw_stamp_up = (6, 1) + (0,) * (actions.MAX_ARITY - 2)
    grid, selected_after_second, _selected_region_after_second, _, valid = actions.execute(
        stamp_idx, raw_stamp_up, grid, selected_after_first, selected_region_after_first
    )
    assert valid
    assert selected_after_second == selected


def test_stamp_selected_is_invalid_with_no_current_selection():
    idx = actions.ACTION_BY_NAME["stamp_selected"]
    new_grid, new_selected, _new_selected_region, _decoded, valid = actions.execute(
        idx, (0,) * actions.MAX_ARITY, OBJECTS_GRID
    )
    assert not valid
    assert new_grid == OBJECTS_GRID
    assert new_selected[0] is None


def test_stamp_selected_direction_arg_decodes_mod_8():
    # Raw direction args >= 8 wrap around to the same 8-entry menu, mirroring
    # move_selected's own DIRECTION_ARG mod-4 wraparound.
    idx = actions.ACTION_BY_NAME["stamp_selected"]
    selected = {0: frozenset({(1, 1), (2, 1)}), 1: None}
    for raw_direction, expected_decoded in ((0, 0), (7, 7), (8, 0), (15, 7)):
        raw_args = (0, raw_direction) + (0,) * (actions.MAX_ARITY - 2)
        _, _, _, decoded, valid = actions.execute(idx, raw_args, OBJECTS_GRID, selected)
        assert valid
        assert decoded["direction"] == expected_decoded


# ADR-0019: `canvas_mostcolor`/`swap_two_least_colors` are derived (composed
# from more than one `dsl` call, like `fill_cell`/`canvas`/`commit` above) -
# verified by direct unit test against hand-constructed grids, not the
# generic 1:1-`dsl`-call checks earlier in this file.
def test_canvas_mostcolor_builds_a_fresh_canvas_filled_with_the_grids_most_common_color():
    grid = ((1, 1, 2), (1, 3, 3), (1, 1, 1))  # 1 is most common (6 of 9 cells)
    result = actions.ACTIONS[actions.ACTION_BY_NAME["canvas_mostcolor"]].fn(grid, 2, 4)
    assert result == ((1, 1, 1, 1), (1, 1, 1, 1))


def test_canvas_mostcolor_ignores_the_original_grids_shape():
    # Like `canvas`, the output shape is whatever height/width is passed,
    # independent of the input grid's own shape.
    grid = ((5,),)
    result = actions.ACTIONS[actions.ACTION_BY_NAME["canvas_mostcolor"]].fn(grid, 3, 3)
    assert result == ((5, 5, 5), (5, 5, 5), (5, 5, 5))


def test_swap_two_least_colors_swaps_the_grids_two_least_common_colors():
    # 1 appears once (least common), 2 appears twice (next-least, i.e. the
    # least common of what remains after clearing 1 to 0), 0 is the
    # background/most common color. Clearing 1 to 0 first, then replacing
    # the (now-least-common) 2 with 1, nets out to swapping 1 and 2.
    grid = ((0, 0, 0), (0, 2, 2), (0, 0, 1))
    result = actions.ACTIONS[actions.ACTION_BY_NAME["swap_two_least_colors"]].fn(grid)
    assert result == ((0, 0, 0), (0, 1, 1), (0, 0, 0))


def test_swap_two_least_colors_matches_the_derived_dsl_composition_directly():
    grid = ((3, 3, 4), (3, 5, 5), (3, 3, 3))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["swap_two_least_colors"]]
    a = dsl.leastcolor(grid)
    g2 = dsl.replace(grid, a, 0)
    b = dsl.leastcolor(g2)
    expected = dsl.replace(g2, b, a)
    assert action.fn(grid) == expected


# ADR-0021: 6 more derived actions - see `arc_env/actions.py`'s module
# docstring for what each does.


def test_select_leastcolor_picks_the_grids_least_common_color():
    # Background 0 is most common; of the two objects, color 3 (1 cell) is
    # rarer than color 2 (2 cells) - `select_leastcolor` should pick the
    # single "3" cell, not the "2" object `select_largest`/`select_smallest`
    # would pick.
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_leastcolor"]]
    assert action.fn(OBJECTS_GRID) == frozenset({(1, 3)})


def test_select_leastcolor_returns_empty_for_a_grid_with_no_objects():
    blank = ((0, 0), (0, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_leastcolor"]]
    assert action.fn(blank) == frozenset()


def test_select_all_selects_every_object_in_the_grid():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_all"]]
    assert action.fn(OBJECTS_GRID) == frozenset({(1, 1), (2, 1), (1, 3)})


def test_select_by_size_picks_only_objects_of_the_given_size():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_by_size"]]
    assert action.fn(OBJECTS_GRID, 1) == frozenset({(1, 3)})  # the 1-cell "3" object
    assert action.fn(OBJECTS_GRID, 2) == frozenset({(1, 1), (2, 1)})  # the 2-cell "2" object


def test_select_by_size_returns_empty_when_no_object_has_that_size():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_by_size"]]
    assert action.fn(OBJECTS_GRID, 5) == frozenset()


# A diagonal staircase of 4 differently-colored single cells (2, 3, 4, 6),
# plus an edge-adjacent 2-cell "5" pair. Since `univalued=False`, adjacency
# alone (not color) decides merging - `select_largest_multicolor` (diagonal=
# True) merges the whole staircase into one 4-cell object, bigger than the
# "5" pair; `select_largest_multicolor_no_diag` (diagonal=False) splits the
# staircase into four isolated 1-cell objects, so the edge-adjacent "5" pair
# (2 cells) becomes the largest instead - mirroring the existing
# `select_largest`/`select_largest_no_diag` contrast on `DIAGONAL_CHAIN_GRID`.
MULTICOLOR_DIAGONAL_GRID = (
    (2, 0, 0, 0, 5),
    (0, 3, 0, 0, 5),
    (0, 0, 4, 0, 0),
    (0, 0, 0, 6, 0),
)


def test_select_largest_multicolor_no_diag_treats_diagonally_adjacent_cells_as_separate_objects():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_largest_multicolor_no_diag"]]
    assert action.fn(MULTICOLOR_DIAGONAL_GRID) == frozenset({(0, 4), (1, 4)})


def test_select_largest_multicolor_merges_the_diagonal_staircase_on_the_same_grid():
    # Contrast case: the existing (diagonal=True) selector picks the bigger,
    # diagonally-merged staircase instead, on this exact same grid.
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_largest_multicolor"]]
    assert action.fn(MULTICOLOR_DIAGONAL_GRID) == frozenset({(0, 0), (1, 1), (2, 2), (3, 3)})


def test_select_largest_multicolor_no_diag_returns_empty_for_a_grid_with_no_objects():
    blank = ((0, 0), (0, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["select_largest_multicolor_no_diag"]]
    assert action.fn(blank) == frozenset()


def test_switch_least_most_colors_swaps_the_grids_least_and_most_common_colors():
    # 1 is most common (6 of 9 cells); of the rest, 2 (1 cell) is rarer than
    # 3 (2 cells), so 2 is least common - switch swaps 1 and 2 directly
    # (unlike `swap_two_least_colors`, which operates on the two rarest
    # colors).
    grid = ((1, 1, 2), (1, 3, 3), (1, 1, 1))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["switch_least_most_colors"]]
    assert action.fn(grid) == ((2, 2, 1), (2, 3, 3), (2, 2, 2))


def test_switch_least_most_colors_matches_the_derived_dsl_composition_directly():
    grid = ((4, 4, 4), (4, 7, 7), (4, 4, 4))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["switch_least_most_colors"]]
    expected = dsl.switch(grid, dsl.leastcolor(grid), dsl.mostcolor(grid))
    assert action.fn(grid) == expected


def test_fractal_expand_cellwise_tiles_and_upscales_for_factor_2():
    grid = ((1, 2), (3, 4))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fractal_expand_cellwise"]]
    assert action.fn(grid, 2) == (
        (1, 0, 0, 2),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (3, 0, 0, 4),
    )


def test_fractal_expand_cellwise_tiles_and_upscales_for_factor_3():
    grid = ((1, 0), (0, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fractal_expand_cellwise"]]
    assert action.fn(grid, 3) == (
        (1, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (1, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    )


# ADR-0023 (Bucket C): 3 more derived actions - see `arc_env/actions.py`'s
# module docstring for what each does.

# Four color-5 "dots" at the corners of a 5x5 grid; the interior box
# (`dsl.inbox` of those dots) is the outline of the inner 3x3 - one cell in
# from each dot. The inner 3x3 is mostly color 2 with a single color-3 cell
# at its center, so `dsl.leastcolor(dsl.trim(box))` (the box is the whole
# grid here, so `trim` strips its own outer border down to that same inner
# 3x3) picks 3, and filling the inbox outline with 3 nets out to the whole
# inner 3x3 reading 3 (the center cell was already 3).
INBOX_DOT_GRID = (
    (5, 0, 0, 0, 5),
    (0, 2, 2, 2, 0),
    (0, 2, 3, 2, 0),
    (0, 2, 2, 2, 0),
    (5, 0, 0, 0, 5),
)


def test_fill_inbox_by_dot_color_fills_the_box_interior_with_its_own_least_color():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_inbox_by_dot_color"]]
    result = action.fn(INBOX_DOT_GRID, 5)
    assert result == (
        (5, 0, 0, 0, 5),
        (0, 3, 3, 3, 0),
        (0, 3, 3, 3, 0),
        (0, 3, 3, 3, 0),
        (5, 0, 0, 0, 5),
    )


def test_fill_inbox_by_dot_color_matches_the_derived_dsl_composition_directly():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_inbox_by_dot_color"]]
    dots = dsl.ofcolor(INBOX_DOT_GRID, 5)
    box = dsl.subgrid(dots, INBOX_DOT_GRID)
    expected = dsl.fill(INBOX_DOT_GRID, dsl.leastcolor(dsl.trim(box)), dsl.inbox(dots))
    assert action.fn(INBOX_DOT_GRID, 5) == expected


# A 4x4 grid split into 4 2x2 quadrants: UL has a single color-7 cell, UR a
# single color-4 cell, LL a single color-8 cell, LR (the base the result
# builds from) is blank. `fill_quadrant_from_colors(7, 4, 8)` reads each of
# UL/UR/LL's own designated color's local indices and paints them onto LR
# at the same local positions, sequentially (LL first, then UR, then UL).
QUADRANT_GRID = (
    (7, 0, 0, 0),
    (0, 0, 0, 4),
    (0, 8, 0, 0),
    (0, 0, 0, 0),
)


def test_fill_quadrant_from_colors_paints_each_quadrants_own_color_onto_the_fourth():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_quadrant_from_colors"]]
    result = action.fn(QUADRANT_GRID, 7, 4, 8)
    assert result == ((7, 8), (0, 4))


def test_fill_quadrant_from_colors_leaves_the_fourth_quadrants_own_content_alone_when_no_colors_found():
    # None of UL/UR/LL contain their designated color, so LR (the base the
    # result builds from) comes back exactly as it started.
    grid = (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 9, 0),
    )  # LR quadrant (rows 2-3, cols 2-3) has a lone color-9 cell at local (1, 0)
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_quadrant_from_colors"]]
    result = action.fn(grid, 7, 4, 8)
    assert result == ((0, 0), (9, 0))


def test_repeat_mirror_tile_vconcats_the_grid_with_its_own_hmirror_twice():
    grid = ((1, 2), (3, 4))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["repeat_mirror_tile"]]
    assert action.fn(grid) == (
        (1, 2),
        (3, 4),
        (1, 2),
        (3, 4),
        (1, 2),
    )


def test_repeat_mirror_tile_matches_the_derived_dsl_composition_directly():
    grid = ((5, 6, 7), (8, 9, 1))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["repeat_mirror_tile"]]
    go = dsl.vconcat(grid, dsl.hmirror(grid[:-1]))
    expected = dsl.vconcat(go, dsl.hmirror(go[:-1]))
    assert action.fn(grid) == expected


# ADR-0024: 4 more derived actions from the `free_by_name` re-check.
def test_quad_rotate_tile_tiles_the_grid_with_its_own_90_180_270_rotations():
    grid = ((1, 2), (3, 4))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["quad_rotate_tile"]]
    assert action.fn(grid) == (
        (1, 2, 3, 1),
        (3, 4, 4, 2),
        (2, 4, 4, 3),
        (1, 3, 2, 1),
    )


def test_quad_rotate_tile_matches_the_derived_dsl_composition_directly():
    grid = ((5, 6, 7), (8, 9, 1))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["quad_rotate_tile"]]
    r90, r180, r270 = dsl.rot90(grid), dsl.rot180(grid), dsl.rot270(grid)
    expected = dsl.vconcat(dsl.hconcat(grid, r90), dsl.hconcat(r270, r180))
    assert action.fn(grid) == expected


def test_quad_mirror_tile_stacks_hconcat_self_vmirror_with_its_own_hmirror():
    grid = ((1, 2), (3, 4))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["quad_mirror_tile"]]
    assert action.fn(grid) == (
        (1, 2, 2, 1),
        (3, 4, 4, 3),
        (3, 4, 4, 3),
        (1, 2, 2, 1),
    )


def test_quad_mirror_tile_matches_the_derived_dsl_composition_directly():
    grid = ((5, 6, 7), (8, 9, 1))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["quad_mirror_tile"]]
    top = dsl.hconcat(grid, dsl.vmirror(grid))
    expected = dsl.vconcat(top, dsl.hmirror(top))
    assert action.fn(grid) == expected


def test_stack3_vmirror_tile_is_distinct_from_quad_mirror_tile():
    grid = ((1, 2), (3, 4))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["stack3_vmirror_tile"]]
    assert action.fn(grid) == (
        (4, 3, 3, 4),
        (2, 1, 1, 2),
        (2, 1, 1, 2),
        (4, 3, 3, 4),
        (4, 3, 3, 4),
        (2, 1, 1, 2),
    )


def test_stack3_vmirror_tile_matches_the_derived_dsl_composition_directly():
    grid = ((5, 6, 7), (8, 9, 1))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["stack3_vmirror_tile"]]
    left = dsl.hconcat(dsl.vmirror(grid), grid)
    stacked = dsl.vconcat(left, dsl.hmirror(left))
    stacked = dsl.vconcat(stacked, left)
    expected = dsl.hmirror(stacked)
    assert action.fn(grid) == expected


def test_left_third_action_is_the_literal_same_function_as_the_region_helper():
    # `left_third` (the standalone action) must be the exact same function
    # object as `_left_third` (ADR-0022's private region-scoping helper),
    # not a second implementation that could drift from it.
    action = actions.ACTIONS[actions.ACTION_BY_NAME["left_third"]]
    assert action.fn is _left_third


def test_left_third_action_matches_the_existing_region_helper_usage():
    action = actions.ACTIONS[actions.ACTION_BY_NAME["left_third"]]
    assert action.fn(THIRDS_GRID) == _left_third(THIRDS_GRID) == ((0, 1), (6, 7))


# ADR-0025: 10 more derived actions, 5 new `arc-dsl` primitives (`delta`,
# `frontiers`, `palette`, `dedupe`, `asobject`) get their first curated use.


def test_fill_delta_by_color_fills_the_bounding_box_around_a_colors_cells():
    # `delta(ofcolor(grid, 5))` is every cell in color-5's bounding box that
    # isn't itself color 5 - here the two color-5 corners' 3x3 bounding box,
    # minus those two corners.
    grid = ((5, 0, 0), (0, 0, 0), (0, 0, 5))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_delta_by_color"]]
    assert action.fn(grid, 5, 9) == (
        (5, 9, 9),
        (9, 9, 9),
        (9, 9, 5),
    )


def test_fill_delta_by_color_matches_the_derived_dsl_composition_directly():
    grid = ((0, 0, 4, 0), (0, 0, 0, 0), (4, 0, 0, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_delta_by_color"]]
    expected = dsl.fill(grid, 7, dsl.delta(dsl.ofcolor(grid, 4)))
    assert action.fn(grid, 4, 7) == expected


def test_fill_frontiers_fills_every_full_row_and_column_of_one_color():
    # Row 0 and row 2 are each a single repeated color across the whole
    # width - full-width frontiers. No column is uniform, so only those two
    # rows get filled.
    grid = ((1, 1, 1), (2, 3, 2), (1, 1, 1))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_frontiers"]]
    assert action.fn(grid) == (
        (2, 2, 2),
        (2, 3, 2),
        (2, 2, 2),
    )


def test_fill_frontiers_matches_the_derived_dsl_composition_directly():
    grid = ((4, 4, 4, 4), (5, 6, 7, 8), (4, 4, 4, 4), (1, 1, 1, 1))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_frontiers"]]
    expected = dsl.fill(grid, 2, dsl.merge(dsl.frontiers(grid)))
    assert action.fn(grid) == expected


def test_switch_palette_then_zero_five_switches_the_two_colors_then_zeroes_five():
    grid = ((5, 5, 2), (2, 2, 5), (5, 2, 2))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["switch_palette_then_zero_five"]]
    assert action.fn(grid) == (
        (2, 2, 0),
        (0, 0, 2),
        (2, 0, 0),
    )


def test_switch_palette_then_zero_five_matches_the_derived_dsl_composition_directly():
    grid = ((5, 5, 3), (3, 3, 5), (5, 3, 3))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["switch_palette_then_zero_five"]]
    colors = dsl.palette(grid)
    a, b = dsl.first(colors), dsl.last(colors)
    expected = dsl.replace(dsl.switch(grid, a, b), 5, 0)
    assert action.fn(grid) == expected


def test_tile_alternating_column_mirror_alternates_the_first_column_and_its_mirror():
    # 2 rows x 5 cols, both rows solid: the first column beside its own
    # `hmirror` (row order swapped) tiles out to fill the full width.
    grid = ((3, 3, 3, 3, 3), (9, 9, 9, 9, 9))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["tile_alternating_column_mirror"]]
    assert action.fn(grid) == (
        (3, 9, 3, 9, 3),
        (9, 3, 9, 3, 9),
    )


def test_tile_alternating_column_mirror_matches_the_derived_dsl_composition_directly():
    grid = ((4, 4, 4, 4, 4, 4), (7, 7, 7, 7, 7, 7))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["tile_alternating_column_mirror"]]
    height, width = len(grid), len(grid[0])
    base = dsl.crop(grid, (0, 0), (height, 1))
    unit = dsl.hconcat(base, dsl.hmirror(base))
    tiled = unit
    while len(tiled[0]) < width:
        tiled = dsl.hconcat(tiled, unit)
    expected = dsl.crop(tiled, (0, 0), (height, width))
    assert action.fn(grid) == expected


def test_dedupe_grid_both_axes_crops_to_nonzero_bbox_then_collapses_repeats():
    # The nonzero bounding box is a 4x4 block made of four repeated 2x2
    # quadrants - dedupe on both axes collapses it down to 2x2.
    grid = (
        (0, 0, 0, 0, 0),
        (0, 2, 2, 3, 3),
        (0, 2, 2, 3, 3),
        (0, 4, 4, 5, 5),
        (0, 4, 4, 5, 5),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["dedupe_grid_both_axes"]]
    assert action.fn(grid) == ((2, 3), (4, 5))


def test_dedupe_grid_both_axes_matches_the_derived_dsl_composition_directly():
    grid = ((0, 0, 0), (0, 6, 6), (0, 6, 6))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["dedupe_grid_both_axes"]]
    nonzero = dsl.difference(dsl.asindices(grid), dsl.ofcolor(grid, 0))
    cropped = dsl.subgrid(nonzero, grid)
    expected = dsl.rot270(dsl.dedupe(dsl.rot90(dsl.dedupe(cropped))))
    assert action.fn(grid) == expected


def test_fill_holes_in_object_bbox_fills_the_background_gap_inside_the_object():
    # A single ring-shaped color-8 object with one background (0) cell in
    # its interior - that hole becomes 2, the rest of the grid untouched.
    grid = (
        (0, 0, 0, 0, 0),
        (0, 8, 8, 8, 0),
        (0, 8, 0, 8, 0),
        (0, 8, 8, 8, 0),
        (0, 0, 0, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_holes_in_object_bbox"]]
    assert action.fn(grid) == (
        (0, 0, 0, 0, 0),
        (0, 8, 8, 8, 0),
        (0, 8, 2, 8, 0),
        (0, 8, 8, 8, 0),
        (0, 0, 0, 0, 0),
    )


def test_fill_holes_in_object_bbox_matches_the_derived_dsl_composition_directly():
    grid = ((6, 6, 6), (6, 3, 6), (6, 3, 3))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_holes_in_object_bbox"]]
    objs = dsl.objects(grid, True, True, True)
    obj = dsl.argmax(objs, dsl.size)
    origin = dsl.ulcorner(obj)
    cropped = dsl.subgrid(obj, grid)
    recolored = dsl.replace(cropped, dsl.mostcolor(grid), 2)
    expected = dsl.paint(grid, dsl.shift(dsl.asobject(recolored), origin))
    assert action.fn(grid) == expected


def test_tile_by_mostcolor_pastes_a_grid_copy_at_every_mostcolor_position():
    # `mostcolor` is 1 (three cells vs. one). The grid tiles itself at each
    # of its own 1-valued positions ((0,0), (0,1), (1,0)) and leaves the
    # (1,1) position (value 2) as background.
    grid = ((1, 1), (1, 2))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["tile_by_mostcolor"]]
    assert action.fn(grid) == (
        (1, 1, 1, 1),
        (1, 2, 1, 2),
        (1, 1, 0, 0),
        (1, 2, 0, 0),
    )


def test_tile_by_mostcolor_matches_the_derived_dsl_composition_directly():
    grid = ((3, 3, 5), (3, 5, 5), (5, 5, 5))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["tile_by_mostcolor"]]
    height, width = len(grid), len(grid[0])
    mc = dsl.mostcolor(grid)
    obj = dsl.asobject(grid)
    expected = dsl.canvas(0, (height * height, width * width))
    for row in range(height):
        for col in range(width):
            if grid[row][col] == mc:
                expected = dsl.paint(expected, dsl.shift(obj, (row * height, col * width)))
    assert action.fn(grid) == expected


def test_paint_vmirrored_righthalf_onto_lefthalf_mirrors_objects_across_the_middle():
    grid = ((0, 0, 0, 3), (0, 0, 3, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["paint_vmirrored_righthalf_onto_lefthalf"]]
    assert action.fn(grid) == (
        (3, 0),
        (0, 3),
    )


def test_paint_vmirrored_righthalf_onto_lefthalf_matches_the_derived_dsl_composition_directly():
    grid = ((0, 0, 7, 0), (0, 0, 0, 7))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["paint_vmirrored_righthalf_onto_lefthalf"]]
    mirrored = dsl.vmirror(dsl.righthalf(grid))
    objs = dsl.objects(mirrored, True, False, True)
    expected = dsl.paint(dsl.lefthalf(grid), dsl.merge(objs))
    assert action.fn(grid) == expected


def test_fill_nonsingleton_foreground_recolors_only_multi_cell_objects():
    # Foreground color 3: the L-shaped 3-cell object recolors to 8, the two
    # isolated singleton 3's stay 3.
    grid = ((3, 3, 0), (0, 3, 0), (3, 0, 3))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_nonsingleton_foreground"]]
    assert action.fn(grid) == (
        (8, 8, 0),
        (0, 8, 0),
        (3, 0, 3),
    )


def test_fill_nonsingleton_foreground_matches_the_derived_dsl_composition_directly():
    grid = ((4, 4, 0), (0, 4, 0), (4, 0, 4))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_nonsingleton_foreground"]]
    fg = next(v for row in grid for v in row if v != 0)
    objs = dsl.colorfilter(dsl.objects(grid, True, False, False), fg)
    non_singletons = dsl.difference(objs, dsl.sizefilter(objs, 1))
    expected = dsl.fill(grid, 8, dsl.merge(non_singletons))
    assert action.fn(grid) == expected


def test_recolor_objects_by_size_recolors_size_1_2_3_objects_to_3_2_1():
    grid = (
        (6, 6, 6, 6, 6, 6, 6),
        (6, 0, 6, 0, 0, 6, 6),
        (6, 6, 6, 6, 6, 6, 6),
        (6, 0, 0, 6, 6, 6, 6),
        (6, 0, 6, 6, 6, 6, 6),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["recolor_objects_by_size"]]
    assert action.fn(grid) == (
        (6, 6, 6, 6, 6, 6, 6),
        (6, 3, 6, 2, 2, 6, 6),
        (6, 6, 6, 6, 6, 6, 6),
        (6, 1, 1, 6, 6, 6, 6),
        (6, 1, 6, 6, 6, 6, 6),
    )


def test_recolor_objects_by_size_matches_the_derived_dsl_composition_directly():
    grid = ((9, 9, 9, 9), (9, 0, 9, 0), (9, 9, 9, 9), (9, 0, 0, 9))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["recolor_objects_by_size"]]
    objs = dsl.objects(grid, True, False, True)
    expected = dsl.fill(grid, 3, dsl.merge(dsl.sizefilter(objs, 1)))
    expected = dsl.fill(expected, 2, dsl.merge(dsl.sizefilter(objs, 2)))
    expected = dsl.fill(expected, 1, dsl.merge(dsl.sizefilter(objs, 3)))
    assert action.fn(grid) == expected


# ADR-0026: 13 more derived actions, 10 new `arc-dsl` primitives (`numcolors`,
# `backdrop`, `outbox`, `shoot`, `center`, `normalize`, `leastcommon`, `box`,
# `hfrontier`, `connect`) get their first curated use.


def test_canvas_by_symmetry_returns_1_for_a_symmetric_grid_else_7():
    sym_grid = ((1, 2, 1),)  # a palindromic row is vmirror-symmetric
    nonsym_grid = ((1, 2, 3), (4, 5, 6))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["canvas_by_symmetry"]]
    assert action.fn(sym_grid) == ((1,),)
    assert action.fn(nonsym_grid) == ((7,),)


def test_canvas_by_symmetry_matches_the_derived_dsl_composition_directly():
    grid = ((3, 3), (3, 3))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["canvas_by_symmetry"]]
    symmetric = (
        dsl.hmirror(grid) == grid
        or dsl.vmirror(grid) == grid
        or dsl.dmirror(grid) == grid
        or dsl.cmirror(grid) == grid
    )
    expected = dsl.canvas(1 if symmetric else 7, (1, 1))
    assert action.fn(grid) == expected


def test_tophalf_or_lefthalf_by_equality_picks_tophalf_when_halves_match():
    # tophalf == bottomhalf here, so tophalf wins.
    grid = ((1, 2), (1, 2))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["tophalf_or_lefthalf_by_equality"]]
    assert action.fn(grid) == ((1, 2),)


def test_tophalf_or_lefthalf_by_equality_falls_back_to_lefthalf_otherwise():
    grid = ((1, 2, 3, 4), (5, 6, 7, 8))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["tophalf_or_lefthalf_by_equality"]]
    assert action.fn(grid) == ((1, 2), (5, 6))


def test_tophalf_or_lefthalf_by_equality_matches_the_derived_dsl_composition_directly():
    grid = ((9, 1, 2, 3), (4, 5, 6, 7))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["tophalf_or_lefthalf_by_equality"]]
    top, bottom = dsl.tophalf(grid), dsl.bottomhalf(grid)
    expected = top if top == bottom else dsl.lefthalf(grid)
    assert action.fn(grid) == expected


def test_mirror_crop_by_rectangle_marker_autodetects_the_marker_and_picks_a_crop():
    # Color 1 is a solid 1x2 rectangle (the marker); color 3 is an L-shape
    # (not a rectangle), so it's never mistaken for the marker.
    grid = (
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0),
        (3, 0, 0, 0, 0),
        (3, 3, 0, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["mirror_crop_by_rectangle_marker"]]
    assert action.fn(grid) == ((0, 1),)


def test_mirror_crop_by_rectangle_marker_matches_the_derived_dsl_composition_directly():
    grid = (
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0),
        (5, 0, 0, 0, 0),
        (5, 5, 0, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["mirror_crop_by_rectangle_marker"]]
    marker = next(
        c for c in dsl.palette(grid)
        if dsl.ofcolor(grid, c) and dsl.ofcolor(grid, c) == dsl.backdrop(dsl.ofcolor(grid, c))
    )
    idx = dsl.ofcolor(grid, marker)
    hcrop = dsl.subgrid(idx, dsl.hmirror(grid))
    vcrop = dsl.subgrid(idx, dsl.vmirror(grid))
    expected = vcrop if marker in dsl.palette(hcrop) else hcrop
    assert action.fn(grid) == expected


def test_diagonal_canvas_by_object_count_builds_an_nxn_diagonal_canvas():
    # 3 objects of color 5 on a color-0 background -> a 3x3 canvas, 0
    # background, 5 on the diagonal.
    grid = (
        (0, 0, 0, 0, 0),
        (0, 5, 0, 5, 0),
        (0, 0, 0, 0, 0),
        (0, 5, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["diagonal_canvas_by_object_count"]]
    assert action.fn(grid) == (
        (5, 0, 0),
        (0, 5, 0),
        (0, 0, 5),
    )


def test_diagonal_canvas_by_object_count_matches_the_derived_dsl_composition_directly():
    grid = ((0, 0, 0), (0, 7, 0), (7, 0, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["diagonal_canvas_by_object_count"]]
    n = len(dsl.objects(grid, True, False, True))
    canvas_ = dsl.canvas(dsl.mostcolor(grid), (n, n))
    diagonal = frozenset((i, i) for i in range(n))
    expected = dsl.fill(canvas_, dsl.leastcolor(grid), diagonal)
    assert action.fn(grid) == expected


def test_canvas_row_by_foreground_count_sizes_and_colors_by_foreground_cell_count():
    # 3 cells of the one non-zero color (6) -> a 1x3 row of 6.
    grid = ((0, 0, 0), (0, 6, 0), (6, 0, 6))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["canvas_row_by_foreground_count"]]
    assert action.fn(grid) == ((6, 6, 6),)


def test_canvas_row_by_foreground_count_matches_the_derived_dsl_composition_directly():
    grid = ((0, 4, 0), (4, 0, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["canvas_row_by_foreground_count"]]
    fg = next(v for row in grid for v in row if v != 0)
    count = sum(1 for row in grid for v in row if v == fg)
    expected = dsl.canvas(fg, (1, count))
    assert action.fn(grid) == expected


def test_bar_chart_canvas_by_size4_count_builds_a_fixed_total_5_bar_chart():
    # One size-4 object of color 1 -> 1 cell of color 1 beside 4 cells of
    # the grid's own mostcolor (0).
    grid = ((1, 1, 0, 0), (1, 1, 0, 0), (0, 0, 0, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["bar_chart_canvas_by_size4_count"]]
    assert action.fn(grid) == ((1, 0, 0, 0, 0),)


def test_bar_chart_canvas_by_size4_count_matches_the_derived_dsl_composition_directly():
    grid = ((1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["bar_chart_canvas_by_size4_count"]]
    objs = dsl.objects(grid, True, False, True)
    size4_ones = dsl.sizefilter(dsl.colorfilter(objs, 1), 4)
    n = dsl.size(size4_ones)
    bar = dsl.canvas(1, (1, n))
    rest = dsl.canvas(dsl.mostcolor(grid), (1, 5 - n))
    expected = dsl.hconcat(bar, rest)
    assert action.fn(grid) == expected


def test_upscale_by_numcolors_minus_one_scales_by_numcolors_minus_one():
    # 3 distinct colors (0, 1, 2) -> upscale by 2.
    grid = ((0, 1), (2, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["upscale_by_numcolors_minus_one"]]
    assert action.fn(grid) == (
        (0, 0, 1, 1),
        (0, 0, 1, 1),
        (2, 2, 0, 0),
        (2, 2, 0, 0),
    )


def test_upscale_by_numcolors_minus_one_matches_the_derived_dsl_composition_directly():
    grid = ((0, 1, 2), (3, 0, 0))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["upscale_by_numcolors_minus_one"]]
    expected = dsl.upscale(grid, dsl.numcolors(grid) - 1)
    assert action.fn(grid) == expected


def test_fill_outbox_of_band_intersection_rings_the_two_bands_crossing_point():
    # A full-height column (color 3) crosses a full-width row (color 2) at
    # (1, 1) - the outbox is the 8-cell ring around that single cell.
    grid = (
        (0, 3, 0, 0),
        (2, 2, 2, 2),
        (0, 3, 0, 0),
        (0, 3, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_outbox_of_band_intersection"]]
    assert action.fn(grid) == (
        (4, 4, 4, 0),
        (4, 2, 4, 2),
        (4, 4, 4, 0),
        (0, 3, 0, 0),
    )


def test_fill_outbox_of_band_intersection_matches_the_derived_dsl_composition_directly():
    grid = (
        (0, 0, 5, 0, 0),
        (0, 0, 5, 0, 0),
        (6, 6, 6, 6, 6),
        (0, 0, 5, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_outbox_of_band_intersection"]]
    intersection = frozenset({(2, 2)})
    expected = dsl.fill(grid, 4, dsl.outbox(intersection))
    assert action.fn(grid) == expected


def test_shoot_diagonals_from_objects_shoots_from_each_objects_own_ulcorner():
    # Color-1 object shoots up-left (NEG_UNITY) from its ulcorner; color-2
    # object shoots down-right (UNITY) from its ulcorner.
    grid = (
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 1, 1, 0, 0),
        (0, 0, 1, 1, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["shoot_diagonals_from_objects"]]
    assert action.fn(grid) == (
        (1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 0, 1, 1, 0, 0),
        (0, 0, 1, 1, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    )


def test_shoot_diagonals_from_objects_matches_the_derived_dsl_composition_directly():
    grid = ((0, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["shoot_diagonals_from_objects"]]
    objs = dsl.objects(grid, True, False, True)
    expected = grid
    for obj in dsl.colorfilter(objs, 1):
        expected = dsl.fill(expected, 1, dsl.shoot(dsl.ulcorner(obj), constants.NEG_UNITY))
    for obj in dsl.colorfilter(objs, 2):
        expected = dsl.fill(expected, 2, dsl.shoot(dsl.ulcorner(obj), constants.UNITY))
    assert action.fn(grid) == expected


def test_stamp_shape_at_singleton_echoes_stamps_the_anchor_at_every_marker_dot():
    grid = (
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 6, 7, 0, 0, 0, 0, 0),
        (0, 8, 9, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 5, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["stamp_shape_at_singleton_echoes"]]
    assert action.fn(grid) == (
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 6, 7, 0, 0, 0, 0, 0),
        (0, 8, 9, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 6, 7, 0, 0),
        (0, 0, 0, 0, 8, 9, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
    )


def test_stamp_shape_at_singleton_echoes_matches_the_derived_dsl_composition_directly():
    grid = (
        (0, 0, 0, 0, 0, 0),
        (0, 3, 4, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 2, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["stamp_shape_at_singleton_echoes"]]
    objs = dsl.objects(grid, False, False, True)
    marker_color = dsl.color(dsl.first(dsl.sizefilter(objs, 1)))
    marker_objs = dsl.colorfilter(objs, marker_color)
    anchor = dsl.first(dsl.difference(objs, marker_objs))
    normalized = dsl.normalize(anchor)
    anchor_center = dsl.center(normalized)
    expected = grid
    for obj in marker_objs:
        echo_center = dsl.center(obj)
        offset = (echo_center[0] - anchor_center[0], echo_center[1] - anchor_center[1])
        expected = dsl.paint(expected, dsl.shift(normalized, offset))
    assert action.fn(grid) == expected


def test_crop_to_leastcommon_quadrant_returns_the_odd_one_out_quadrant():
    # 3 quadrants are solid color 1; the bottom-right is solid color 2 (the
    # least-common content among the 4).
    grid = (
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 2, 2),
        (1, 1, 2, 2),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["crop_to_leastcommon_quadrant"]]
    assert action.fn(grid) == ((2, 2), (2, 2))


def test_crop_to_leastcommon_quadrant_matches_the_derived_dsl_composition_directly():
    grid = ((3, 3, 5, 5), (3, 3, 3, 3), (3, 3, 3, 3), (3, 3, 3, 3))
    action = actions.ACTIONS[actions.ACTION_BY_NAME["crop_to_leastcommon_quadrant"]]
    left, right = dsl.lefthalf(grid), dsl.righthalf(grid)
    quadrants = (dsl.tophalf(left), dsl.tophalf(right), dsl.bottomhalf(left), dsl.bottomhalf(right))
    expected = dsl.leastcommon(quadrants)
    assert action.fn(grid) == expected


def test_fill_backdrop_and_box_by_rarity_extends_the_box_to_include_the_dot():
    # A complete 4x4 box (outline 4, interior 6) with a dot (9, the rarest
    # color) below it - the merged bbox extends down to the dot's own row,
    # and the box/interior colors are repainted across that larger extent.
    grid = (
        (0, 0, 0, 0, 0, 0, 0),
        (0, 4, 4, 4, 4, 0, 0),
        (0, 4, 6, 6, 4, 0, 0),
        (0, 4, 6, 6, 4, 0, 0),
        (0, 4, 4, 4, 4, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 9, 0, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_backdrop_and_box_by_rarity"]]
    assert action.fn(grid) == (
        (0, 0, 0, 0, 0, 0, 0),
        (0, 4, 4, 4, 4, 0, 0),
        (0, 4, 6, 6, 4, 0, 0),
        (0, 4, 6, 6, 4, 0, 0),
        (0, 4, 6, 6, 4, 0, 0),
        (0, 4, 6, 6, 4, 0, 0),
        (0, 4, 6, 6, 4, 0, 0),
        (0, 4, 4, 4, 4, 0, 0),
    )


def test_fill_backdrop_and_box_by_rarity_matches_the_derived_dsl_composition_directly():
    grid = (
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 2, 0, 0),
        (0, 2, 7, 2, 0, 0),
        (0, 2, 2, 2, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 5, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_backdrop_and_box_by_rarity"]]
    bg = dsl.mostcolor(grid)
    bg_cells = dsl.ofcolor(grid, bg)
    merged = dsl.merge(dsl.objects(grid, True, False, True))
    dot_color = dsl.leastcolor(grid)
    outline_color = interior_color = None
    for c in dsl.palette(grid):
        if c in (bg, dot_color):
            continue
        cells = dsl.ofcolor(grid, c)
        touches_bg = any(n in bg_cells for cell in cells for n in dsl.dneighbors(cell))
        if touches_bg:
            outline_color = c
        else:
            interior_color = c
    expected = dsl.fill(grid, interior_color, dsl.backdrop(merged))
    expected = dsl.fill(expected, outline_color, dsl.box(merged))
    assert action.fn(grid) == expected


def test_mirror_border_decoration_draws_each_halfs_own_border_and_dot_frontier():
    grid = (
        (0, 0, 0, 0, 0, 0),
        (0, 0, 4, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 8, 0, 0),
        (0, 0, 0, 0, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["mirror_border_decoration"]]
    assert action.fn(grid) == (
        (4, 4, 4, 4, 4, 4),
        (4, 4, 4, 4, 4, 4),
        (4, 0, 0, 0, 0, 4),
        (8, 0, 0, 0, 0, 8),
        (8, 8, 8, 8, 8, 8),
        (8, 8, 8, 8, 8, 8),
    )


def test_mirror_border_decoration_matches_the_derived_dsl_composition_directly():
    grid = (
        (0, 0, 0, 0),
        (0, 3, 0, 0),
        (0, 0, 0, 7),
        (0, 0, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["mirror_border_decoration"]]
    height, width = len(grid), len(grid[0])
    half = height // 2
    bg = dsl.mostcolor(grid)
    top_color = bottom_color = None
    top_loc = bottom_loc = None
    for c in dsl.palette(grid):
        if c == bg:
            continue
        loc = next(iter(dsl.ofcolor(grid, c)))
        if loc[0] < half:
            top_color, top_loc = c, loc
        else:
            bottom_color, bottom_loc = c, loc
    expected = grid
    expected = dsl.fill(expected, top_color, dsl.hfrontier(top_loc))
    expected = dsl.fill(expected, bottom_color, dsl.hfrontier(bottom_loc))
    expected = dsl.fill(expected, top_color, dsl.connect((0, 0), (0, width - 1)))
    expected = dsl.fill(expected, bottom_color, dsl.connect((height - 1, 0), (height - 1, width - 1)))
    expected = dsl.fill(expected, top_color, dsl.connect((0, 0), (half - 1, 0)))
    expected = dsl.fill(expected, top_color, dsl.connect((0, width - 1), (half - 1, width - 1)))
    expected = dsl.fill(expected, bottom_color, dsl.connect((half, 0), (height - 1, 0)))
    expected = dsl.fill(expected, bottom_color, dsl.connect((half, width - 1), (height - 1, width - 1)))
    assert action.fn(grid) == expected


# ADR-0026 hardening: two of the new actions (`fill_backdrop_and_box_by_
# rarity`, `mirror_border_decoration`) auto-detect colors by scanning the
# grid's palette - on a grid that lacks the structure they expect (e.g. no
# distinct outline/interior color pair, or no marker dot in one half), a
# naive implementation left an unassigned `None` reaching `dsl.fill`, which
# silently produced a grid with `None` cells instead of raising. That only
# blew up much later in `arc_env/env.py`'s observation encoding (caught by
# the slow PPO e2e tests, whose real rollouts apply the *full* action space
# to `67a3c6ac`'s own grids, not just each action's target task). The fix is
# twofold: each function now raises on its own "found nothing" branch, and
# `execute()` has a generic `_is_valid_grid` backstop that rejects any
# malformed grid as an invalid no-op (Q7) at the same layer that already
# checks the result's shape.


def test_fill_backdrop_and_box_by_rarity_never_produces_none_cells_on_an_unrelated_grid():
    # A grid shaped like `67a3c6ac`'s (few colors, no distinct outline/
    # interior color pair to classify) - the exact family that broke this
    # action before the fix. The bare function must raise rather than return
    # a grid with `None` cells.
    grid = (
        (8, 8, 8, 8, 8, 8),
        (8, 0, 0, 0, 0, 8),
        (8, 0, 0, 0, 0, 8),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 3, 0, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["fill_backdrop_and_box_by_rarity"]]
    with pytest.raises(ValueError):
        action.fn(grid)


def test_mirror_border_decoration_never_produces_none_cells_on_an_unrelated_grid():
    # Only one marker dot (no second dot in the other half) - the bare
    # function must raise rather than return a grid with `None` cells.
    grid = (
        (0, 0, 0, 0),
        (0, 5, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    action = actions.ACTIONS[actions.ACTION_BY_NAME["mirror_border_decoration"]]
    with pytest.raises(ValueError):
        action.fn(grid)


# Per action, a grid that genuinely lacks the structure that action needs -
# `fill_backdrop_and_box_by_rarity` has no distinct outline/interior color
# pair; `mirror_border_decoration` has only one marker dot (nothing in the
# other half).
_STRUCTURELESS_GRIDS = {
    "fill_backdrop_and_box_by_rarity": (
        (8, 8, 8, 8, 8, 8),
        (8, 0, 0, 0, 0, 8),
        (8, 0, 0, 0, 0, 8),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 3, 0, 0, 0),
    ),
    "mirror_border_decoration": (
        (0, 0, 0, 0),
        (0, 5, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    ),
}


@pytest.mark.parametrize("action_name", sorted(_STRUCTURELESS_GRIDS))
def test_execute_treats_a_structureless_grid_as_an_invalid_no_op(action_name):
    # Through the real `execute()` path: an action that can't do its job on
    # this grid comes back as an invalid no-op (grid unchanged, valid False),
    # never a grid containing `None`, matching Q7's invalid-action contract.
    grid = _STRUCTURELESS_GRIDS[action_name]
    index = actions.ACTION_BY_NAME[action_name]
    new_grid, _, _, _, valid = actions.execute(index, (), grid, None, None)
    assert valid is False
    assert new_grid == grid


def test_is_valid_grid_rejects_malformed_grids():
    # The generic backstop `execute()` funnels every action result through -
    # a `None` cell, an out-of-range color, a ragged row, or a degenerate
    # shape must all be rejected, so any action (existing or future) that
    # produces one is treated as an invalid no-op rather than corrupting
    # downstream observation encoding.
    assert actions._is_valid_grid(((1, 2), (3, 4))) is True
    assert actions._is_valid_grid(((0, 9), (5, 5))) is True
    assert actions._is_valid_grid(((1, None), (3, 4))) is False
    assert actions._is_valid_grid(((1, 2), (3, 10))) is False  # color > 9
    assert actions._is_valid_grid(((1, 2), (3, -1))) is False  # color < 0
    assert actions._is_valid_grid(((1, 2), (3, 4, 5))) is False  # ragged
    assert actions._is_valid_grid(()) is False  # empty


def test_execute_rejects_an_action_that_returns_a_none_cell(monkeypatch):
    # A generic guard test: patch any transform action to return a grid with
    # a `None` cell and confirm `execute()` rejects it as invalid (no-op),
    # rather than passing the malformed grid downstream. `identity` is a
    # zero-arg transform, so this exercises the ordinary-transform branch.
    index = actions.ACTION_BY_NAME["identity"]
    original = actions.ACTIONS[index]
    patched = actions.Action(original.name, lambda grid: ((0, None), (1, 2)), original.args, original.kind)
    patched_actions = list(actions.ACTIONS)
    patched_actions[index] = patched
    monkeypatch.setattr(actions, "ACTIONS", patched_actions)
    new_grid, _, _, _, valid = actions.execute(index, (), ((1, 2), (3, 4)), None, None)
    assert valid is False
    assert new_grid == ((1, 2), (3, 4))
