"""The 8 hand-written object-space programs for the set-op cluster (Gate 1).

Each program is a list of `(action_index, raw_args)` over `object_actions`.
Action indices: 0 split, 1 select_color, 2 combine, 3 paint_canvas,
4 paint_onto_region. raw_args decode by modulo (see object_actions): axis
0=top/bottom 1=left/right; slot/region 0=a 1=b; op 0=intersect 1=union
2=symdiff 3=difference; color = value 0..9.

Every one of the 8 uses the SAME five-verb vocabulary — only the args differ.
That generality (one small reusable vocabulary, not 8 bespoke actions) is the
Gate-1-beyond-expressibility claim.
"""

# fmt: off
PROGRAMS = {
    # tophalf/bottomhalf, blanks (0) of each, intersect, paint green(3) on a 0-canvas
    "6430c8c4": [(0, (0, 0, 0)), (1, (0, 0, 0)), (1, (1, 1, 0)), (2, (0, 0, 0)), (3, (0, 3, 0))],
    # same shape, paint red(2)
    "94f9d214": [(0, (0, 0, 0)), (1, (0, 0, 0)), (1, (1, 1, 0)), (2, (0, 0, 0)), (3, (0, 2, 0))],
    # same, but canvas prefilled green(3) and intersection punched to 0
    "ce4f8723": [(0, (0, 0, 0)), (1, (0, 0, 0)), (1, (1, 1, 0)), (2, (0, 0, 0)), (3, (3, 0, 0))],
    # left/right split, intersect blanks, paint green(3)
    "f2829549": [(0, (1, 0, 0)), (1, (0, 0, 0)), (1, (1, 1, 0)), (2, (0, 0, 0)), (3, (0, 3, 0))],
    # top/bottom, intersect blanks, paint red(2)
    "fafffa47": [(0, (0, 0, 0)), (1, (0, 0, 0)), (1, (1, 1, 0)), (2, (0, 0, 0)), (3, (0, 2, 0))],
    # top/bottom, symmetric-difference of blanks, paint green(3)
    "99b1bc43": [(0, (0, 0, 0)), (1, (0, 0, 0)), (1, (1, 1, 0)), (2, (2, 0, 0)), (3, (0, 3, 0))],
    # top/bottom, color-2 cells, symmetric-difference, paint green(3) on 6x5 (=region shape)
    "3428a4f5": [(0, (0, 0, 0)), (1, (0, 0, 2)), (1, (1, 1, 2)), (2, (2, 0, 0)), (3, (0, 3, 0))],
    # left/right, union of color-4(left) and color-3(right), paint brown(6) ONTO the left region
    "dae9d2b5": [(0, (1, 0, 0)), (1, (0, 0, 4)), (1, (1, 1, 3)), (2, (1, 0, 0)), (4, (6, 0, 0))],
}
# fmt: on

TARGET_TASK_IDS = list(PROGRAMS.keys())
