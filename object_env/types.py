"""The grammar's type set and the first-class `Obj`.

ADR-0029 names a small type set — Grid, Region, IndexSet, Object, Color, SetOp,
Axis. Two kinds live in that list:

- **Dataflow types** (`GRID`, `REGION`, `INDEXSET`, `OBJECT`) name what a
  *named slot* in `ObjState` holds. The grammar type-checks the read-before-
  write structure over these (see `grammar.py`).
- **Parameter types** (`COLOR`, `SETOP`, `AXIS`) name a small enumerable
  argument domain an action draws from.

V5 additionally needs two parameter domains the ADR's 7-name sketch didn't
enumerate — `DIRECTION` (a 4-way move offset, distinct from the 2-way split
`AXIS`) and `SIZE` (a structural canvas dimension). Surfacing exactly this kind
of type/verb-set adjustment while the object *substrate* stays put is what the
slice exists to do (SLICES.md V5 "rests on assumptions").
"""

from dataclasses import dataclass
from enum import Enum

Grid = tuple  # tuple[tuple[int, ...], ...]
Indices = frozenset  # frozenset[tuple[int, int]]


class ArgType(Enum):
    # dataflow types (named slots)
    GRID = "Grid"
    REGION = "Region"
    INDEXSET = "IndexSet"
    OBJECT = "Object"
    # parameter types
    COLOR = "Color"
    SETOP = "SetOp"
    AXIS = "Axis"
    DIRECTION = "Direction"  # V5 addition (4-way), see module docstring
    SIZE = "Size"  # V5 addition (structural dimension)


@dataclass(frozen=True)
class Obj:
    """A first-class, addressable object: a color plus the cells it occupies,
    with queryable attributes derived on demand. `cells` are absolute grid
    indices (the `dsl.toindices` form), so an `Obj` re-composes onto the grid
    via `dsl.toobject(cells, grid)` / `dsl.subgrid(cells, grid)`."""

    color: int
    cells: Indices
    region: str | None = None  # optional region membership tag
    role: str | None = None  # extension point (signaller/director etc.)

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def bbox(self) -> tuple:
        """(min_row, min_col, max_row, max_col) — empty -> a zero box."""
        if not self.cells:
            return (0, 0, 0, 0)
        rows = [i for i, _ in self.cells]
        cols = [j for _, j in self.cells]
        return (min(rows), min(cols), max(rows), max(cols))

    @property
    def height(self) -> int:
        r0, _, r1, _ = self.bbox
        return r1 - r0 + 1 if self.cells else 0

    @property
    def width(self) -> int:
        _, c0, _, c1 = self.bbox
        return c1 - c0 + 1 if self.cells else 0

    @property
    def shape_signature(self) -> Indices:
        """Cells normalized so the bounding box's upper-left sits at the
        origin — a translation-invariant signature two same-shaped objects
        share regardless of where they sit on the grid."""
        if not self.cells:
            return frozenset()
        r0, c0, _, _ = self.bbox
        return frozenset((i - r0, j - c0) for i, j in self.cells)
