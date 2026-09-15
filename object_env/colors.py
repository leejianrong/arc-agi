"""Derived colors (ADR-0029 commitment #3).

A COLOR parameter may be a literal 0-9 *or* a query resolved against the current
state at execution time — "the grid's most/least-common color", "the region's
background". Because these read the actual grid rather than a hardcoded constant,
a found program keeps working when `re-arc` randomizes the palette (the
ADR-0025/0026 failure mode the literal solvers hit). The verify harness measures
exactly that generalization.
"""

from dataclasses import dataclass

from arc_env._dsl import dsl
from object_env.types import Grid


@dataclass(frozen=True)
class DerivedColor:
    """A grid-derived color parameter. `query` names how to read it; `source`
    picks which slot to read from (default the working grid)."""

    query: str  # "most" | "least"
    source: str = "grid"  # slot name to resolve against

    def __str__(self) -> str:
        return f"{self.query}({self.source})"


_QUERIES = {
    "most": dsl.mostcolor,
    "least": dsl.leastcolor,
}


def resolve(color, state) -> int:
    """Resolve a COLOR parameter to a concrete 0-9 value. A plain int is
    itself; a `DerivedColor` reads its query against the named slot's grid."""
    if isinstance(color, DerivedColor):
        grid: Grid = state.get(color.source)
        if grid is None:
            grid = state.grid
        return _QUERIES[color.query](grid)
    return int(color)


# The COLOR domain the typed enumerator draws from: every literal plus the
# grid-derived queries. Kept small so search stays tractable.
COLOR_DOMAIN = list(range(10)) + [DerivedColor("most"), DerivedColor("least")]
