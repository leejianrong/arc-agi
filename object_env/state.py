"""`ObjState` — the named typed slots the grammar's dataflow types live in.

Decision (V5 fork 1, user-approved): **named typed slots**, not an SSA register
file. A bounded, fixed set of slots mirrors the shipped dual-slot mental model
(ADR-0020) and gives construction-time type-checking for free — a program step
that references an unfilled slot is rejected before it ever runs.

Each slot has one fixed `ArgType`; `SLOT_TYPES` is the single source of truth
the grammar's symbolic type-env walk reads. A slot is "filled" when its value is
not `None` (the `grid` slot is always filled). Updates are functional
(`dataclasses.replace`), so a program never mutates a state in place.
"""

from dataclasses import dataclass, replace

from object_env.types import ArgType, Grid, Indices, Obj

# slot name -> the single dataflow type it holds. The grammar type-checks
# against this table; `state.py` and `actions.py` must agree with it.
SLOT_TYPES: dict[str, ArgType] = {
    "grid": ArgType.GRID,
    "region_a": ArgType.REGION,
    "region_b": ArgType.REGION,
    "set_a": ArgType.INDEXSET,
    "set_b": ArgType.INDEXSET,
    "obj": ArgType.OBJECT,
}


@dataclass(frozen=True)
class ObjState:
    grid: Grid
    region_a: Grid | None = None
    region_b: Grid | None = None
    set_a: Indices | None = None
    set_b: Indices | None = None
    obj: Obj | None = None

    def filled(self) -> frozenset:
        """The set of slot names currently holding a value — the *runtime*
        analogue of the grammar's symbolic type-env."""
        names = {"grid"}
        for name in ("region_a", "region_b", "set_a", "set_b", "obj"):
            if getattr(self, name) is not None:
                names.add(name)
        return frozenset(names)

    def get(self, slot: str):
        return getattr(self, slot)

    def set(self, **updates) -> "ObjState":
        return replace(self, **updates)
