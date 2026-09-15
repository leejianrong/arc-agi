"""object_env — the V5 object substrate + typed action grammar (ADR-0029).

A new, parallel track to the shipped `arc_env/` single-mutable-grid agent. It
must NOT be imported *by* `arc_env/`; reuse flows the other way — this package
leans on `arc_env._dsl` (the vendored arc-dsl executor) and, in the harness,
`arc_env.{task_loader, reward, re_arc}`.

The headline is the *typed, compositional action grammar* (the POC lever that
turned a 0/8 free-form search into 8/8): actions declare typed slots, only
type-valid compositions are constructible, and the search enumerates over that
structured space rather than a flat pick-any-of-N gene list.
"""

from object_env.state import ObjState
from object_env.types import ArgType, Obj

__all__ = ["ArgType", "Obj", "ObjState"]
