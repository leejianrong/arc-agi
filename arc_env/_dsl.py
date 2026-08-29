"""Import shim for the vendored, flat-file `third_party/arc-dsl/` package.

`dsl.py` there does `from arc_types import *` as a bare top-level import
(matching its upstream layout), so it must be imported with that directory on
`sys.path` rather than as a normal dotted submodule. Every `arc_env` module
that needs the DSL imports it from here instead of repeating the path hack.
"""

import sys
from pathlib import Path

_ARC_DSL_DIR = Path(__file__).resolve().parent.parent / "third_party" / "arc-dsl"
if str(_ARC_DSL_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_DSL_DIR))

import constants
import dsl

__all__ = ["constants", "dsl"]
