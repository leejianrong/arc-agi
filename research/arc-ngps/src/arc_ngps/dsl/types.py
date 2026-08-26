from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, Union


class Ty(Enum):
    GRID = auto()
    OBJSET = auto()
    OBJ = auto()
    INT = auto()
    COLOR = auto()
    BOOL = auto()
    COORD = auto()


@dataclass(frozen=True)
class TypedValue:
    ty: Ty
    value: Any
