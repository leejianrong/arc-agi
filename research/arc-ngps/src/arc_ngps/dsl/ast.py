from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol

from .types import Ty


class Node(Protocol):
    def to_sexpr(self) -> str: ...
    def to_json(self) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class Program:
    """A program is a single expression that maps a Grid -> Grid."""
    expr: "Expr"

    def to_json(self) -> Dict[str, Any]:
        return {"type": "Program", "expr": self.expr.to_json()}

    def to_sexpr(self) -> str:
        return f"(Program {self.expr.to_sexpr()})"


class Expr(Protocol):
    out_ty: Ty
    def to_json(self) -> Dict[str, Any]: ...
    def to_sexpr(self) -> str: ...


@dataclass(frozen=True)
class VarGrid:
    """The input grid variable."""
    out_ty: Ty = Ty.GRID

    def to_json(self) -> Dict[str, Any]:
        return {"type": "VarGrid"}

    def to_sexpr(self) -> str:
        return "grid"


@dataclass(frozen=True)
class ConstColor:
    c: int  # 0..9
    out_ty: Ty = Ty.COLOR

    def to_json(self) -> Dict[str, Any]:
        return {"type": "ConstColor", "c": int(self.c)}

    def to_sexpr(self) -> str:
        return f"(color {int(self.c)})"


@dataclass(frozen=True)
class SelectColor:
    grid: Expr
    color: Expr
    out_ty: Ty = Ty.OBJSET

    def to_json(self) -> Dict[str, Any]:
        return {"type": "SelectColor", "grid": self.grid.to_json(), "color": self.color.to_json()}

    def to_sexpr(self) -> str:
        return f"(select_color {self.grid.to_sexpr()} {self.color.to_sexpr()})"


@dataclass(frozen=True)
class Paint:
    grid: Expr
    objs: Expr
    color: Expr
    out_ty: Ty = Ty.GRID

    def to_json(self) -> Dict[str, Any]:
        return {"type": "Paint", "grid": self.grid.to_json(), "objs": self.objs.to_json(), "color": self.color.to_json()}

    def to_sexpr(self) -> str:
        return f"(paint {self.grid.to_sexpr()} {self.objs.to_sexpr()} {self.color.to_sexpr()})"


@dataclass(frozen=True)
class Translate:
    objs: Expr
    dy: int
    dx: int
    out_ty: Ty = Ty.OBJSET

    def to_json(self) -> Dict[str, Any]:
        return {"type": "Translate", "objs": self.objs.to_json(), "dy": int(self.dy), "dx": int(self.dx)}

    def to_sexpr(self) -> str:
        return f"(translate {self.objs.to_sexpr()} {int(self.dy)} {int(self.dx)})"


@dataclass(frozen=True)
class Compose:
    """Compose g(f(grid)) for convenience."""
    f: Expr
    g: Expr
    out_ty: Ty = Ty.GRID

    def to_json(self) -> Dict[str, Any]:
        return {"type": "Compose", "f": self.f.to_json(), "g": self.g.to_json()}

    def to_sexpr(self) -> str:
        return f"(compose {self.f.to_sexpr()} {self.g.to_sexpr()})"
