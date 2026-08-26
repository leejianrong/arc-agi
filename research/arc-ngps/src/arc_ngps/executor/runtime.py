from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
import numpy as np

from arc_ngps.dsl.types import Ty, TypedValue
from arc_ngps.dsl import ast as A
from .grid_ops import select_color_objs, translate_objs, paint


class ExecError(RuntimeError):
    pass


def eval_expr(expr: Any, grid: np.ndarray) -> TypedValue:
    # Minimal evaluator; extend with more ops + safety checks.
    if isinstance(expr, A.VarGrid):
        return TypedValue(Ty.GRID, grid)
    if isinstance(expr, A.ConstColor):
        return TypedValue(Ty.COLOR, int(expr.c))
    if isinstance(expr, A.SelectColor):
        g = eval_expr(expr.grid, grid)
        c = eval_expr(expr.color, grid)
        if g.ty != Ty.GRID or c.ty != Ty.COLOR:
            raise ExecError("Type error in SelectColor")
        objs = select_color_objs(g.value, c.value)
        return TypedValue(Ty.OBJSET, objs)
    if isinstance(expr, A.Translate):
        os = eval_expr(expr.objs, grid)
        if os.ty != Ty.OBJSET:
            raise ExecError("Type error in Translate")
        return TypedValue(Ty.OBJSET, translate_objs(os.value, expr.dy, expr.dx))
    if isinstance(expr, A.Paint):
        g = eval_expr(expr.grid, grid)
        os = eval_expr(expr.objs, grid)
        c = eval_expr(expr.color, grid)
        if g.ty != Ty.GRID or os.ty != Ty.OBJSET or c.ty != Ty.COLOR:
            raise ExecError("Type error in Paint")
        return TypedValue(Ty.GRID, paint(g.value, os.value, c.value))
    if isinstance(expr, A.Compose):
        # NOTE: this is just a placeholder: composition typically needs function-typed terms.
        # For now assume f and g are Grid->Grid expressions inlined against VarGrid.
        mid = eval_expr(expr.f, grid)
        if mid.ty != Ty.GRID:
            raise ExecError("Type error in Compose.f")
        out = eval_expr(expr.g, mid.value)
        if out.ty != Ty.GRID:
            raise ExecError("Type error in Compose.g")
        return out

    raise ExecError(f"Unsupported expr node: {type(expr)}")


def run_program(prog: A.Program, grid: np.ndarray) -> np.ndarray:
    v = eval_expr(prog.expr, grid)
    if v.ty != Ty.GRID:
        raise ExecError("Program did not return a Grid")
    return v.value
