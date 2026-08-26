from __future__ import annotations
from typing import Any, Dict

from .ast import (
    Program, VarGrid, ConstColor, SelectColor, Paint, Translate, Compose
)

# Minimal JSON parser scaffold; extend as DSL grows.

def expr_from_json(j: Dict[str, Any]):
    t = j["type"]
    if t == "VarGrid":
        return VarGrid()
    if t == "ConstColor":
        return ConstColor(c=int(j["c"]))
    if t == "SelectColor":
        return SelectColor(grid=expr_from_json(j["grid"]), color=expr_from_json(j["color"]))
    if t == "Paint":
        return Paint(grid=expr_from_json(j["grid"]), objs=expr_from_json(j["objs"]), color=expr_from_json(j["color"]))
    if t == "Translate":
        return Translate(objs=expr_from_json(j["objs"]), dy=int(j["dy"]), dx=int(j["dx"]))
    if t == "Compose":
        return Compose(f=expr_from_json(j["f"]), g=expr_from_json(j["g"]))
    raise ValueError(f"Unknown node type: {t}")


def program_from_json(j: Dict[str, Any]) -> Program:
    if j["type"] != "Program":
        raise ValueError("Expected Program")
    return Program(expr=expr_from_json(j["expr"]))
