import numpy as np
from arc_ngps.dsl.parse import program_from_json
from arc_ngps.executor.runtime import run_program

def test_paint_translate_smoke():
    grid = np.array([
        [0,1,0],
        [0,1,0],
        [0,0,0],
    ], dtype=np.int64)

    prog_j = {
        "type":"Program",
        "expr":{
            "type":"Paint",
            "grid":{"type":"VarGrid"},
            "objs":{"type":"Translate","objs":{"type":"SelectColor","grid":{"type":"VarGrid"},"color":{"type":"ConstColor","c":1}},"dy":0,"dx":1},
            "color":{"type":"ConstColor","c":1}
        }
    }
    prog = program_from_json(prog_j)
    out = run_program(prog, grid)
    assert out.shape == grid.shape
    # shifted paint should add 1s at x+1 (clipped)
    assert out[0,2] == 1
