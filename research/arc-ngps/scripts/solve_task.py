from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch

from arc_ngps.data.arc_dataset import load_arc_task
from arc_ngps.data.tokenizer import grid_to_tokens, pad_tokenized_grids
from arc_ngps.models.ngps_model import NGPSModel, NGPSConfig
from arc_ngps.search.beam import beam_search
from arc_ngps.search.verify import verify_program
from arc_ngps.dsl.parse import program_from_json

# NOTE: This script is a placeholder. It shows where the decode->execute->verify loop plugs in.
# You need: (1) a token vocabulary + (2) token->AST parser + (3) grammar constraints.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_json", type=str)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    task = load_arc_task(Path(args.task_json))
    device = torch.device(args.device)

    model = NGPSModel(NGPSConfig()).to(device)
    model.eval()

    # Prepare training pairs token tensors (variable length padding)
    P = len(task.train)
    in_toks = [grid_to_tokens(p.inp) for p in task.train]
    out_toks = [grid_to_tokens(p.out) for p in task.train]

    in_pad = pad_tokenized_grids(in_toks)
    out_pad = pad_tokenized_grids(out_toks)

    # Add batch dimension and pair dimension: [B=1,P,N]
    in_tokens = in_pad["tokens"][None, :, :]
    in_pos = in_pad["pos"][None, :, :, :]
    in_mask = in_pad["mask"][None, :, :]

    out_tokens = out_pad["tokens"][None, :, :]
    out_pos = out_pad["pos"][None, :, :, :]
    out_mask = out_pad["mask"][None, :, :]

    # Collapse: currently model expects fixed P dimension; convert to [1,P,N] already.
    # Decode tokens (stub)
    bos, eos = 1, 2

    hyp = model.task_hypothesis(in_tokens, in_pos, in_mask, out_tokens, out_pos, out_mask)  # [1,D]

    def next_logits_fn(prefix: torch.LongTensor) -> torch.Tensor:
        return model.decoder.next_logits(prefix.to(device), hyp.to(device))

    beams = beam_search(next_logits_fn, bos=bos, eos=eos, beam_size=8, max_len=64)
    print(f"Got {len(beams)} candidate token sequences (unconstrained).")

    # Token->AST is not implemented; for now demonstrate verification with a hard-coded toy program.
    # Replace this with parsing decoded tokens into DSL AST.
    toy = {
        "type": "Program",
        "expr": {
            "type": "Paint",
            "grid": {"type": "VarGrid"},
            "objs": {"type": "Translate",
                     "objs": {"type": "SelectColor", "grid": {"type": "VarGrid"}, "color": {"type": "ConstColor", "c": 1}},
                     "dy": 0, "dx": 1},
            "color": {"type": "ConstColor", "c": 1}
        }
    }
    prog = program_from_json(toy)

    train_pairs = [(p.inp, p.out) for p in task.train]
    vr = verify_program(prog, train_pairs)
    print("Toy program verify:", vr.ok, vr.errors[:3])

    if task.test:
        pred = __import__("arc_ngps.executor.runtime", fromlist=["run_program"]).run_program(prog, task.test[0])
        print("Predicted test output shape:", pred.shape)


if __name__ == "__main__":
    main()
