"""GP genome representation and operators (ADR-0003, SLICES.md V4).

A "program" is a list of genes, `(primitive_index, raw_args)`, applied to a
grid in order via `arc_env.actions.execute` - the same `(primitive, args)`
shape the env's action `Dict` space and `arc_env.task_loader`'s solver-call
tables already use. Not a tree/AST: `arc_env.actions` deliberately excludes
higher-order primitives (ADR-0001), so there is no composition to
represent - every curated action already maps one grid directly to the
next, which is exactly what a flat, ordered gene list captures. "Population
of DSL-program ASTs" (SLICES.md) is this, at the same granularity PPO's
per-step action already uses; nothing here builds a separate AST layer,
matching ADR-0001's "no separate AST or executor layer of our own."

"Syntactically valid" for a gene/program (the crossover/mutation unit-test
requirement) means "a well-formed action Dict" - `primitive_index` in
range and each raw arg in `[0, RAW_ARG_RANGE)` - not "every gene executes
successfully against some particular grid" (that's state-dependent, and is
what `arc_env.actions.execute`'s own validity check already handles at
evaluation time, independently of genome operators).
"""

import random

from arc_env import actions

Gene = tuple[int, tuple[int, ...]]  # (primitive_index, raw_args)
Program = list[Gene]

MIN_LENGTH = 1


def random_gene(rng: random.Random) -> Gene:
    primitive_index = rng.randrange(len(actions.ACTIONS))
    raw_args = tuple(rng.randrange(actions.RAW_ARG_RANGE) for _ in range(actions.MAX_ARITY))
    return (primitive_index, raw_args)


def random_program(rng: random.Random, max_length: int) -> Program:
    length = rng.randint(MIN_LENGTH, max(MIN_LENGTH, max_length))
    return [random_gene(rng) for _ in range(length)]


def is_well_formed_gene(gene: Gene) -> bool:
    primitive_index, raw_args = gene
    if not (0 <= primitive_index < len(actions.ACTIONS)):
        return False
    if len(raw_args) != actions.MAX_ARITY:
        return False
    return all(0 <= a < actions.RAW_ARG_RANGE for a in raw_args)


def is_well_formed_program(program: Program, max_length: int) -> bool:
    return MIN_LENGTH <= len(program) <= max_length and all(is_well_formed_gene(g) for g in program)


def crossover(parent1: Program, parent2: Program, rng: random.Random, max_length: int) -> Program:
    """Single-point splice: `parent1`'s genes up to a random cut, then
    `parent2`'s genes from a random cut, capped to `max_length` and never
    empty (falls back to a single gene from whichever parent has one)."""

    cut1 = rng.randint(0, len(parent1))
    cut2 = rng.randint(0, len(parent2))
    child = parent1[:cut1] + parent2[cut2:]
    if not child:
        child = [rng.choice(parent1 or parent2)]
    return child[:max_length]


def mutate(program: Program, rng: random.Random, mutation_rate: float, max_length: int) -> Program:
    """Applies, independently and with probability `mutation_rate` each:
    resampling a single argument of a random gene (a finer-grained
    perturbation than replacing the whole gene - lets multi-argument
    actions like `commit`/`canvas` be hill-climbed toward one at a time
    instead of needing to land on all of them simultaneously by chance),
    point-mutating a whole random gene, inserting a fresh random gene, and
    deleting a random gene (never below `MIN_LENGTH`)."""

    program = list(program)

    if rng.random() < mutation_rate and program:
        i = rng.randrange(len(program))
        primitive_index, raw_args = program[i]
        arg_idx = rng.randrange(len(raw_args))
        new_raw_args = list(raw_args)
        new_raw_args[arg_idx] = rng.randrange(actions.RAW_ARG_RANGE)
        program[i] = (primitive_index, tuple(new_raw_args))

    if rng.random() < mutation_rate and program:
        i = rng.randrange(len(program))
        program[i] = random_gene(rng)

    if rng.random() < mutation_rate and len(program) < max_length:
        i = rng.randint(0, len(program))
        program.insert(i, random_gene(rng))

    if rng.random() < mutation_rate and len(program) > MIN_LENGTH:
        i = rng.randrange(len(program))
        del program[i]

    return program
