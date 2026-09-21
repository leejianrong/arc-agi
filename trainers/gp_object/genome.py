"""GP genome representation and operators over the object grammar (SLICES.md
V6): each gene selects one entry from the *current* type-legal step menu
(`object_env.grammar.legal_steps`), not an unconstrained `(action, args)`
pair - so a genome is "constrained by the grammar's types/positions", not
the flat pick-any-of-N list SLICES.md rules out (the shape the F15 POC's
`gp_object.py` already tried, and the arm that scored 0/8).

A genome is a flat `list[int]` - as simple as `trainers/gp/genome.py`'s flat
gene list, but each int is a position-relative index, not a global
`(primitive_index, raw_args)` tuple: `to_program` walks the genome left to
right, and at each position resolves `gene % len(menu)` against
`object_env.grammar.legal_steps(filled, ACTIONS)`, the menu of steps legal
given everything filled so far. Because every entry in that menu already
satisfies the grammar's read-before-write rule, every step `to_program`
appends is legal by construction - `Program(steps)`'s own construction-time
type-check (`object_env.grammar.type_check`) can never fail on its output.
That is the concrete, provable answer to this module's named unit-test
requirement: crossover and mutation only ever produce grammar-valid
programs, because there is no path from a well-formed genome to an
ill-typed `Program` at all.

The menu is never empty: `"grid"` is always filled (nothing clears it), and
several actions (`select_largest`, `trim`, `replace_color`,
`canvas_mostcolor`, `split`, ...) need only `"grid"`. So `to_program` always
consumes every gene, and genome length equals program length - `MIN_LENGTH`/
`max_length` map directly onto program depth, the same mental model
`trainers/gp/genome.py` already uses.

"Well-formed" for a raw genome is the same loose bar flat GP's genes use:
length within bounds, each gene a non-negative int under a fixed range -
not "every gene reaches a specific step" (that's `to_program`'s job, and it
always succeeds).
"""

import random

from object_env.actions import ACTIONS
from object_env.grammar import Program, legal_steps

Gene = int
Genome = list[Gene]

MIN_LENGTH = 1
GENE_RANGE = 1000  # comfortably larger than any legal_steps menu size


def random_gene(rng: random.Random) -> Gene:
    return rng.randrange(GENE_RANGE)


def random_genome(rng: random.Random, max_length: int) -> Genome:
    length = rng.randint(MIN_LENGTH, max(MIN_LENGTH, max_length))
    return [random_gene(rng) for _ in range(length)]


def is_well_formed_gene(gene: Gene) -> bool:
    return isinstance(gene, int) and 0 <= gene < GENE_RANGE


def is_well_formed_genome(genome: Genome, max_length: int) -> bool:
    return MIN_LENGTH <= len(genome) <= max_length and all(is_well_formed_gene(g) for g in genome)


def crossover(parent1: Genome, parent2: Genome, rng: random.Random, max_length: int) -> Genome:
    """Single-point splice: `parent1`'s genes up to a random cut, then
    `parent2`'s genes from a random cut, capped to `max_length` and never
    empty (falls back to a single gene from whichever parent has one)."""

    cut1 = rng.randint(0, len(parent1))
    cut2 = rng.randint(0, len(parent2))
    child = parent1[:cut1] + parent2[cut2:]
    if not child:
        child = [rng.choice(parent1 or parent2)]
    return child[:max_length]


def mutate(genome: Genome, rng: random.Random, mutation_rate: float, max_length: int) -> Genome:
    """Applies, independently and with probability `mutation_rate` each:
    resampling a random gene, inserting a fresh random gene, and deleting a
    random gene (never below `MIN_LENGTH`) - the same three whole-gene
    operators `trainers/gp/genome.py` uses, minus its extra "resample one
    argument" operator, which doesn't apply here (a gene has no sub-args to
    target - see module docstring)."""

    genome = list(genome)

    if rng.random() < mutation_rate and genome:
        i = rng.randrange(len(genome))
        genome[i] = random_gene(rng)

    if rng.random() < mutation_rate and len(genome) < max_length:
        i = rng.randint(0, len(genome))
        genome.insert(i, random_gene(rng))

    if rng.random() < mutation_rate and len(genome) > MIN_LENGTH:
        i = rng.randrange(len(genome))
        del genome[i]

    return genome


def to_program(genome: Genome, actions=ACTIONS) -> Program:
    """Deterministically decode a genome into a grammar-valid `Program` by
    walking its genes left to right and, at each position, indexing into
    the current legal-step menu (see module docstring). Never raises."""

    filled = frozenset({"grid"})
    steps = []
    for gene in genome:
        menu = legal_steps(filled, actions)
        step = menu[gene % len(menu)]
        steps.append(step)
        filled = (filled | set(step.action.writes(step.args))) - set(step.action.clears(step.args))
    return Program(steps)
