"""Unit tests (SLICES.md V6): crossover and mutation operators over the
object grammar always produce a well-formed genome, and - the slice's named
requirement - decoding any such genome into a `Program`
(`trainers.gp_object.genome.to_program`) always yields a grammar-valid
program (never raises `object_env.grammar.GrammarError`, and satisfies
`type_check`)."""

import random

import pytest

from object_env.grammar import type_check
from trainers.gp_object.genome import (
    GENE_RANGE,
    MIN_LENGTH,
    crossover,
    is_well_formed_gene,
    is_well_formed_genome,
    mutate,
    random_gene,
    random_genome,
    to_program,
)

MAX_LENGTH = 6


def test_random_gene_is_well_formed():
    rng = random.Random(0)
    for _ in range(200):
        assert is_well_formed_gene(random_gene(rng))


def test_random_genome_is_well_formed():
    rng = random.Random(0)
    for _ in range(50):
        assert is_well_formed_genome(random_genome(rng, MAX_LENGTH), MAX_LENGTH)


@pytest.mark.parametrize("seed", range(30))
def test_crossover_always_produces_a_well_formed_genome(seed):
    rng = random.Random(seed)
    g1 = random_genome(rng, MAX_LENGTH)
    g2 = random_genome(rng, MAX_LENGTH)
    child = crossover(g1, g2, rng, MAX_LENGTH)
    assert is_well_formed_genome(child, MAX_LENGTH)


@pytest.mark.parametrize("seed", range(30))
def test_mutate_always_produces_a_well_formed_genome(seed):
    rng = random.Random(seed)
    genome = random_genome(rng, MAX_LENGTH)
    mutated = mutate(genome, rng, mutation_rate=1.0, max_length=MAX_LENGTH)  # rate=1.0: exercise every operator
    assert is_well_formed_genome(mutated, MAX_LENGTH)


def test_crossover_never_produces_an_empty_genome():
    rng = random.Random(0)
    for _ in range(50):
        g1, g2 = [random_gene(rng)], [random_gene(rng)]  # both minimal length
        child = crossover(g1, g2, rng, MAX_LENGTH)
        assert len(child) >= MIN_LENGTH


def test_mutate_never_shrinks_below_min_length():
    rng = random.Random(0)
    genome = [random_gene(rng)]  # already at MIN_LENGTH
    for _ in range(50):
        genome = mutate(genome, rng, mutation_rate=1.0, max_length=MAX_LENGTH)
        assert len(genome) >= MIN_LENGTH


def test_mutate_never_exceeds_max_length():
    rng = random.Random(0)
    genome = random_genome(rng, MAX_LENGTH)
    for _ in range(50):
        genome = mutate(genome, rng, mutation_rate=1.0, max_length=MAX_LENGTH)
        assert len(genome) <= MAX_LENGTH


def test_crossover_and_mutate_do_not_mutate_their_inputs():
    rng = random.Random(0)
    g1 = random_genome(rng, MAX_LENGTH)
    g2 = random_genome(rng, MAX_LENGTH)
    g1_copy, g2_copy = list(g1), list(g2)
    crossover(g1, g2, rng, MAX_LENGTH)
    mutate(g1, rng, mutation_rate=1.0, max_length=MAX_LENGTH)
    assert g1 == g1_copy
    assert g2 == g2_copy


def test_is_well_formed_gene_rejects_out_of_range():
    assert not is_well_formed_gene(GENE_RANGE)
    assert not is_well_formed_gene(-1)


# The named V6 requirement: crossover/mutation only ever produce
# grammar-valid programs. `to_program` can only ever append steps drawn
# from `object_env.grammar.legal_steps`, so this is provable by
# construction - these tests are the regression guard against that
# invariant quietly breaking under a future refactor.


@pytest.mark.parametrize("seed", range(30))
def test_random_genome_always_decodes_to_a_grammar_valid_program(seed):
    rng = random.Random(seed)
    genome = random_genome(rng, MAX_LENGTH)
    program = to_program(genome)
    type_check(program.steps)  # raises GrammarError on any violation
    assert len(program.steps) == len(genome)  # the menu is never empty ("grid" always filled)


@pytest.mark.parametrize("seed", range(30))
def test_crossover_output_always_decodes_to_a_grammar_valid_program(seed):
    rng = random.Random(seed)
    g1 = random_genome(rng, MAX_LENGTH)
    g2 = random_genome(rng, MAX_LENGTH)
    child = crossover(g1, g2, rng, MAX_LENGTH)
    type_check(to_program(child).steps)


@pytest.mark.parametrize("seed", range(30))
def test_mutate_output_always_decodes_to_a_grammar_valid_program(seed):
    rng = random.Random(seed)
    genome = random_genome(rng, MAX_LENGTH)
    mutated = mutate(genome, rng, mutation_rate=1.0, max_length=MAX_LENGTH)
    type_check(to_program(mutated).steps)
