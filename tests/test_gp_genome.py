"""Unit tests (SLICES.md V4): crossover and mutation operators always
produce a syntactically valid (type-correct) program given valid
parents/inputs."""

import random

import pytest

from arc_env import actions
from trainers.gp.genome import (
    MIN_LENGTH,
    crossover,
    is_well_formed_gene,
    is_well_formed_program,
    mutate,
    random_gene,
    random_program,
)

MAX_LENGTH = 6


def test_random_gene_is_well_formed():
    rng = random.Random(0)
    for _ in range(200):
        assert is_well_formed_gene(random_gene(rng))


def test_random_program_is_well_formed():
    rng = random.Random(0)
    for _ in range(50):
        assert is_well_formed_program(random_program(rng, MAX_LENGTH), MAX_LENGTH)


@pytest.mark.parametrize("seed", range(30))
def test_crossover_always_produces_a_well_formed_program(seed):
    rng = random.Random(seed)
    p1 = random_program(rng, MAX_LENGTH)
    p2 = random_program(rng, MAX_LENGTH)
    child = crossover(p1, p2, rng, MAX_LENGTH)
    assert is_well_formed_program(child, MAX_LENGTH)


@pytest.mark.parametrize("seed", range(30))
def test_mutate_always_produces_a_well_formed_program(seed):
    rng = random.Random(seed)
    program = random_program(rng, MAX_LENGTH)
    mutated = mutate(program, rng, mutation_rate=1.0, max_length=MAX_LENGTH)  # rate=1.0: exercise every operator
    assert is_well_formed_program(mutated, MAX_LENGTH)


def test_crossover_never_produces_an_empty_program():
    rng = random.Random(0)
    for _ in range(50):
        p1, p2 = [random_gene(rng)], [random_gene(rng)]  # both minimal length
        child = crossover(p1, p2, rng, MAX_LENGTH)
        assert len(child) >= MIN_LENGTH


def test_mutate_never_shrinks_below_min_length():
    rng = random.Random(0)
    program = [random_gene(rng)]  # already at MIN_LENGTH
    for _ in range(50):
        program = mutate(program, rng, mutation_rate=1.0, max_length=MAX_LENGTH)
        assert len(program) >= MIN_LENGTH


def test_mutate_never_exceeds_max_length():
    rng = random.Random(0)
    program = random_program(rng, MAX_LENGTH)
    for _ in range(50):
        program = mutate(program, rng, mutation_rate=1.0, max_length=MAX_LENGTH)
        assert len(program) <= MAX_LENGTH


def test_crossover_and_mutate_do_not_mutate_their_inputs():
    rng = random.Random(0)
    p1 = random_program(rng, MAX_LENGTH)
    p2 = random_program(rng, MAX_LENGTH)
    p1_copy, p2_copy = list(p1), list(p2)
    crossover(p1, p2, rng, MAX_LENGTH)
    mutate(p1, rng, mutation_rate=1.0, max_length=MAX_LENGTH)
    assert p1 == p1_copy
    assert p2 == p2_copy


def test_is_well_formed_gene_rejects_out_of_range_primitive():
    assert not is_well_formed_gene((len(actions.ACTIONS), (0,) * actions.MAX_ARITY))
    assert not is_well_formed_gene((-1, (0,) * actions.MAX_ARITY))


def test_is_well_formed_gene_rejects_out_of_range_arg():
    assert not is_well_formed_gene((0, (actions.RAW_ARG_RANGE,) * actions.MAX_ARITY))
