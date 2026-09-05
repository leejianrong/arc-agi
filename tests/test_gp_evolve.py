"""End-to-end test (SLICES.md V4): GP run against a fixture task with a
known short solving program (vmirror, PLAN.md's own named example) finds a
matching program within a small, fixed generation budget."""

from arc_env.task_loader import load_task
from trainers.gp.evolve import GPConfig, run_gp
from trainers.gp.fitness import evaluate_fitness


def test_gp_finds_a_perfect_program_for_the_vmirror_fixture_task():
    task = load_task("67a3c6ac")  # solved by a single vmirror
    config = GPConfig(population_size=50, n_generations=30, max_program_length=4, seed=0)

    result = run_gp(task, config)

    assert result.best_fitness[0] == 1.0
    assert result.n_generations_run <= config.n_generations

    # The found program genuinely solves every train AND test pair, not
    # just what evaluate_fitness (train-only) already checked.
    full_task = load_task("67a3c6ac")
    all_pairs_task = full_task.__class__(
        task_id=full_task.task_id, train=full_task.train + full_task.test, test=()
    )
    assert evaluate_fitness(result.best_program, all_pairs_task).fitness[0] == 1.0


# ADR-0014: generation-snapshot capture for replay-across-evolution.


def test_snapshots_are_taken_at_the_configured_interval():
    task = load_task("67a3c6ac")
    # A trivial (single-vmirror-gene) task can solve in generation 0, which
    # would only ever exercise the "snapshot 0" and "final" cases - use a
    # long-if-it-ran budget with a snapshot_interval unlikely to divide
    # evenly into whatever early generation it actually stops at, so the
    # "every Nth generation" branch gets covered too if it runs long enough.
    config = GPConfig(population_size=50, n_generations=30, max_program_length=4, seed=0, snapshot_interval=5)

    result = run_gp(task, config)

    generations = [g for g, _ in result.snapshots]
    assert generations[0] == 0
    assert generations[-1] == result.n_generations_run - 1  # final generation always snapshotted
    assert generations == sorted(set(generations))  # strictly increasing, no duplicates
    for g in generations[:-1]:  # every non-final snapshot lands on the configured interval
        assert g % config.snapshot_interval == 0


def test_snapshot_interval_of_one_captures_every_generation():
    task = load_task("6150a2bd")  # solved by a single rot180
    config = GPConfig(population_size=20, n_generations=10, max_program_length=3, seed=1, snapshot_interval=1)

    result = run_gp(task, config)

    generations = [g for g, _ in result.snapshots]
    assert generations == list(range(result.n_generations_run))


def test_final_snapshot_program_matches_the_best_program_found():
    # Elitism guarantees each generation's own best fitness is non-decreasing
    # (the previous top genomes carry over unchanged and re-score
    # identically), so the final snapshot's program should always be the
    # same one `run_gp` reports as `best_program`.
    task = load_task("b1948b0a")  # solved by a single replace(6, 2)
    config = GPConfig(population_size=30, n_generations=15, max_program_length=3, seed=2, snapshot_interval=4)

    result = run_gp(task, config)

    final_generation, final_program = result.snapshots[-1]
    assert final_generation == result.n_generations_run - 1
    assert final_program == result.best_program


def test_snapshots_survive_early_stop_on_perfect_fitness():
    task = load_task("67a3c6ac")  # reliably solves well before 100 generations
    config = GPConfig(population_size=100, n_generations=100, seed=0, snapshot_interval=10)

    result = run_gp(task, config)

    assert result.best_fitness[0] == 1.0
    assert result.n_generations_run < config.n_generations  # actually stopped early
    assert result.snapshots[-1][0] == result.n_generations_run - 1
    assert result.snapshots[-1][1] == result.best_program
