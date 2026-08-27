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
