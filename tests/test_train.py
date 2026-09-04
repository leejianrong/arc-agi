"""Unit tests for `train.py`'s standalone helpers (not the slow end-to-end
training loops - see `tests/test_train_ppo.py`/`test_train_gp.py` for those).

Regression test for a real crash: a full `make train` run against 67a3c6ac
hit `GenerationError` from `arc_env.re_arc.generate_pair` after 200 PPO
updates' worth of rollouts - a narrow difficulty band can make every re-arc
attempt land on a degenerate (input == output) instance for this task's
generator, and `MAX_ATTEMPTS` retries within `generate_pair` isn't infinite.
`make_next_pair_fn`'s `next_pair` must fall back to a native train pair
instead of propagating that, or a long enough run always eventually dies.
"""

import random

import train
from arc_env.re_arc import GenerationError

TASK_ID = "67a3c6ac"


def test_next_pair_falls_back_to_a_native_pair_when_re_arc_generation_fails(monkeypatch):
    def always_fails(*args, **kwargs):
        raise GenerationError("simulated exhaustion")

    monkeypatch.setattr(train, "generate_pair", always_fails)

    next_pair = train.make_next_pair_fn(TASK_ID, re_arc_prob=1.0, rng=random.Random(0))
    got_task_id, pair = next_pair()

    assert got_task_id == TASK_ID
    assert pair.input != pair.output  # a real curated train pair, not a degenerate one


# KAN-1177: PPO runs were seen reaching a real success rate at some
# intermediate checkpoint, then losing it by the final one - `is_new_best_eval`
# is the comparison behind `checkpoints/best.pt`, a safety net that keeps the
# best-observed eval checkpoint around instead of trusting only the last one.


def test_is_new_best_eval_true_when_no_eval_has_run_yet():
    assert train.is_new_best_eval(eval_reward=-5.0, best_eval_reward=None) is True


def test_is_new_best_eval_true_when_reward_strictly_improves():
    assert train.is_new_best_eval(eval_reward=1.0, best_eval_reward=0.5) is True


def test_is_new_best_eval_false_when_reward_is_equal_or_worse():
    assert train.is_new_best_eval(eval_reward=0.5, best_eval_reward=0.5) is False
    assert train.is_new_best_eval(eval_reward=0.2, best_eval_reward=0.5) is False
