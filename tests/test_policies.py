import numpy as np
import pytest

from ultrarl.config import EnvConfig, EvalSetting
from ultrarl.policies import (
    budget_aware,  # noqa: F401  (import check)
    build_heuristic,
)
from ultrarl.rollout import evaluate

NAMES = ["random", "always_think", "verify_every_step", "cue_triggered", "budget_aware_0", "budget_aware_3"]


def heuristic_act(name, seed=0):
    policy = build_heuristic(name, seed)
    return lambda _obs, observable, mask: policy(observable, mask)


@pytest.mark.parametrize("name", NAMES)
def test_heuristics_respect_the_mask(name):
    # ChainEnv.step raises on masked actions, so a clean rollout proves mask compliance.
    metrics = evaluate(heuristic_act(name), EnvConfig(), EvalSetting("t", (3, 14)), run_seed=0, episodes=200)
    assert 0.0 <= metrics["success"] <= 1.0


def test_always_think_success_equals_fault_free_rate():
    m = evaluate(heuristic_act("always_think"), EnvConfig(), EvalSetting("t", (3, 6)), 0, 1000)
    assert m["success"] == pytest.approx(1.0 - m["fault_rate"])
    assert m["fault_detected"] == 0.0


def test_cue_triggered_is_perfect_with_perfect_cues():
    cfg = EnvConfig(cue_hit_rate=1.0, cue_false_positive_rate=0.0)
    m = evaluate(heuristic_act("cue_triggered"), cfg, EvalSetting("t", (3, 14)), 0, 500)
    assert m["success"] == 1.0
    assert m["verify_precision"] == 1.0


def test_verify_every_step_is_perfect_with_ample_budget():
    cfg = EnvConfig(budget_extra=20)
    m = evaluate(heuristic_act("verify_every_step"), cfg, EvalSetting("t", (3, 14)), 0, 500)
    assert m["success"] == 1.0 and m["fault_detected"] == 1.0


def test_cue_triggered_verify_rates():
    m = evaluate(heuristic_act("cue_triggered"), EnvConfig(), EvalSetting("t", (3, 6)), 0, 500)
    assert m["verify_rate_cue1"] == 1.0
    assert m["verify_rate_cue0"] == 0.0


def test_budget_aware_never_checks_when_repair_is_unaffordable():
    cfg = EnvConfig(budget_extra=2)  # slack 2 < verify + repair
    m = evaluate(heuristic_act("budget_aware_0"), cfg, EvalSetting("t", (3, 9)), 0, 200)
    assert m["verify_count"] == 0.0


def test_budget_aware_accounts_for_action_costs():
    # slack 3 < verify (2) + revise (1) + redo (1): a check could never be followed by a repair
    cfg = EnvConfig(budget_extra=3, verify_cost=2)
    m = evaluate(heuristic_act("budget_aware_0"), cfg, EvalSetting("t", (3, 9)), 0, 200)
    assert m["verify_count"] == 0.0
    m = evaluate(heuristic_act("budget_aware_0"), cfg.replace(budget_extra=4), EvalSetting("t", (3, 9)), 0, 200)
    assert m["verify_count"] > 0.0


def test_unknown_heuristic():
    with pytest.raises(ValueError):
        build_heuristic("oracle", 0)


def test_random_policy_is_seeded():
    a = evaluate(heuristic_act("random", 3), EnvConfig(), EvalSetting("t", (3, 6)), 0, 100)
    b = evaluate(heuristic_act("random", 3), EnvConfig(), EvalSetting("t", (3, 6)), 0, 100)
    assert all(np.isclose(a[k], b[k], equal_nan=True) for k in a)
