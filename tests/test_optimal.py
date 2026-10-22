import numpy as np
import pytest

from ultrarl.config import EnvConfig, EvalSetting
from ultrarl.env import ChainEnv
from ultrarl.optimal import OptimalPlanner, exhaustive_success, fault_prior, initial_slack, optimal_success
from ultrarl.policies import heuristic_act
from ultrarl.rollout import evaluate


def test_fault_prior_matches_the_environment():
    cfg = EnvConfig()
    env = ChainEnv(cfg, seed=0)
    counts = np.zeros(8)
    n = 20000
    for i in range(n):
        env.reset(seed=[9, i], depth_range=(7, 7))
        if env.fault_index is not None:
            counts[env.fault_index] += 1
    prior = fault_prior(cfg, 7)
    assert prior.sum() == pytest.approx(cfg.p_fault)
    assert np.abs(counts[:7] / n - prior).max() < 0.01


def test_value_is_one_when_every_step_can_be_checked_and_repaired():
    cfg = EnvConfig()  # slack 6: four checks plus one undo-and-redo
    assert OptimalPlanner(cfg, 4).value(0, initial_slack(cfg, 4)) == pytest.approx(1.0)
    assert OptimalPlanner(cfg, 5).value(0, initial_slack(cfg, 5)) < 1.0


def test_value_without_faults_or_without_budget():
    assert optimal_success(EnvConfig(p_fault=0.0), (3, 14)) == pytest.approx(1.0)
    cfg = EnvConfig(budget_extra=0)  # no check is affordable: success iff no fault
    assert optimal_success(cfg, (6, 6)) == pytest.approx(1.0 - cfg.p_fault)


CONFIGS = [EnvConfig(), EnvConfig(verify_cost=2, budget_extra=5), EnvConfig(fault_profile="uniform")]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_policy_achieves_the_planned_value(cfg):
    setting = EvalSetting("t", (8, 8))
    m = evaluate(heuristic_act("optimal", 0, cfg), cfg, setting, run_seed=0, episodes=4000)
    planned = optimal_success(cfg, setting.depth_range)
    assert abs(m["success"] - planned) < 4 * np.sqrt(planned * (1 - planned) / 4000)


def test_optimal_beats_the_heuristics():
    cfg, setting = EnvConfig(), EvalSetting("t", (7, 14))
    planned = optimal_success(cfg, setting.depth_range)
    for name in ("cue_triggered", "budget_aware_1", "verify_every_step"):
        assert evaluate(heuristic_act(name, 0), cfg, setting, 0, 2000)["success"] < planned


TIGHT = [
    EnvConfig(budget_extra=2),
    EnvConfig(budget_extra=3, verify_cost=2),
    EnvConfig(budget_extra=4, cue_hit_rate=0.6, cue_false_positive_rate=0.35),
    EnvConfig(budget_extra=3, fault_profile="uniform"),
]


@pytest.mark.parametrize("cfg", TIGHT)
@pytest.mark.parametrize("depth", [4, 5])
def test_no_back_checking_restriction_does_not_bind_on_short_chains(cfg, depth):
    # Exhaustive search over every history-dependent policy, including undoing to check earlier steps.
    planned = OptimalPlanner(cfg, depth).value(0, initial_slack(cfg, depth))
    assert exhaustive_success(cfg, depth) == pytest.approx(planned, abs=1e-12)
    assert planned < 1.0


def test_exhaustive_search_can_beat_the_restricted_planner():
    # Positive control: with expensive checks and plenty of slack, undoing to check an
    # earlier step pays, so the exhaustive search must find more than the planner.
    cfg = EnvConfig(p_fault=0.95, verify_cost=3, revise_cost=1, budget_extra=7)
    planned = OptimalPlanner(cfg, 3).value(0, initial_slack(cfg, 3))
    assert exhaustive_success(cfg, 3) > planned + 0.05
