import math
from types import SimpleNamespace

import pytest

from ultrarl.config import EnvConfig
from ultrarl.env import Action, ChainEnv
from ultrarl.rewards import StepEvent, compute_reward

T, V, R, C = Action.THINK, Action.VERIFY, Action.REVISE, Action.COMMIT


def rewards_for(cfg, actions, depth=3, fault=1, cues=None):
    env = ChainEnv(cfg)
    env.reset(seed=0, depth_range=(depth, depth), fault_index=fault)
    if cues is not None:
        env.cues = cues
    return [env.step(a)[1] for a in actions]


def test_outcome_only_is_terminal_plus_cost():
    cfg = EnvConfig(reward_scheme="outcome_only")
    rs = rewards_for(cfg, [T, V, T, V, R, T, T, C])
    assert all(math.isclose(r, -cfg.step_penalty) for r in rs[:-1])
    assert math.isclose(rs[-1], cfg.correct_reward)


def test_commit_failure_reward():
    cfg = EnvConfig(reward_scheme="outcome_only", budget_penalty=False)
    assert rewards_for(cfg, [T, T, T, C])[-1] == cfg.incorrect_reward
    assert rewards_for(cfg, [C])[-1] == cfg.incorrect_reward  # committing early fails


def test_budget_penalty_switch():
    cfg = EnvConfig(reward_scheme="outcome_only", budget_penalty=False)
    assert rewards_for(cfg, [T])[0] == 0.0


def test_naive_process_rewards_hidden_progress():
    cfg = EnvConfig(reward_scheme="naive_process", budget_penalty=False)
    # transition 1 is faulty: the second THINK lands on a wrong value and earns nothing
    rs = rewards_for(cfg, [T, T, V])
    assert rs[0] == pytest.approx(cfg.naive_progress_bonus)
    assert rs[1] == 0.0
    assert rs[2] == pytest.approx(cfg.naive_verify_bonus)  # catching the fault


def test_monitorability_terms():
    cfg = EnvConfig(reward_scheme="monitorability_aware", budget_penalty=False)
    d = 3
    gap = cfg.ma_gap_penalty / d
    # cues: step 1 no cue, step 2 cue (the faulty transition 1)
    rs = rewards_for(cfg, [T, V, T, V, V], depth=d, fault=1, cues=[0, 0, 1, 0])
    assert rs[0] == pytest.approx(-gap)  # unchecked gap grows 0 -> 1
    # safe check of an un-cued step: safe bonus - low-cue penalty, gap closes 1 -> 0
    assert rs[1] == pytest.approx(cfg.ma_safe_verify_bonus - cfg.ma_low_cue_penalty + gap)
    assert rs[2] == pytest.approx(-gap)
    # fault catch on a cued step: fault bonus + cue bonus, gap closes
    assert rs[3] == pytest.approx(cfg.ma_fault_verify_bonus + cfg.ma_cue_verify_bonus + gap)
    # repeat verification is wasted
    assert rs[4] == pytest.approx(-cfg.ma_waste_verify_penalty)


def test_fault_catch_on_uncued_step_has_no_low_cue_penalty():
    cfg = EnvConfig(reward_scheme="monitorability_aware", budget_penalty=False)
    rs = rewards_for(cfg, [T, T, V], depth=3, fault=1, cues=[0, 0, 0, 0])
    assert rs[2] == pytest.approx(cfg.ma_fault_verify_bonus + 2 * cfg.ma_gap_penalty / 3)


def test_verifying_downstream_of_hidden_fault_is_wasted():
    cfg = EnvConfig(reward_scheme="monitorability_aware", budget_penalty=False)
    rs = rewards_for(cfg, [T, T, V], depth=3, fault=0, cues=[0, 0, 1, 0])
    assert rs[2] == pytest.approx(-cfg.ma_waste_verify_penalty + 2 * cfg.ma_gap_penalty / 3)


def test_gap_term_telescopes_to_terminal_gap():
    cfg = EnvConfig(reward_scheme="monitorability_aware", budget_penalty=False, correct_reward=0.0)
    env = ChainEnv(cfg)
    env.reset(seed=0, depth_range=(5, 5), fault_index=None)
    env.cues = [0] * 6
    total = 0.0
    for a in [T, T, V, T, R, T, T, T, C]:
        total += env.step(a)[1]
    verify_terms = cfg.ma_safe_verify_bonus - cfg.ma_low_cue_penalty
    assert total == pytest.approx(verify_terms - cfg.ma_gap_penalty * 3 / 5)


def test_unknown_scheme_is_rejected_at_config_and_reward_level():
    with pytest.raises(ValueError):
        EnvConfig(reward_scheme="nope")
    # compute_reward guards against configs that bypass EnvConfig validation
    cfg = SimpleNamespace(**{**vars(EnvConfig()), "reward_scheme": "nope"})
    with pytest.raises(ValueError, match="unknown reward scheme"):
        compute_reward(cfg, StepEvent(action="THINK", depth=3, cost=1))


def test_fault_bonus_on_repair_moves_the_bonus_from_detection_to_repair():
    base = dict(reward_scheme="monitorability_aware", budget_penalty=False, ma_gap_penalty=0.0)
    cues = [0, 0, 1, 0]
    on_detect = rewards_for(EnvConfig(**base), [T, T, V, R, T], depth=3, fault=1, cues=cues)
    on_repair = rewards_for(
        EnvConfig(**base, ma_fault_bonus_on_repair=True), [T, T, V, R, T], depth=3, fault=1, cues=cues
    )
    cfg = EnvConfig(**base)
    assert on_detect[2] == pytest.approx(cfg.ma_fault_verify_bonus + cfg.ma_cue_verify_bonus)
    assert on_repair[2] == pytest.approx(cfg.ma_cue_verify_bonus)  # detection alone: cue bonus only
    assert on_detect[4] == pytest.approx(0.0)
    assert on_repair[4] == pytest.approx(cfg.ma_fault_verify_bonus)  # paid when the fault is re-executed correctly


def test_gap_refund_makes_downstream_check_net_positive():
    # Documents a property of the original reward discussed in the paper: the gap
    # refund outweighs the wasted-check penalty for a check downstream of a hidden fault.
    cfg = EnvConfig(reward_scheme="monitorability_aware")
    rs = rewards_for(cfg, [T, T, V], depth=3, fault=0, cues=[0, 0, 0, 0])
    assert rs[2] > 0
