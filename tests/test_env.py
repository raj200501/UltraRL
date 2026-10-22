import numpy as np
import pytest

from ultrarl.config import EnvConfig
from ultrarl.env import Action, ChainEnv, VerifyStatus

T, V, R, C = Action.THINK, Action.VERIFY, Action.REVISE, Action.COMMIT


def play(env: ChainEnv, actions):
    info = {}
    for a in actions:
        _, _, done, info = env.step(a)
    return info


def test_fault_free_chain_succeeds_without_checking(env):
    env.reset(seed=0, depth_range=(4, 4), fault_index=None)
    info = play(env, [T, T, T, T, C])
    assert info["metrics"]["success"] == 1.0
    assert np.isnan(info["metrics"]["fault_detected"])


def test_unchecked_fault_always_fails(env):
    env.reset(seed=0, depth_range=(4, 4), fault_index=1)
    info = play(env, [T, T, T, T, C])
    assert info["metrics"]["success"] == 0.0
    assert info["metrics"]["fault_detected"] == 0.0


def test_detect_revise_redo_repairs_the_fault(env):
    env.reset(seed=0, depth_range=(3, 3), fault_index=1)
    play(env, [T, V])
    assert env.last_verify == VerifyStatus.SAFE
    play(env, [T, V])
    assert env.last_verify == VerifyStatus.UNSAFE and env.fault_located
    info = play(env, [R, T, T, C])
    assert info["metrics"]["success"] == 1.0
    assert info["metrics"]["fault_detected"] == 1.0
    assert info["metrics"]["verify_tp"] == 1.0


def test_verify_is_local_and_misses_downstream_corruption(env):
    env.reset(seed=0, depth_range=(4, 4), fault_index=0)
    play(env, [T, T])  # fault executed at transition 0, now at step 2
    assert not env.belief_valid
    play(env, [V])
    assert env.last_verify == VerifyStatus.SAFE  # transition 1 itself is fine
    assert not env.fault_located


def test_mask_at_start_and_when_budget_exhausted():
    env = ChainEnv(EnvConfig(budget_extra=0))
    env.reset(seed=0, depth_range=(3, 3))
    mask = env.action_mask()
    assert mask[T] and mask[C] and not mask[V] and not mask[R]
    play(env, [T, T, T])
    assert env.remaining_budget == 0
    assert env.action_mask().tolist() == [False, False, False, True]


def test_masked_actions_raise(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(V)
    env.step(C)
    with pytest.raises(RuntimeError):
        env.step(C)


def test_disabled_actions_are_masked():
    env = ChainEnv(EnvConfig(allow_verify=False, allow_revise=False))
    env.reset(seed=0)
    env.step(T)
    mask = env.action_mask()
    assert not mask[V] and not mask[R]


def test_budget_accounting(env):
    env.reset(seed=0, depth_range=(5, 5))
    start = env.remaining_budget
    assert start == 5 + env.cfg.budget_extra
    info = play(env, [T, V, R, T, C])
    assert env.remaining_budget == start - 4
    assert info["metrics"]["cost_spent"] == 4.0
    assert info["metrics"]["num_actions"] == 5.0


def test_random_play_always_terminates():
    env = ChainEnv(EnvConfig())
    rng = np.random.default_rng(0)
    for ep in range(300):
        env.reset(seed=ep, depth_range=(3, 14))
        for _ in range(100):
            _, _, done, _ = env.step(int(rng.choice(np.flatnonzero(env.action_mask()))))
            if done:
                break
        assert done


def test_reset_is_deterministic_given_seed(env):
    first = env.reset(seed=[2, 1, 7, 3], depth_range=(3, 9))
    snapshot = (env.depth, env.fault_index, list(env.cues), list(env.true_values))
    env.reset(seed=123)
    second = env.reset(seed=[2, 1, 7, 3], depth_range=(3, 9))
    assert np.array_equal(first, second)
    assert snapshot == (env.depth, env.fault_index, list(env.cues), list(env.true_values))


def test_faulty_op_always_changes_the_value():
    env = ChainEnv(EnvConfig(p_fault=1.0))
    for ep in range(500):
        env.reset(seed=ep, depth_range=(3, 9))
        values = env.true_values
        for t, tr in enumerate(env.transitions):
            from ultrarl.env import apply_op

            assert apply_op(values[t], tr.faulty_op) != apply_op(values[t], tr.safe_op)


def test_cue_and_fault_rates_match_config():
    cfg = EnvConfig(p_fault=0.8, cue_hit_rate=0.85, cue_false_positive_rate=0.15)
    env = ChainEnv(cfg)
    faults, hits, fault_steps, false_pos, clean_steps = 0, 0, 0, 0, 0
    n = 4000
    for ep in range(n):
        env.reset(seed=ep)
        if env.fault_index is not None:
            faults += 1
            fault_steps += 1
            hits += env.cues[env.fault_index + 1]
        for t in range(env.depth):
            if t != env.fault_index:
                clean_steps += 1
                false_pos += env.cues[t + 1]
    assert abs(faults / n - 0.8) < 0.03
    assert abs(hits / fault_steps - 0.85) < 0.03
    assert abs(false_pos / clean_steps - 0.15) < 0.02


@pytest.mark.parametrize("profile", ["middle", "uniform"])
def test_fault_index_in_range(profile):
    env = ChainEnv(EnvConfig(p_fault=1.0, fault_profile=profile))
    for ep in range(200):
        env.reset(seed=ep, depth_range=(1, 9))
        assert 0 <= env.fault_index < env.depth


def test_config_validation():
    with pytest.raises(ValueError):
        EnvConfig(verify_cost=0)
    with pytest.raises(ValueError):
        EnvConfig(p_fault=1.5)
    with pytest.raises(KeyError):
        EnvConfig().replace(not_a_field=1)


def test_cue_selectivity_counts_fresh_opportunities_only(env):
    env.reset(seed=0, depth_range=(3, 3), fault_index=None)
    env.cues = [0, 1, 0, 0]
    info = play(env, [T, V, V, T, T, C])
    m = info["metrics"]
    # step 1 (cue 1): verified once while fresh; the repeat VERIFY is not an opportunity.
    assert m["fresh_opps_cue1"] == 1 and m["fresh_verifies_cue1"] == 1
    # steps 2 and 3 (cue 0) were fresh opportunities where the agent chose not to verify.
    assert m["fresh_opps_cue0"] == 2 and m["fresh_verifies_cue0"] == 0


def test_incomplete_commit_metrics():
    env = ChainEnv(EnvConfig(budget_extra=0))
    env.reset(seed=0, depth_range=(3, 3), fault_index=None)
    m = play(env, [T, C])["metrics"]
    assert m["incomplete_commit"] == 1.0 and m["voluntary_incomplete_commit"] == 1.0
    env.reset(seed=0, depth_range=(3, 3), fault_index=None)
    m = play(env, [T, T, T, C])["metrics"]
    assert m["incomplete_commit"] == 0.0 and m["success"] == 1.0
