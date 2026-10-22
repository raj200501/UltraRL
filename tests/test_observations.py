import numpy as np
import pytest

from ultrarl.config import EnvConfig
from ultrarl.env import Action, ChainEnv
from ultrarl.observations import OBSERVATION_KINDS, observation_dim

T, V, R = Action.THINK, Action.VERIFY, Action.REVISE


@pytest.mark.parametrize("kind", OBSERVATION_KINDS)
def test_observation_shape_and_dtype(kind):
    env = ChainEnv(EnvConfig(observation=kind))
    obs = env.reset(seed=0)
    assert obs.shape == (observation_dim(kind),) and obs.dtype == np.float32
    assert env.obs_dim == observation_dim(kind)


def _obs_after(kind, depth, actions):
    env = ChainEnv(EnvConfig(observation=kind))
    env.reset(seed=0, depth_range=(depth, depth), fault_index=None)
    env.cues = [0] * (depth + 1)
    obs = None
    for a in actions:
        obs = env.step(a)[0]
    return obs


def test_invariant_observation_does_not_depend_on_chain_length():
    # Same situation relative to the chain (two steps in, one checked) on short and long chains.
    short = _obs_after("invariant", 4, [T, V, T])
    long = _obs_after("invariant", 12, [T, V, T])
    assert np.array_equal(short, long)
    # ... and at the end of the chain.
    assert np.array_equal(_obs_after("invariant", 4, [T] * 4), _obs_after("invariant", 12, [T] * 12)[:])


def test_absolute_observation_depends_on_chain_length():
    assert not np.array_equal(_obs_after("absolute", 4, [T, V, T]), _obs_after("absolute", 12, [T, V, T]))


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        observation_dim("polar")


def test_ratio_feature_is_slack_per_remaining_step():
    env = ChainEnv(EnvConfig(observation="ratio", budget_extra=6))
    obs = env.reset(seed=0, depth_range=(4, 4))
    assert obs[-1] == pytest.approx(min(6 / 4, 2.0) / 2.0)


def test_memory_and_pending_bits_after_a_caught_fault():
    def bits(actions):
        env = ChainEnv(EnvConfig(observation="memory"))
        env.reset(seed=0, depth_range=(5, 5), fault_index=1)
        for a in actions:
            env.step(a)
        o = env.observable()
        return o.fault_located, o.repair_pending

    assert bits([T, T]) == (False, False)  # fault executed but not caught
    assert bits([T, T, V]) == (True, True)  # caught
    assert bits([T, T, V, T]) == (True, True)  # continuing past it: still pending
    assert bits([T, T, V, R, T]) == (True, False)  # repaired: memory stays set, pending clears


def test_history_window_codes_order_and_padding():
    from ultrarl.env import HISTORY_EVENTS, HISTORY_LENGTH
    from ultrarl.observations import HISTORY_CODES, HISTORY_STEPS

    assert HISTORY_STEPS == HISTORY_LENGTH and len(HISTORY_EVENTS) == HISTORY_CODES
    env = ChainEnv(EnvConfig(observation="history"))
    env.reset(seed=0, depth_range=(5, 5), fault_index=1)
    for a in (T, T, V, R):  # THINK, THINK, VERIFY (UNSAFE: step 1 is the fault), REVISE
        obs = env.step(a)[0]
    window = obs[9:].reshape(HISTORY_STEPS, HISTORY_CODES)
    assert not window[:4].any()  # zero padding before the first action, oldest first
    assert [int(np.argmax(r)) for r in window[4:]] == [0, 0, 2, 3]  # THINK, THINK, UNSAFE, REVISE


def test_short_history_keeps_only_the_last_three_actions():
    from ultrarl.observations import HISTORY_CODES, observation_dim

    env = ChainEnv(EnvConfig(observation="history3"))
    env.reset(seed=0, depth_range=(5, 5), fault_index=1)
    for a in (T, T, V, R):
        obs = env.step(a)[0]
    assert obs.shape == (observation_dim("history3"),) == (9 + 3 * HISTORY_CODES,)
    window = obs[9:].reshape(3, HISTORY_CODES)
    assert [int(np.argmax(r)) for r in window] == [0, 2, 3]  # THINK, UNSAFE, REVISE
    env.reset(seed=0, depth_range=(5, 5))
    obs = env.step(T)[0]
    assert not obs[9:19].any() and obs[19] == 1.0  # padded, then the one THINK
