import numpy as np
import pytest

from ultrarl.config import EnvConfig
from ultrarl.env import Action, ChainEnv
from ultrarl.observations import observation_dim

T, V = Action.THINK, Action.VERIFY


@pytest.mark.parametrize("kind", ["absolute", "invariant"])
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
