"""Quick checks of the analysis scripts behind numbers quoted in the paper."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ultrarl.config import EnvConfig
from ultrarl.dqn import DQNAgent
from ultrarl.env import ChainEnv

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q_probe_statistics_are_consistent():
    q_probe = _load("q_probe")
    q_probe.EPISODES = 40
    env = ChainEnv(EnvConfig())
    agent = DQNAgent(env.obs_dim, env.num_actions, hidden_size=16)  # untrained, so it acts arbitrarily
    stats = q_probe.probe(agent, env, seed=0, depth_range=(3, 6), gamma=0.99)
    assert stats["unsafe_points"] >= 0
    for key in ("think_rate", "aliased_rate"):
        if key in stats and not np.isnan(stats[key]):
            assert 0.0 <= stats[key] <= 1.0
    for key in ("return_think", "return_revise", "return_on_track"):
        if key in stats:
            assert -3.0 < stats[key] < 2.0  # bounded by the terminal rewards plus small shaping


def test_discounted_suffix_sums():
    q_probe = _load("q_probe")
    assert q_probe.discounted_suffix_sums([1.0, 0.0, 2.0], 0.5) == pytest.approx([1.5, 1.0, 2.0])


def test_observation_coverage_ranks_the_encoders():
    coverage = _load("observation_coverage")
    coverage.EPISODES = 150
    seen = {kind: set(coverage.collect(kind, (3, 6), "train_like")) for kind in ("absolute", "invariant")}
    unseen = {
        kind: np.mean([k not in seen[kind] for k in coverage.collect(kind, (10, 14), "heldout_xlong")]) for kind in seen
    }
    assert unseen["absolute"] > 0.9 > 0.3 > unseen["invariant"]
