"""Quick checks of the analysis scripts behind numbers quoted in the paper."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ultrarl.config import EnvConfig
from ultrarl.env import Action, ChainEnv

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptedAgent:
    """Checks every new step and repairs on UNSAFE (invariant observation), with Q-values that
    prefer exactly the action it takes."""

    def act(self, obs, mask):
        at_end, at_start, unsafe, fresh = obs[0] == 1, obs[1] == 1, obs[8] == 1, obs[6] == 1 and obs[7] == 0
        if unsafe and mask[Action.REVISE]:
            return int(Action.REVISE)
        if not at_start and fresh and obs[5] == 0 and mask[Action.VERIFY]:
            return int(Action.VERIFY)
        return int(Action.COMMIT if at_end or not mask[Action.THINK] else Action.THINK)

    def q_values(self, obs):
        q = np.zeros(4)
        q[self.act(obs, np.ones(4, dtype=bool))] = 1.0
        return q


def test_q_probe_on_a_scripted_repairing_agent():
    q_probe = _load("q_probe")
    q_probe.EPISODES = 60
    env = ChainEnv(EnvConfig(observation="invariant"))
    stats = q_probe.probe(ScriptedAgent(), env, seed=0, depth_range=(3, 6), gamma=0.99)
    assert stats["unsafe_points"] > 10
    assert stats["think_rate"] == 0.0  # it always repairs
    assert stats["q_revise"] == 1.0 and stats["q_think"] == 0.0
    # continuing past a caught fault always fails, so repairing is worth far more
    assert stats["return_think"] < -1.0 and stats["return_revise"] > stats["return_think"] + 1.0
    assert 0.0 <= stats["aliased_rate"] <= 1.0
    assert stats["q_start"] == 1.0  # the scripted Q puts 1 on the chosen action
    assert -2.0 < stats["return_start"] < 1.5


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
