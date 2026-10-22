"""Quick checks of the analysis scripts behind numbers quoted in the paper."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

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
    # with little slack, some UNSAFE results come when the repair no longer fits in the budget;
    # those are counted but not probed
    tight = q_probe.probe(
        ScriptedAgent(), ChainEnv(EnvConfig(observation="invariant", budget_extra=2)), 0, (3, 6), 0.99
    )
    assert tight["unaffordable_points"] > 0 and stats["unaffordable_points"] < tight["unaffordable_points"]


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


def test_agent_coverage_rollouts_mark_uncued_check_opportunities():
    coverage = _load("agent_coverage")
    from ultrarl.config import EvalSetting

    env = ChainEnv(EnvConfig(observation="invariant"))
    keys, uncued, whole = coverage.rollouts(ScriptedAgent(), env, 0, 0, EvalSetting("train_like", (3, 6)), 20)
    assert len(keys) == len(uncued) == len(whole) > 20
    assert whole[0] and not all(whole)  # the episode outgrows an 8-action window when checking every step
    opportunities = [u for u in uncued if u is not None]
    assert opportunities and all(opportunities)  # the scripted agent checks every new step


def _save_agent(path, seed, episode=1000):
    from ultrarl.dqn import DQNAgent

    torch.manual_seed(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    DQNAgent(9, 4, 8).save(path, {"checkpoint_episode": episode})


def test_compare_checkpoints_reports_identical_differing_and_missing(tmp_path, capsys):
    compare = _load("compare_checkpoints")
    ref, new = tmp_path / "ref", tmp_path / "new"
    for root in (ref, new):
        _save_agent(root / "checkpoints" / "a.pt", seed=0)
    assert compare.main(str(ref), str(new)) == 0
    _save_agent(new / "checkpoints" / "a.pt", seed=1)  # different weights
    _save_agent(ref / "checkpoints" / "b.pt", seed=2)  # not retrained
    assert compare.main(str(ref), str(new)) == 1
    out = capsys.readouterr().out
    assert "differs: a.pt" in out and "missing: b.pt" in out and "this platform" in out


def test_compare_summaries_flags_a_shifted_method(tmp_path, capsys):
    compare = _load("compare_summaries")
    rng = np.random.default_rng(0)
    for name, shift in (("ref", 0.0), ("new", 0.3)):
        (tmp_path / name).mkdir()
        rows = [{"setting": "train_like", "method": m, "seed": k, "success": rng.uniform(0.4, 0.5) + shift * (m == "b")}
                for m in ("a", "b", "cue_triggered") for k in range(10)]  # fmt: skip
        pd.DataFrame(rows).to_csv(tmp_path / name / "eval_per_seed.csv", index=False)
        pd.DataFrame({"method": ["a", "b"]}).to_csv(tmp_path / name / "train_runs.csv", index=False)
    frame = compare.compare(tmp_path / "ref", tmp_path / "new").set_index("method")
    assert set(frame.index) == {"a", "b"} and frame.loc["b", "p"] < 0.001 < frame.loc["a", "p"]
    assert compare.main(str(tmp_path / "ref"), str(tmp_path / "new")) == 0
    assert "1 of 2 rows differ" in capsys.readouterr().out
