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


def test_agent_coverage_rollouts_flag_uncued_and_unaffordable_points():
    coverage = _load("agent_coverage")
    from ultrarl.config import EvalSetting

    env = ChainEnv(EnvConfig(observation="invariant"))
    points = coverage.rollouts(ScriptedAgent(), env, 0, 0, EvalSetting("train_like", (3, 6)), 20)
    assert len(points) > 20 and points.whole.iloc[0] and not points.whole.all()
    assert points.verify[points.uncued].all()  # the scripted agent checks every new step
    tight = coverage.rollouts(ScriptedAgent(), ChainEnv(EnvConfig(observation="invariant", budget_extra=2)), 0, 0,
                              EvalSetting("train_like", (3, 6)), 20)  # fmt: skip
    assert tight.broke.any() and tight.spend[tight.broke].any()  # it keeps checking with no slack left


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


def test_compare_summaries_reports_paired_shifts_and_headline_comparisons(tmp_path, capsys):
    from ultrarl.report import PRIMARY

    compare = _load("compare_summaries")
    rng = np.random.default_rng(0)
    base = {"ma_inv": 0.8, "ma_abs": 0.5, "outcome_inv": 0.6, "outcome_abs": 0.3, "cue_triggered": 0.85}
    assert {m for a, b, _ in PRIMARY for m in (a, b)} <= set(base)
    for name, shift in (("ref", 0.0), ("new", 0.4)):  # the retrained ma_abs is 0.4 better: it now beats ma_inv
        (tmp_path / name).mkdir()
        rows = [{"setting": s, "method": m, "seed": k, "success": v + shift * (m == "ma_abs") + rng.normal(0, 0.01)}
                for m, v in base.items() for s in compare.LENGTHS for k in range(10)]  # fmt: skip
        pd.DataFrame(rows).to_csv(tmp_path / name / "eval_per_seed.csv", index=False)
        pd.DataFrame({"method": [m for m in base if m != "cue_triggered"]}).to_csv(
            tmp_path / name / "train_runs.csv", index=False
        )
    frame = compare.differences(tmp_path / "ref", tmp_path / "new").set_index(["method", "setting"])
    assert "cue_triggered" not in frame.index.get_level_values(0)  # hand-coded policies are not retrained
    assert frame.loc[("ma_abs", "train_like"), "lo"] > 0.3 and frame.loc[("ma_inv", "train_like"), "hi"] < 0.05
    head = compare.headline(tmp_path / "ref", tmp_path / "new").set_index(["a", "b", "setting"])
    assert not head.loc[("ma_inv", "ma_abs", "heldout_long"), "same_sign"]
    assert head.loc[("ma_inv", "outcome_inv", "train_like"), "same_sign"]
    assert compare.main(str(tmp_path / "ref"), str(tmp_path / "new")) == 0
    out = capsys.readouterr().out
    assert "intervals exclude zero" in out and "4 of 6 headline comparisons keep their sign" in out


def test_low_detection_inspection_counts_checks_and_their_positions():
    inspect = _load("inspect_low_detection")
    inspect.EPISODES = 20
    stats = inspect.inspect(ScriptedAgent(), ChainEnv(EnvConfig(observation="invariant")), 0, (3, 6))
    assert stats["checks_min"] >= 1 and stats["checks_mean"] <= stats["checks_max"]
    assert 0 < stats["final_step_share"] < 1  # it checks every step, not only the last
    assert stats["check_rate_cue0"] == stats["check_rate_cue1"] == 1.0


def test_agent_coverage_finds_training_length_observations_familiar():
    # regression: pandas' Series.isin mishandles these bytes keys and reported almost everything unseen
    coverage = _load("agent_coverage")
    coverage.REFERENCE_EPISODES, coverage.EPISODES = 300, 50
    rows = {r["setting"]: r for r in coverage.job(("main", "ma_inv", 0, "best"))}
    assert rows["train_like"]["unseen_fraction"] < 0.1
    assert rows["heldout_xlong"]["unseen_fraction"] < 0.2


def test_show_episode_traces_a_repair():
    show = _load("show_episode")
    lines = show.trace(ScriptedAgent(), ChainEnv(EnvConfig(observation="invariant")), 0, 0, (4, 4))
    assert lines[0].startswith("chain of 4 steps")
    assert lines[-1].split()[-1] in ("success", "failure") and "COMMIT" in lines[-1]
    unsafe = [i for i, line in enumerate(lines) if "UNSAFE" in line]
    assert all("REVISE" in lines[i + 1] for i in unsafe)  # the scripted agent always repairs


def test_show_episode_classifies_the_action_after_a_catch():
    show = _load("show_episode")
    lines = [
        "chain",
        "  step 2/4  cue 1  slack  3  VERIFY UNSAFE",
        "  step 2/4  cue 1  slack  2  REVISE ",
        "  step 3/4  cue 0  slack  1  VERIFY UNSAFE",
        "  step 3/4  cue 0  slack  0  THINK  ",
    ]
    assert show.after_catch(lines) == {"repair", "continue"}
    assert show.after_catch(lines[:3]) == {"repair"}


def test_compare_pngs_ignores_compression_but_not_pixels():
    from io import BytesIO

    from PIL import Image

    compare = _load("compare_pngs")
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[2:5, 3:6] = (200, 30, 30, 255)

    def png(array, level):
        out = BytesIO()
        Image.fromarray(array).save(out, format="png", compress_level=level)
        return out.getvalue()

    fast, small = png(image, 1), png(image, 9)
    assert fast != small and compare.same_pixels(fast, small)
    changed = image.copy()
    changed[0, 0, 0] = 1
    assert not compare.same_pixels(fast, png(changed, 1))
