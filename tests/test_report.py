"""End-to-end test of the figure/table/macro generation on a synthetic results tree
covering every experiment the paper reports."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ultrarl import report
from ultrarl.config import load_experiment
from ultrarl.observations import OBSERVATION_KINDS
from ultrarl.run import FINAL_SUFFIX, SUMMARY_METRICS
from ultrarl.stats import summarize

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENTS = ("main", "ablations", "sensitivity", "followups", "encoders")


def _fake_experiment(root: Path, name: str, rng: np.random.Generator, seeds=(0, 1, 2)) -> None:
    exp = load_experiment(CONFIGS / f"{name}.yaml")
    names = [(m.name, m.display_name) for m in exp.methods]
    names += [(m.name + FINAL_SUFFIX, m.display_name + FINAL_SUFFIX) for m in exp.learned_methods]
    rows = []
    for setting in exp.eval_settings:
        for method, display in names:
            for seed in seeds:
                row = {"setting": setting.name, "method": method, "display_name": display, "seed": seed}
                row.update({k: float(rng.uniform(0.1, 0.9)) for k in SUMMARY_METRICS})
                rows.append(row)
    per_seed = pd.DataFrame(rows)
    out = root / name
    (out / "curves").mkdir(parents=True)
    (out / "config.yaml").write_text((CONFIGS / f"{name}.yaml").read_text())
    per_seed.to_csv(out / "eval_per_seed.csv", index=False)
    summarize(per_seed, ["setting", "method", "display_name"], SUMMARY_METRICS).to_csv(
        out / "eval_summary.csv", index=False
    )
    episodes = np.arange(1000, 21000, 1000)
    for m in exp.learned_methods:
        for seed in seeds:
            pd.DataFrame(
                {
                    "method": m.name,
                    "seed": seed,
                    "episode": episodes,
                    "train_like_success": rng.uniform(0.1, 0.95, episodes.size),
                    "train_like_fault_detected": rng.uniform(0.0, 1.0, episodes.size),
                    "heldout_long_success": rng.uniform(0.1, 0.9, episodes.size),
                }
            ).to_csv(out / "curves" / f"{m.name}_seed{seed}.csv", index=False)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("report")
    rng = np.random.default_rng(0)
    for name in EXPERIMENTS:
        _fake_experiment(tmp / "results", name, rng)
    pd.DataFrame(
        [
            {"observation": kind, "setting": s, "unseen_fraction": float(rng.uniform()), "points": 100}
            for kind in OBSERVATION_KINDS
            for s in ("heldout_long", "heldout_xlong")
        ]
    ).to_csv(tmp / "results" / "observation_coverage.csv", index=False)
    probe = [
        {"method": m, "checkpoint": which, "seed": seed, **{k: float(rng.uniform()) for k in report.PROBE_METRICS}}
        for m in ("ma_inv", "ma_repair")
        for which in ("best", "final")
        for seed in range(3)
    ]
    pd.DataFrame(probe).to_csv(tmp / "results" / "q_probe.csv", index=False)
    stage = tmp / "results" / "pilot" / "stage1"
    stage.mkdir(parents=True)
    pilot = [
        {"setting": s, "method": m, "display_name": m, "success": 0.5, "success_lo": 0.4, "success_hi": 0.6}
        for s in ("train_like", "heldout_long")
        for m in ("ma_inv_lr0.001_tu200", "outcome_inv_tau005_lr3e4")
    ]
    pd.DataFrame(pilot).to_csv(stage / "eval_summary.csv", index=False)
    report.main(["--results", str(tmp / "results"), "--paper", str(tmp / "paper")])
    return tmp / "paper"


def test_report_writes_every_figure_and_table(generated):
    for name in ("length_generalization", "learning_curves", "cue_selectivity", "policy_churn", "optimal_decisions"):
        assert (generated / "figures" / f"{name}.pdf").exists()
    for name in (
        "main_results",
        "main_results_final",
        "shift_results",
        "behaviour_heldout_long",
        "behaviour_heldout_xlong",
        "ablations_results",
        "ablations_results_final",
        "sensitivity_results",
        "sensitivity_rewards",
        "pilot_results",
        "followup_results",
        "encoder_results",
    ):
        text = (generated / "tables" / f"{name}.tex").read_text()
        assert text.startswith("\\begin{tabular}") and text.rstrip().endswith("\\end{tabular}"), name


def test_report_macros_are_well_formed_and_cover_every_experiment(generated):
    macros = (generated / "numbers.tex").read_text()
    defs = re.findall(r"\\csname ur@([^\\]+)\\endcsname\{((?:[^{}]|\{[^{}]*\})*)\}", macros)
    keys = {k for k, _ in defs}
    assert len(keys) == len(defs), "duplicate macro keys"
    for key in (
        "main/ma_inv/heldout_long/success",
        "diff/ma_inv-ma_abs/heldout_long/success/pholm",
        "main/ma_inv/collapse_norepair_rate/pct",
        "main/ma_inv/collapse_rate/lo",
        "ablations/no_cue_terms/collapse_rate",
        "abl/ma_inv-no_cue_terms/heldout_xlong/success/p",
        "sens/ma_inv_polyak-ma_abs_polyak/heldout_xlong/success",
        "followups/ma_repair/collapse_rate/pct",
        "diff/ma_repair@final-ma_inv@final/heldout_long/success",
        "diff/ma_abs_nopos-ma_abs/heldout_xlong/success",
        "coverage/absolute_nopos/heldout_xlong/unseen_rate/pct",
        "probe/ma_inv@final/think_rate/pct",
        "probe/ma_repair/q_gap/lo",
        "main/optimal/lengths/max/pct",
        "ablations/check_paying@final/train_like/min",
        "diff/no_cue_terms-outcome_inv/heldout_xlong/success/p",
        "abl/ma_inv@final-no_waste_penalty@final/heldout_long/success",
        "sens/ma_inv_polyak-outcome_inv_polyak/heldout_long/success",
        "sensitivity/naive_inv_sync1000/collapse_rate/pct",
    ):
        assert key in keys, key
    for _, value in defs:
        assert value == "--" or re.fullmatch(r"(\\ensuremath\{-\})?[0-9.]+(\\%)?|<0\.001", value), value


def test_curve_stats_separates_never_check_from_detect_without_repair():
    curves = pd.DataFrame(
        {
            "method": "m",
            "seed": [0] * 4,
            "episode": [6000, 7000, 8000, 9000],
            "train_like_success": [0.9, 0.3, 0.25, 0.3],
            "train_like_fault_detected": [0.95, 0.9, 0.05, 0.3],
        }
    )
    per_seed = pd.DataFrame({"method": ["m"], "setting": ["train_like"], "success": [0.4], "seed": [0]})
    stats = report.curve_stats(curves, per_seed)
    assert stats["main/m/collapse_rate"] == pytest.approx(0.75)
    assert stats["main/m/collapse_norepair_rate"] == pytest.approx(0.25)
    assert stats["main/m/collapse_nevercheck_rate"] == pytest.approx(0.25)
    assert stats["main/m/seeds_peak_above_half"] == "1"
    assert stats["main/m/seeds_selected_above_half"] == "0"


def test_holm_adjustment_applies_only_to_primary_comparisons():
    from ultrarl.stats import holm

    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
