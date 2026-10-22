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
EXPERIMENTS = ("main", "ablations", "sensitivity", "followups", "encoders", "process")


def _fake_experiment(root: Path, name: str, rng: np.random.Generator, seeds=(0, 1, 2)) -> None:
    exp = load_experiment(CONFIGS / f"{name}.yaml")
    names = [(m.name, m.display_name) for m in exp.methods]
    names += [(m.name + FINAL_SUFFIX, m.display_name + FINAL_SUFFIX) for m in exp.learned_methods]
    rows = []
    for setting in exp.eval_settings:
        for method, display in names:
            for seed in seeds:
                row = {"setting": setting.name, "method": method, "display_name": display, "seed": seed}
                row.update({k: float(rng.uniform(0.1, 0.9)) for k in (*SUMMARY_METRICS, "fault_rate")})
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
    pd.DataFrame(
        [{"method": m.name, "seed": seed, "episodes": 20000, "selected_episode": int(rng.choice(episodes)),
          "wall_seconds": 240.0, "selection_success": float(rng.uniform())} for m in exp.learned_methods
         for seed in seeds]
    ).to_csv(out / "train_runs.csv", index=False)  # fmt: skip
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
    pd.DataFrame(
        [
            {"method": "ma_history", "seed": seed, "setting": "heldout_xlong", "unseen_fraction": 0.3,
             "uncued_check_seen": 0.6, "uncued_check_unseen": np.nan if seed else 0.1, "whole_episode_in_window": 0.5}
            for seed in range(3)
        ]
    ).to_csv(tmp / "results" / "agent_coverage.csv", index=False)  # fmt: skip
    probe = [
        {
            "method": m,
            "checkpoint": which,
            "seed": seed,
            "unsafe_points": 40,
            "unaffordable_points": 10,
            **{k: float(rng.uniform()) for k in report.PROBE_METRICS},
        }
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
        "probe_results",
        "pending_results",
        "process_results",
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
        "probe/ma_inv/think_bias",
        "diff/ma_pending@final-ma_inv@final/heldout_long/success",
        "diff/outcome_pending-outcome_inv/train_like/success/p",
        "main/optimal/lengths/max/pct",
        "ablations/check_paying@final/train_like/min",
        "diff/no_cue_terms-outcome_inv/heldout_xlong/success/p",
        "abl/ma_inv@final-no_waste_penalty@final/heldout_long/success",
        "sens/ma_inv_polyak-outcome_inv_polyak/heldout_long/success",
        "sensitivity/naive_inv_sync1000/collapse_rate/pct",
        "sensitivity/outcome_inv_40k/collapse_rate/pct",
        "main/outcome_inv/first_above_half_median",
        "main/ma_inv/collapse_after_first_rate/pct",
        "followups/ma_repair/selected_episode_median",
        "followups/naive_pending/collapse_norepair_rate/pct",
        "diff/ma_pending@final-outcome_pending@final/heldout_long/success/p",
    ):
        assert key in keys, key
    for _, value in defs:
        assert value == "--" or re.fullmatch(r"(\\ensuremath\{-\})?[0-9.,]+(\\%)?|<0\.001", value), value


def test_curve_stats_separates_low_detection_from_detect_without_repair():
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
    assert stats["main/m/collapse_lowdetect_rate"] == pytest.approx(0.25)
    assert stats["main/m/seeds_peak_above_half"] == "1"
    assert stats["main/m/seeds_selected_above_half"] == "0"


def test_holm_adjustment_applies_only_to_primary_comparisons():
    from ultrarl.stats import holm

    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    methods = {m for a, b, _ in report.COMPARISONS for m in (a, b)}
    settings = {s for _, _, s in report.COMPARISONS}
    per_seed = pd.DataFrame(
        [
            {"method": m, "setting": s, "seed": k, "success": rng.uniform()}
            for m in methods
            for s in settings
            for k in range(5)
        ]
    )
    frame = report.key_numbers(per_seed)
    primary = frame[frame.primary]
    assert len(primary) == len(report.PRIMARY) and frame.loc[~frame.primary, "p_holm"].isna().all()
    assert np.allclose(primary.p_holm, holm(primary.p_value.tolist()))


def test_unrepaired_loss_counts_seeds_without_detections_as_zero():
    per_seed = pd.DataFrame(
        {
            "method": ["m", "m"],
            "setting": ["s", "s"],
            "fault_rate": [0.8, 0.8],
            "fault_detected": [0.5, 0.0],
            "success_given_detected": [0.5, np.nan],
        }
    )
    assert report.unrepaired_loss_macros(per_seed, "x")["x/m/s/unrepaired_rate"] == pytest.approx(0.1)


def test_compute_macros_counts_untimed_runs_at_the_mean_for_their_length(tmp_path):
    for name, rows in {
        "a": [(20000, 100.0), (20000, np.nan), (40000, 400.0)],
        "b": [(20000, 300.0)],
    }.items():
        (tmp_path / name).mkdir()
        frame = pd.DataFrame(rows, columns=["episodes", "wall_seconds"]).assign(method="m", seed=range(len(rows)))
        frame.to_csv(tmp_path / name / "train_runs.csv", index=False)
    (tmp_path / "ablations").mkdir()  # its ma_inv runs repeat the main experiment's and count once
    shared = pd.DataFrame({"method": ["ma_inv", "x"], "seed": 0, "episodes": 20000, "wall_seconds": [3600.0, 7200.0]})
    shared.to_csv(tmp_path / "ablations" / "train_runs.csv", index=False)
    macros = report.compute_macros(tmp_path, ["a", "b", "ablations"])
    assert macros["runs/untimed"] == "1" and macros["runs/saved"] == "6" and macros["runs/distinct"] == "5"
    assert macros["runs/ablations/own"] == "1"
    mean_20k = (100 + 300 + 3600 + 7200) / 4  # the untimed 20k-episode run counts at this
    distinct = (100 + mean_20k + 400 + 300 + 7200) / 3600
    assert macros["runs/core_hours"] == f"{distinct:.0f}"
    assert macros["runs/reproduce_hours_on_4_cores"] == f"{(distinct + 1) / 4:.0f}"


def test_curve_comparisons_pair_late_success_and_first_crossing_by_seed():
    rows = []
    for seed in range(3):
        for method, values in {"a": [0.6, 0.9, 0.9], "b": [0.2, 0.6, 0.5 + 0.1 * seed]}.items():
            rows += [{"method": method, "seed": seed, "episode": e, "train_like_success": v}
                     for e, v in zip([1000, 6000, 7000], values, strict=True)]  # fmt: skip
    frame = report.curve_comparisons(pd.DataFrame(rows), [("a", "b")]).set_index("metric")
    assert frame.loc["late_success", "mean"] == pytest.approx(0.9 - np.mean([0.55, 0.6, 0.65]))
    assert frame.loc["late_success", "n_positive"] == 3
    assert frame.loc["first_above_half", "mean"] == pytest.approx(-5000)  # a crosses at 1,000, b at 6,000


def test_every_result_the_paper_references_is_in_the_committed_numbers():
    paper = Path(__file__).resolve().parents[1] / "paper"
    defined = set(re.findall(r"\\csname ur@(.+?)\\endcsname", (paper / "numbers.tex").read_text(encoding="utf-8")))
    missing = report.used_macros(paper) - defined
    assert not missing, sorted(missing)[:10]
