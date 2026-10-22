"""End-to-end test of the figure/table/macro generation on a synthetic results tree."""

import re

import numpy as np
import pandas as pd
import pytest

from ultrarl import report
from ultrarl.config import load_experiment
from ultrarl.run import FINAL_SUFFIX, SUMMARY_METRICS
from ultrarl.stats import summarize

from .test_train import CONFIG


def _fake_results(root, seeds=(0, 1, 2)):
    exp = load_experiment(CONFIG)
    rng = np.random.default_rng(0)
    rows = []
    names = [(m.name, m.display_name) for m in exp.methods]
    names += [(m.name + FINAL_SUFFIX, m.display_name + FINAL_SUFFIX) for m in exp.learned_methods]
    for setting in exp.eval_settings:
        for name, display in names:
            for seed in seeds:
                row = {"setting": setting.name, "method": name, "display_name": display, "seed": seed}
                row.update({k: float(rng.uniform(0.1, 0.9)) for k in SUMMARY_METRICS})
                rows.append(row)
    per_seed = pd.DataFrame(rows)
    main_dir = root / "main"
    (main_dir / "curves").mkdir(parents=True)
    per_seed.to_csv(main_dir / "eval_per_seed.csv", index=False)
    summarize(per_seed, ["setting", "method", "display_name"], SUMMARY_METRICS).to_csv(
        main_dir / "eval_summary.csv", index=False
    )
    for m in exp.learned_methods:
        for seed in seeds:
            episodes = np.arange(1000, 21000, 1000)
            pd.DataFrame(
                {
                    "method": m.name,
                    "seed": seed,
                    "episode": episodes,
                    "train_like_success": rng.uniform(0.1, 0.95, episodes.size),
                    "train_like_fault_detected": rng.uniform(0.0, 1.0, episodes.size),
                    "heldout_long_success": rng.uniform(0.1, 0.9, episodes.size),
                }
            ).to_csv(main_dir / "curves" / f"{m.name}_seed{seed}.csv", index=False)
    return per_seed


def test_report_writes_all_artifacts_and_valid_macros(tmp_path):
    _fake_results(tmp_path / "results")
    report.main(["--results", str(tmp_path / "results"), "--paper", str(tmp_path / "paper")])
    paper = tmp_path / "paper"
    for name in ("length_generalization", "learning_curves", "cue_selectivity", "policy_churn"):
        assert (paper / "figures" / f"{name}.pdf").exists()
    for name in ("main_results", "main_results_final", "shift_results", "behaviour_heldout_long"):
        text = (paper / "tables" / f"{name}.tex").read_text()
        assert text.startswith("\\begin{tabular}") and text.rstrip().endswith("\\end{tabular}")
    macros = (paper / "numbers.tex").read_text()
    defs = re.findall(r"\\csname ur@([^\\]+)\\endcsname\{([^}]*)\}", macros)
    keys = {k for k, _ in defs}
    assert "main/ma_inv/heldout_long/success" in keys
    assert "diff/ma_inv-ma_abs/heldout_long/success/p" in keys
    assert "main/ma_inv/collapse_norepair_rate/pct" in keys
    assert len(keys) == len(defs), "duplicate macro keys"
    for _, value in defs:
        assert value == "--" or re.fullmatch(r"-?[0-9.]+(\\%)?|<0\.001", value), value


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
