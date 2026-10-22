import numpy as np
import pandas as pd
import pytest

from ultrarl.stats import paired_difference, sign_flip_pvalue, summarize, t_interval


def test_t_interval_matches_textbook_formula():
    values = np.array([0.2, 0.4, 0.6, 0.8])
    low, high = t_interval(values)
    half = 3.182446305284263 * values.std(ddof=1) / 2.0  # t_{0.975, 3}
    assert low == pytest.approx(0.5 - half) and high == pytest.approx(0.5 + half)
    assert t_interval(np.array([0.3])) == (0.3, 0.3)
    assert all(np.isnan(t_interval(np.array([np.nan]))))


def test_t_interval_has_nominal_coverage_on_normal_data():
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(2000):
        low, high = t_interval(rng.normal(0.0, 1.0, size=10))
        hits += low <= 0.0 <= high
    assert abs(hits / 2000 - 0.95) < 0.015


def test_sign_flip_pvalue():
    assert sign_flip_pvalue(np.ones(10)) == pytest.approx(2 / 1024)
    assert sign_flip_pvalue(np.array([1.0, -1.0, 1.0, -1.0])) == 1.0
    with pytest.raises(ValueError):
        sign_flip_pvalue(np.ones(20))


def test_summarize_clips_probabilities_and_counts_seeds():
    df = pd.DataFrame(
        {
            "setting": ["s"] * 6,
            "method": ["a"] * 3 + ["b"] * 3,
            "seed": [0, 1, 2] * 2,
            "success": [0.0, 0.0, 0.1, 0.5, 0.4, 0.3],
            "fault_detected": [np.nan, 0.5, 0.7, 1.0, 1.0, 1.0],
        }
    )
    summary = summarize(df, ["setting", "method"], ["success", "fault_detected"])
    a = summary[summary.method == "a"].iloc[0]
    assert a["success_lo"] == 0.0  # clipped at 0
    assert a["fault_detected_n"] == 2 and a["success_n"] == 3


def test_paired_difference():
    df = pd.DataFrame(
        {
            "setting": ["s"] * 6,
            "method": ["a"] * 3 + ["b"] * 3,
            "seed": [0, 1, 2] * 2,
            "success": [0.9, 0.8, 0.7, 0.5, 0.4, 0.3],
        }
    )
    diff = paired_difference(df, "success", "a", "b", "s")
    assert np.isclose(diff["mean"], 0.4) and np.isclose(diff["lo"], 0.4) and diff["n"] == 3
    assert diff["n_positive"] == 3 and diff["p_value"] == pytest.approx(0.25)
