import numpy as np
import pandas as pd

from ultrarl.stats import bootstrap_ci, paired_difference, summarize


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(0).normal(0.5, 0.1, size=10)
    low, high = bootstrap_ci(values)
    assert low < values.mean() < high
    assert bootstrap_ci(np.array([0.3])) == (0.3, 0.3)
    assert all(np.isnan(bootstrap_ci(np.array([np.nan]))))


def test_summarize_and_paired_difference():
    df = pd.DataFrame(
        {
            "setting": ["s"] * 6,
            "method": ["a"] * 3 + ["b"] * 3,
            "seed": [0, 1, 2] * 2,
            "success": [0.9, 0.8, 0.7, 0.5, 0.4, 0.3],
        }
    )
    summary = summarize(df, ["setting", "method"], ["success"])
    assert np.isclose(summary.loc[summary.method == "a", "success"].item(), 0.8)
    diff = paired_difference(df, "success", "a", "b", "s")
    assert np.isclose(diff["mean"], 0.4) and np.isclose(diff["lo"], 0.4) and diff["n"] == 3
