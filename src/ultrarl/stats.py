"""Across-seed summary statistics.

Intervals are Student-t intervals of the mean over independent seeds. With 10
seeds a plain percentile bootstrap is noticeably narrower than a t-interval (it
uses the 1/n variance and normal quantiles), so it undercovers even for normal
data; t-intervals are the standard small-sample choice. Paired comparisons
(methods evaluated on identical episodes for each seed) also report the number
of seeds favouring each side and an exact sign-flip permutation p-value; the
family of headline comparisons (chosen after the results) is Holm-adjusted.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats as sps

# Suffix marking last-checkpoint rows (e.g. ``ma_inv@final``) in evaluation outputs.
FINAL_SUFFIX = "@final"

# Metrics that are probabilities; their intervals are clipped to [0, 1].
PROBABILITY_METRICS = {
    "success",
    "fault_detected",
    "success_given_fault",
    "success_given_detected",
    "clean_success",
    "verify_precision",
    "verify_rate_cue0",
    "verify_rate_cue1",
    "incomplete_commit",
    "voluntary_incomplete_commit",
    "commit_at_start",
}


def t_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Student-t interval of the mean (NaNs dropped); degenerate for n < 2."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean
    half = sps.t.ppf(0.5 + confidence / 2.0, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size)
    return mean - float(half), mean + float(half)


def sign_flip_pvalue(diffs: np.ndarray) -> float:
    """Exact two-sided paired permutation (sign-flip) test of mean difference = 0."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = diffs.size
    if n == 0:
        return float("nan")
    if n > 16:
        raise ValueError("exact sign-flip test is only implemented for n <= 16")
    observed = abs(diffs.mean())
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((null >= observed - 1e-12).mean())


def summarize(per_seed: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean, 95% t-interval and number of contributing seeds for each group and metric."""
    rows = []
    for keys, group in per_seed.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            low, high = t_interval(values)
            if metric in PROBABILITY_METRICS:
                low, high = max(low, 0.0), min(high, 1.0)
            row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            row[f"{metric}_lo"] = low
            row[f"{metric}_hi"] = high
            row[f"{metric}_n"] = int(np.isfinite(values).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(
    per_seed: pd.DataFrame, metric: str, method_a: str, method_b: str, setting: str
) -> dict[str, float]:
    """Mean of (a - b) over seeds with a t-interval, seed counts and a sign-flip p-value."""
    sub = per_seed[per_seed["setting"] == setting].pivot(index="seed", columns="method", values=metric)
    diff = (sub[method_a] - sub[method_b]).to_numpy(dtype=float)
    diff = diff[~np.isnan(diff)]
    low, high = t_interval(diff)
    return {
        "mean": float(diff.mean()) if diff.size else float("nan"),
        "lo": low,
        "hi": high,
        "n": int(diff.size),
        "n_positive": int((diff > 0).sum()),
        "n_negative": int((diff < 0).sum()),
        "p_value": sign_flip_pvalue(diff),
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()
