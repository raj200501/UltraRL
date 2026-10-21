"""Across-seed summary statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

BOOTSTRAP_SAMPLES = 10_000


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over independent runs (NaNs dropped)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(BOOTSTRAP_SAMPLES, values.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def summarize(per_seed: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean and 95% bootstrap CI over seeds for each group and metric."""
    rows = []
    for keys, group in per_seed.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys, strict=False))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_ci(values)
            row[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            row[f"{metric}_lo"] = low
            row[f"{metric}_hi"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(
    per_seed: pd.DataFrame, metric: str, method_a: str, method_b: str, setting: str
) -> dict[str, float]:
    """Mean of (a - b) over seeds with a bootstrap CI; runs are paired by seed (same eval episodes)."""
    sub = per_seed[per_seed["setting"] == setting].pivot(index="seed", columns="method", values=metric)
    diff = (sub[method_a] - sub[method_b]).to_numpy(dtype=float)
    low, high = bootstrap_ci(diff)
    return {"mean": float(np.nanmean(diff)), "lo": low, "hi": high, "n": int(np.isfinite(diff).sum())}
