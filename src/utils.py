from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sem(values: list[float] | np.ndarray | pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    if array.size <= 1:
        return 0.0
    return float(array.std(ddof=1) / math.sqrt(array.size))


def mean_ignore_nan(values: list[float] | np.ndarray | pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return float("nan")
    return float(array.mean())


def bool_mask_to_tensor(mask: np.ndarray | list[bool], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def format_pm(mean_value: float, se_value: float, decimals: int = 3) -> str:
    if math.isnan(mean_value):
        return "--"
    return f"{mean_value:.{decimals}f} $\\pm$ {se_value:.{decimals}f}"


def merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def listify(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def getenv_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)
