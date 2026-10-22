"""Episode rollouts and aggregation of per-episode metrics."""

from __future__ import annotations

import zlib
from collections.abc import Callable

import numpy as np

from ultrarl.config import EnvConfig, EvalSetting
from ultrarl.env import ChainEnv, Observable

# Independent RNG streams: an episode seed is (stream, run seed, setting id, episode).
# Streams guarantee independent sampling of training, validation and evaluation
# episodes; on training lengths the same small set of task configurations recurs
# in all of them, so training-length evaluation is in-distribution by design.
TRAIN_STREAM = 0
CURVE_STREAM = 1
EVAL_STREAM = 2

ActFn = Callable[[np.ndarray, Observable, np.ndarray], int]


def setting_id(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


def episode_seed(stream: int, run_seed: int, episode: int, setting: str = "") -> list[int]:
    return [stream, run_seed, setting_id(setting), episode]


def run_episode(env: ChainEnv, act: ActFn, seed: list[int], depth_range: tuple[int, int]) -> dict[str, float]:
    obs = env.reset(seed=seed, depth_range=depth_range)
    while True:
        mask = env.action_mask()
        obs, _, done, info = env.step(act(obs, env.observable(), mask))
        if done:
            return info["metrics"]


def evaluate(
    act: ActFn,
    env_config: EnvConfig,
    setting: EvalSetting,
    run_seed: int,
    episodes: int,
    stream: int = EVAL_STREAM,
) -> dict[str, float]:
    """Roll out ``episodes`` episodes and return per-run aggregate metrics."""
    env = ChainEnv(env_config)
    rows = [
        run_episode(env, act, episode_seed(stream, run_seed, i, setting.name), setting.depth_range)
        for i in range(episodes)
    ]
    return aggregate(rows)


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    """Collapse per-episode metrics into one row (means, with ratio metrics pooled)."""
    keys = rows[0].keys()
    cols = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}

    def nanmean(x: np.ndarray) -> float:
        x = x[~np.isnan(x)]
        return float(x.mean()) if x.size else float("nan")

    def ratio(num: float, den: float) -> float:
        return float(num / den) if den > 0 else float("nan")

    out = {
        "episodes": float(len(rows)),
        "success": nanmean(cols["success"]),
        "return": nanmean(cols["return"]),
        "num_actions": nanmean(cols["num_actions"]),
        "cost_spent": nanmean(cols["cost_spent"]),
        "cost_adjusted_accuracy": nanmean(cols["cost_adjusted_accuracy"]),
        "verify_count": nanmean(cols["verify_count"]),
        "revise_count": nanmean(cols["revise_count"]),
        "nonrepair_revise_count": nanmean(cols["nonrepair_revise_count"]),
        "incomplete_commit": nanmean(cols["incomplete_commit"]),
        "voluntary_incomplete_commit": nanmean(cols["voluntary_incomplete_commit"]),
        "commit_at_start": nanmean(cols["commit_at_start"]),
        "fault_rate": nanmean(cols["fault_present"]),
        "fault_detected": nanmean(cols["fault_detected"]),
        "success_given_fault": nanmean(cols["success_given_fault"]),
        "success_given_detected": nanmean(cols["success_given_detected"]),
        "verify_precision": ratio(cols["verify_tp"].sum(), cols["verify_count"].sum()),
        "verify_rate_cue0": ratio(cols["fresh_verifies_cue0"].sum(), cols["fresh_opps_cue0"].sum()),
        "verify_rate_cue1": ratio(cols["fresh_verifies_cue1"].sum(), cols["fresh_opps_cue1"].sum()),
    }
    clean = cols["fault_present"] == 0
    out["clean_success"] = float(cols["success"][clean].mean()) if clean.any() else float("nan")
    return out
