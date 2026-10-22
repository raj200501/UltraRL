"""How novel are held-out observations under a learned agent's own behaviour, and does
the agent act differently on the novel ones?

    python scripts/agent_coverage.py   # writes results/agent_coverage.csv

``observation_coverage.py`` measures novelty under the hand-coded policies, which
isolates the encoder. Here each selected checkpoint is instead rolled out greedily:
its own training-length rollouts form the reference sample, and on fresh
episodes of every length we record which decision points have an observation
outside that sample, and how often the agent checks an un-cued unchecked step at
seen and at unseen points. If over-checking on long chains came from novel
inputs, it would be concentrated at the unseen points. We also record how often
the whole episode so far fits in the action-history window (``HISTORY_LENGTH``
actions), so that the window reveals position and the full record of checks.
"""

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ultrarl.config import load_experiment
from ultrarl.dqn import DQNAgent
from ultrarl.env import HISTORY_LENGTH, Action, ChainEnv
from ultrarl.rollout import CURVE_STREAM, EVAL_STREAM, episode_seed
from ultrarl.train import checkpoint_path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "main": ["outcome_inv", "ma_inv"],
    "followups": ["ma_pending", "outcome_history", "naive_history", "ma_history", "outcome_history3",
                  "naive_history3", "ma_history3"],
}  # fmt: skip
REFERENCE_EPISODES, EPISODES = 2000, 500


def rollouts(agent: DQNAgent, env: ChainEnv, stream: int, seed: int, setting, episodes: int):
    """Observation keys at every decision point; for points where an un-cued unchecked
    step could be checked, whether the agent checked it (None elsewhere); and whether the
    episode so far fits in the history window."""
    keys, uncued, whole = [], [], []
    for i in range(episodes):
        obs = env.reset(seed=episode_seed(stream, seed, i, setting.name), depth_range=setting.depth_range)
        done = False
        while not done:
            mask = env.action_mask()
            action = agent.act(obs, mask)
            o = env.observable()
            fresh = o.step > 0 and not o.step_checked and mask[Action.VERIFY] and o.cue == 0
            keys.append(np.round(obs, 6).tobytes())
            uncued.append(action == Action.VERIFY if fresh else None)
            whole.append(len(env.events) <= HISTORY_LENGTH)
            obs, _, done, _ = env.step(action)
    return keys, uncued, whole


def job(args: tuple[str, str, int]) -> list[dict]:
    torch.set_num_threads(1)
    experiment, name, seed = args
    exp = load_experiment(ROOT / "configs" / f"{experiment}.yaml")
    method = next(m for m in exp.methods if m.name == name)
    settings = {s.name: s for s in exp.eval_settings if not s.env_overrides}
    agent, _ = DQNAgent.load(checkpoint_path(ROOT / "results" / experiment, method, seed, "best"))
    env = ChainEnv(exp.method_env(method))
    seen = set(rollouts(agent, env, CURVE_STREAM, seed, settings["train_like"], REFERENCE_EPISODES)[0])
    rows = []
    for setting in ("train_like", "heldout_long", "heldout_xlong"):
        keys, uncued, whole = rollouts(agent, env, EVAL_STREAM, seed, settings[setting], EPISODES)
        novel = np.array([k not in seen for k in keys])
        checks = np.array([np.nan if u is None else float(u) for u in uncued])
        rows.append(
            {
                "experiment": experiment,
                "method": name,
                "seed": seed,
                "setting": setting,
                "points": len(keys),
                "unseen_fraction": float(novel.mean()),
                "uncued_check_seen": float(np.nanmean(checks[~novel]))
                if (~novel & ~np.isnan(checks)).any()
                else np.nan,
                "uncued_check_unseen": float(np.nanmean(checks[novel]))
                if (novel & ~np.isnan(checks)).any()
                else np.nan,
                "uncued_points_unseen": int((novel & ~np.isnan(checks)).sum()),
                "whole_episode_in_window": float(np.mean(whole)),
            }
        )
    return rows


def main() -> None:
    jobs = []
    for experiment, names in METHODS.items():
        exp = load_experiment(ROOT / "configs" / f"{experiment}.yaml")
        jobs += [(experiment, name, seed) for name in names for seed in exp.seeds]
    with mp.get_context("spawn").Pool(max(1, (mp.cpu_count() or 2) - 1)) as pool:
        rows = [row for chunk in pool.imap(job, jobs) for row in chunk]
    out = ROOT / "results" / "agent_coverage.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
