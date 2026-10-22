"""How novel are observations under a learned agent's own behaviour, and where does
the agent spend budget it cannot afford?

    python scripts/agent_coverage.py   # writes results/agent_coverage.csv

``observation_coverage.py`` measures novelty under the hand-coded policies, which
isolates the encoder. Here each selected and each last checkpoint is instead rolled
out greedily:
its own training-length rollouts form the reference sample, and on fresh episodes
of every length we record

* the share of decision points whose observation is outside that sample;
* at decision points with no slack left and the chain unfinished, where any
  ``VERIFY`` or ``REVISE`` makes finishing impossible, how often the agent spends
  anyway, and how many of these points are unseen. A greedy policy acts at a
  familiar observation as it did at training lengths, so its spending at such
  points is fixed by its training-length behaviour; what differs between agents
  is how often the critical points are unseen, and what they do there;
* the slack spent per episode at familiar and at unseen observations (a
  ``REVISE`` also counts the step it makes the agent redo);
* the un-cued check rate in the first half of the chain, where most of the slack
  is spent;
* how often the whole episode so far fits in the action-history window
  (``HISTORY_LENGTH`` actions), so that the window reveals position and the full
  record of checks.
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


def rollouts(agent: DQNAgent, env: ChainEnv, stream: int, seed: int, setting, episodes: int) -> pd.DataFrame:
    """One row per decision point of greedy rollouts."""
    cfg = env.cfg  # slack used: a VERIFY costs its own price, a REVISE also the redone step
    costs = {Action.VERIFY: cfg.verify_cost, Action.REVISE: cfg.revise_cost + cfg.think_cost}
    rows = []
    for i in range(episodes):
        obs = env.reset(seed=episode_seed(stream, seed, i, setting.name), depth_range=setting.depth_range)
        done = False
        while not done:
            mask = env.action_mask()
            action = agent.act(obs, mask)
            o = env.observable()
            rows.append(
                {
                    "key": np.round(obs, 6).tobytes(),
                    "uncued": bool(o.step > 0 and not o.step_checked and mask[Action.VERIFY] and o.cue == 0),
                    "verify": action == Action.VERIFY,
                    "spend": action in (Action.VERIFY, Action.REVISE),
                    "cost": costs.get(Action(action), 0),
                    "broke": o.slack == 0 and o.step < o.depth,  # spending now leaves too little to finish
                    "early": o.step <= o.depth / 2,
                    "whole": len(env.events) <= HISTORY_LENGTH,
                }
            )
            obs, _, done, _ = env.step(action)
    return pd.DataFrame(rows)


def _rate(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def job(args: tuple[str, str, int, str]) -> list[dict]:
    torch.set_num_threads(1)
    experiment, name, seed, which = args
    exp = load_experiment(ROOT / "configs" / f"{experiment}.yaml")
    method = next(m for m in exp.methods if m.name == name)
    settings = {s.name: s for s in exp.eval_settings if not s.env_overrides}
    agent, _ = DQNAgent.load(checkpoint_path(ROOT / "results" / experiment, method, seed, which))
    env = ChainEnv(exp.method_env(method))
    seen = set(rollouts(agent, env, CURVE_STREAM, seed, settings["train_like"], REFERENCE_EPISODES).key)
    rows = []
    for setting in ("train_like", "heldout_long", "heldout_xlong"):
        points = rollouts(agent, env, EVAL_STREAM, seed, settings[setting], EPISODES)
        points["unseen"] = [key not in seen for key in points.key]  # not Series.isin, which mishandles these bytes
        zero = points[points.broke]
        rows.append(
            {
                "experiment": experiment,
                "method": name,
                "checkpoint": which,
                "seed": seed,
                "setting": setting,
                "points": len(points),
                "unseen_fraction": float(points.unseen.mean()),
                "zero_slack_points": len(zero),
                "zero_slack_unseen": _rate(zero.unseen),
                "zero_slack_spend": _rate(zero.spend),
                "zero_slack_spend_seen": _rate(zero.spend[~zero.unseen]),
                "zero_slack_spend_unseen": _rate(zero.spend[zero.unseen]),
                "slack_spent_seen": float(points.cost[~points.unseen].sum() / EPISODES),
                "slack_spent_unseen": float(points.cost[points.unseen].sum() / EPISODES),
                "uncued_check_early": _rate(points.verify[points.uncued & points.early]),
                "whole_episode_in_window": float(points.whole.mean()),
            }
        )
    return rows


def main() -> None:
    jobs = []
    for experiment, names in METHODS.items():
        exp = load_experiment(ROOT / "configs" / f"{experiment}.yaml")
        jobs += [(experiment, name, seed, which) for name in names for seed in exp.seeds for which in ("best", "final")]
    with mp.get_context("spawn").Pool(max(1, (mp.cpu_count() or 2) - 1)) as pool:
        rows = [row for chunk in pool.imap(job, jobs) for row in chunk]
    out = ROOT / "results" / "agent_coverage.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
