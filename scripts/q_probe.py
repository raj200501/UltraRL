"""What do learned agents value right after a check reports UNSAFE?

    python scripts/q_probe.py   # writes results/q_probe.csv

Each checkpoint is rolled out greedily on training-length evaluation episodes.
At every decision point where the last check reported UNSAFE and REVISE is
available, we record the greedy action, the learned gap Q(REVISE) - Q(THINK),
and the true gap under the same policy: the discounted return of REVISE minus
that of THINK, each followed by greedy play from a copy of the environment
(episodes are deterministic once reset, so this is exact). It also records
how often an observation on the THINK branch, where the running value is wrong,
also occurs at a decision point of the same checkpoint where it is right.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ultrarl.config import load_experiment
from ultrarl.dqn import DQNAgent
from ultrarl.env import Action, ChainEnv, VerifyStatus
from ultrarl.rollout import EVAL_STREAM, episode_seed
from ultrarl.train import checkpoint_path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"main": ["outcome_inv", "naive_inv", "ma_inv"], "followups": ["ma_memory", "ma_repair", "ma_memory_repair"]}
SETTING = "train_like"
EPISODES = 500


def key(obs: np.ndarray) -> bytes:
    return np.round(obs, 6).tobytes()


def greedy_return(agent: DQNAgent, env: ChainEnv, first_action: int, gamma: float, seen: list | None = None) -> float:
    obs, reward, done, _ = env.step(first_action)
    total, discount = reward, gamma
    while not done:
        if seen is not None:
            seen.append(key(obs))
        obs, reward, done, _ = env.step(agent.act(obs, env.action_mask()))
        total += discount * reward
        discount *= gamma
    return total


def probe(agent: DQNAgent, env: ChainEnv, seed: int, depth_range: tuple[int, int], gamma: float) -> dict[str, float]:
    q_think, q_revise, g_think, g_revise, think = [], [], [], [], []
    on_track, past_fault = set(), []
    for i in range(EPISODES):
        obs = env.reset(seed=episode_seed(EVAL_STREAM, seed, i, SETTING), depth_range=depth_range)
        done = False
        while not done:
            mask = env.action_mask()
            if env.belief_valid:
                on_track.add(key(obs))
            if env.observable().last_verify == VerifyStatus.UNSAFE and mask[Action.REVISE] and mask[Action.THINK]:
                q = agent.q_values(obs)
                q_think.append(q[Action.THINK])
                q_revise.append(q[Action.REVISE])
                g_think.append(greedy_return(agent, copy.deepcopy(env), Action.THINK, gamma, past_fault))
                g_revise.append(greedy_return(agent, copy.deepcopy(env), Action.REVISE, gamma))
                think.append(agent.act(obs, mask) == Action.THINK)
            obs, _, done, _ = env.step(agent.act(obs, mask))
    if not think:
        return {"unsafe_points": 0}
    q_think, q_revise, g_think, g_revise = map(np.array, (q_think, q_revise, g_think, g_revise))
    return {
        "unsafe_points": len(think),
        "think_rate": float(np.mean(think)),
        "q_think": float(np.mean(q_think)),
        "q_revise": float(np.mean(q_revise)),
        "return_think": float(np.mean(g_think)),
        "return_revise": float(np.mean(g_revise)),
        # medians: a few far-off states (e.g. at the budget limit) dominate the mean
        "q_gap": float(np.median(q_revise - q_think)),
        "true_gap": float(np.median(g_revise - g_think)),
        "aliased_rate": float(np.mean([k in on_track for k in past_fault])),
    }


def main() -> None:
    torch.set_num_threads(1)
    rows = []
    for experiment, names in METHODS.items():
        exp = load_experiment(ROOT / "configs" / f"{experiment}.yaml")
        setting = next(s for s in exp.eval_settings if s.name == SETTING)
        for method in (m for m in exp.methods if m.name in names):
            env = ChainEnv(exp.method_env(method, setting))
            for seed in exp.seeds:
                for which in ("best", "final"):
                    agent, _ = DQNAgent.load(checkpoint_path(ROOT / "results" / experiment, method, seed, which))
                    stats = probe(agent, env, seed, setting.depth_range, exp.train.gamma)
                    ids = {"experiment": experiment, "method": method.name, "checkpoint": which, "seed": seed}
                    rows.append({**ids, **stats})
    out = ROOT / "results" / "q_probe.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
