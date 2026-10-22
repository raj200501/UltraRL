"""What do learned agents value right after a check reports UNSAFE?

    python scripts/q_probe.py   # writes results/q_probe.csv

Each checkpoint is rolled out greedily on training-length evaluation episodes.

* At every decision point where the last check reported UNSAFE, both REVISE
  and THINK are available and the budget still allows the repair (undoing and
  redoing the step leaves enough to finish the chain), we record the greedy action, the learned values
  Q(REVISE) and Q(THINK), and their true discounted returns under the same
  greedy policy, obtained by playing each action out from a copy of the
  environment (episodes are deterministic once reset, so this is exact).
* As calibration baselines we record the learned value of the greedy action and
  the discounted return that actually followed (a) at the first decision point
  of every episode, where the hidden state follows the task prior and the
  agent's own history cannot create aliasing, and (b) at every decision point where the running value is correct and
  no caught fault is waiting to be repaired ("on track"; this conditions on
  hidden state, so it is only a secondary baseline).
* We also record how often an observation on the THINK branch, while the running
  value is still wrong, also occurs on track in the same checkpoint's rollouts.

Every statistic is a mean over decision points within a checkpoint; the report
averages it over seeds.
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
METHODS = {
    "main": ["outcome_inv", "naive_inv", "ma_inv"],
    "followups": [
        "ma_memory",
        "ma_repair",
        "ma_memory_repair",
        "outcome_pending",
        "naive_pending",
        "ma_pending",
        "outcome_history",
        "naive_history",
        "ma_history",
        "outcome_history3",
        "naive_history3",
        "ma_history3",
    ],  # fmt: skip
}
SETTING = "train_like"
EPISODES = 500


def key(obs: np.ndarray) -> bytes:
    return np.round(obs, 6).tobytes()


def greedy_return(agent: DQNAgent, env: ChainEnv, first_action: int, gamma: float, seen: list | None = None) -> float:
    obs, reward, done, _ = env.step(first_action)
    total, discount = reward, gamma
    while not done:
        if seen is not None and not env.belief_valid:
            seen.append(key(obs))
        obs, reward, done, _ = env.step(agent.act(obs, env.action_mask()))
        total += discount * reward
        discount *= gamma
    return total


def discounted_suffix_sums(rewards: list[float], gamma: float) -> np.ndarray:
    out, running = np.zeros(len(rewards)), 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def probe(agent: DQNAgent, env: ChainEnv, seed: int, depth_range: tuple[int, int], gamma: float) -> dict[str, float]:
    q_think, q_revise, g_think, g_revise, think = [], [], [], [], []
    q_track, g_track, q_start, g_start = [], [], [], []
    on_track, past_fault, unaffordable = set(), [], 0
    repair_cost = env.cfg.revise_cost + env.cfg.think_cost
    for i in range(EPISODES):
        obs = env.reset(seed=episode_seed(EVAL_STREAM, seed, i, SETTING), depth_range=depth_range)
        done, rewards, track_points = False, [], []
        first_value = None
        while not done:
            mask = env.action_mask()
            q = agent.q_values(obs)
            if first_value is None:
                first_value = float(np.max(np.where(mask, q, -np.inf)))
            pending = env.fault_located and not env.repaired
            if env.belief_valid and not pending:
                on_track.add(key(obs))
                track_points.append((len(rewards), float(np.max(np.where(mask, q, -np.inf)))))
            o = env.observable()
            after_unsafe = o.last_verify == VerifyStatus.UNSAFE and mask[Action.REVISE] and mask[Action.THINK]
            if after_unsafe and o.slack < repair_cost:  # the repair is unaffordable: both actions lose
                unaffordable += 1
            elif after_unsafe:
                q_think.append(q[Action.THINK])
                q_revise.append(q[Action.REVISE])
                g_think.append(greedy_return(agent, copy.deepcopy(env), Action.THINK, gamma, past_fault))
                g_revise.append(greedy_return(agent, copy.deepcopy(env), Action.REVISE, gamma))
                think.append(agent.act(obs, mask) == Action.THINK)
            obs, reward, done, _ = env.step(agent.act(obs, mask))
            rewards.append(reward)
        returns = discounted_suffix_sums(rewards, gamma)
        q_start.append(first_value)
        g_start.append(returns[0])
        for t, value in track_points:
            q_track.append(value)
            g_track.append(returns[t])
    stats = {"unsafe_points": len(think), "unaffordable_points": unaffordable}
    stats.update(q_on_track=float(np.mean(q_track)), return_on_track=float(np.mean(g_track)))
    stats.update(q_start=float(np.mean(q_start)), return_start=float(np.mean(g_start)))
    if think:
        stats.update(
            {
                "think_rate": float(np.mean(think)),
                "q_think": float(np.mean(q_think)),
                "q_revise": float(np.mean(q_revise)),
                "return_think": float(np.mean(g_think)),
                "return_revise": float(np.mean(g_revise)),
                "aliased_rate": float(np.mean([k in on_track for k in past_fault])) if past_fault else float("nan"),
            }
        )
    return stats


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
                    stats = probe(agent, env, seed, setting.depth_range, exp.method_train(method).gamma)
                    ids = {"experiment": experiment, "method": method.name, "checkpoint": which, "seed": seed}
                    rows.append({**ids, **stats})
    out = ROOT / "results" / "q_probe.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
