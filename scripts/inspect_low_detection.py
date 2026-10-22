"""Where do low-detection checkpoints check?

    python scripts/inspect_low_detection.py   # writes results/low_detection_inspection.csv

Takes the last checkpoints of the main experiment's outcome-only agents with
invariant observations whose training-length fault detection is below the
low-detection threshold, rolls each out greedily on training-length evaluation
episodes, and records how many checks it makes per episode, the share of checks
that land on the final step, and its check rate at cued and un-cued steps.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ultrarl.config import load_experiment
from ultrarl.dqn import DQNAgent
from ultrarl.env import Action, ChainEnv
from ultrarl.report import LOW_DETECTION
from ultrarl.rollout import EVAL_STREAM, episode_seed
from ultrarl.train import checkpoint_path

ROOT = Path(__file__).resolve().parents[1]
METHOD, SETTING, EPISODES = "outcome_inv", "train_like", 300


def inspect(agent: DQNAgent, env: ChainEnv, seed: int, depth_range: tuple[int, int]) -> dict[str, float]:
    checks, final, cued = [], [], {0: [], 1: []}
    for i in range(EPISODES):
        obs = env.reset(seed=episode_seed(EVAL_STREAM, seed, i, SETTING), depth_range=depth_range)
        done, count = False, 0
        while not done:
            mask, o = env.action_mask(), env.observable()
            action = agent.act(obs, mask)
            if o.step > 0 and not o.step_checked and mask[Action.VERIFY]:
                cued[o.cue].append(action == Action.VERIFY)
            if action == Action.VERIFY:
                count += 1
                final.append(o.step == o.depth)
            obs, _, done, _ = env.step(action)
        checks.append(count)
    return {
        "checks_min": int(min(checks)),
        "checks_max": int(max(checks)),
        "checks_mean": float(np.mean(checks)),
        "final_step_share": float(np.mean(final)) if final else float("nan"),
        "check_rate_cue0": float(np.mean(cued[0])),
        "check_rate_cue1": float(np.mean(cued[1])),
    }


def main() -> None:
    torch.set_num_threads(1)
    exp = load_experiment(ROOT / "configs" / "main.yaml")
    method = next(m for m in exp.methods if m.name == METHOD)
    setting = next(s for s in exp.eval_settings if s.name == SETTING)
    per_seed = pd.read_csv(ROOT / "results" / "main" / "eval_per_seed.csv")
    last = per_seed[(per_seed.method == f"{METHOD}@final") & (per_seed.setting == SETTING)]
    rows = []
    for _, r in last[last.fault_detected < LOW_DETECTION].iterrows():
        seed = int(r.seed)
        agent, _ = DQNAgent.load(checkpoint_path(ROOT / "results" / "main", method, seed, "final"))
        env = ChainEnv(exp.method_env(method))
        rows.append(
            {"seed": seed, "fault_detected": r.fault_detected, **inspect(agent, env, seed, setting.depth_range)}
        )
    out = ROOT / "results" / "low_detection_inspection.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
