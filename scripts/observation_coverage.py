"""How often does each observation encoder produce feature vectors on held-out
lengths that never occur on training lengths?

    python scripts/observation_coverage.py   # writes results/observation_coverage.csv

Observations are collected from rollouts of the hand-coded policies (which act
identically under every encoder), so the comparison isolates the encoder.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ultrarl.config import EnvConfig
from ultrarl.env import ChainEnv
from ultrarl.observations import OBSERVATION_KINDS
from ultrarl.policies import build_heuristic
from ultrarl.rollout import CURVE_STREAM, episode_seed

POLICIES = ["random", "always_think", "verify_every_step", "cue_triggered", "budget_aware_1"]
SETTINGS = {"train_like": (3, 6), "heldout_long": (7, 9), "heldout_xlong": (10, 14)}
EPISODES = 3000


def collect(kind: str, depth_range: tuple[int, int], name: str) -> list[bytes]:
    env = ChainEnv(EnvConfig(observation=kind))
    keys = []
    for p, policy_name in enumerate(POLICIES):
        policy = build_heuristic(policy_name, seed=p)
        for i in range(EPISODES):
            obs = env.reset(seed=episode_seed(CURVE_STREAM, 100 + p, i, name), depth_range=depth_range)
            done = False
            while not done:
                keys.append(np.round(obs, 6).tobytes())
                obs, _, done, _ = env.step(policy(env.observable(), env.action_mask()))
    return keys


def main() -> None:
    rows = []
    for kind in OBSERVATION_KINDS:
        seen = set(collect(kind, SETTINGS["train_like"], "train_like"))
        for name in ("heldout_long", "heldout_xlong"):
            keys = collect(kind, SETTINGS[name], name)
            unseen = np.mean([k not in seen for k in keys])
            rows.append({"observation": kind, "setting": name, "unseen_fraction": float(unseen), "points": len(keys)})
            print(f"{kind:>9} {name:>13}: {unseen:.3f} of decision points have an unseen observation")
    out = Path("results/observation_coverage.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
