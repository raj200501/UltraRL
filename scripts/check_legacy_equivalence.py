"""Check the rewritten environment against the original implementation.

    python scripts/check_legacy_equivalence.py

Extracts the original environment (`src/envs.py` at commit 81ec8cb) from git
history, plays the same random action sequences in both implementations, and
compares action masks, rewards, termination, success and (non-terminal)
observations step by step for every reward scheme.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from ultrarl.config import EnvConfig
from ultrarl.env import ChainEnv

LEGACY_COMMIT = "81ec8cb"
RENAMED = {
    "hint_reliability": "cue_hit_rate",
    "hint_false_positive": "cue_false_positive_rate",
    "ma_hint_verify_bonus": "ma_cue_verify_bonus",
    "ma_low_hint_penalty": "ma_low_cue_penalty",
    "ma_opaque_penalty": "ma_gap_penalty",
}
DROPPED = {"use_risk_hints", "invalid_action_penalty"}  # always on / never triggered by masked agents


def load_legacy():
    def show(path: str) -> str:
        return subprocess.run(
            ["git", "show", f"{LEGACY_COMMIT}:{path}"], check=True, capture_output=True, text=True
        ).stdout

    tmp = Path(tempfile.mkdtemp())
    (tmp / "envs.py").write_text(show("src/envs.py"))
    spec = importlib.util.spec_from_file_location("legacy_envs", tmp / "envs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    config = yaml.safe_load(show("configs/default.yaml"))["env"]
    return module.ProcessReasoningEnv, config


def main(episodes: int = 3000) -> int:
    legacy_env_cls, legacy_cfg = load_legacy()
    steps = mismatches = 0
    for scheme in ("outcome_only", "naive_process", "monitorability_aware"):
        for budget_penalty in (True, False):
            old_cfg = dict(legacy_cfg, reward_scheme=scheme, budget_penalty=budget_penalty)
            new_cfg = {RENAMED.get(k, k): v for k, v in old_cfg.items() if k not in DROPPED}
            old, new = legacy_env_cls(old_cfg), ChainEnv(EnvConfig(**new_cfg))
            policy_rng = np.random.default_rng(1)
            for ep in range(episodes):
                depth_range = (3, 6) if ep % 2 else (7, 9)
                o1, _ = old.reset(seed=ep, depth_range=depth_range)
                o2 = new.reset(seed=ep, depth_range=depth_range)
                assert np.allclose(o1, o2) and old.fault_index == new.fault_index and old.risk_hints == new.cues
                done = False
                while not done:
                    mask = old.available_actions()
                    assert (mask == new.action_mask()).all()
                    valid = np.flatnonzero(mask)
                    weights = np.where(valid == 0, 3.0, 1.0)  # favour THINK so episodes reach the end
                    action = int(policy_rng.choice(valid, p=weights / weights.sum()))
                    o1, r1, d1, _ = old.step(action)
                    o2, r2, d2, _ = new.step(action)
                    steps += 1
                    # the terminal observation after COMMIT is unused by any learner
                    if not (np.isclose(r1, r2) and d1 == d2 and (d1 or np.allclose(o1, o2))):
                        mismatches += 1
                    done = d1
                assert old.success == new.success
    print(f"compared {steps} steps over {6 * episodes} episodes: {mismatches} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
