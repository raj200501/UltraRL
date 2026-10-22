"""Check that two results trees contain bitwise-identical checkpoints.

    python scripts/compare_checkpoints.py results/main results_repro/main

Used by `make reproduce` to verify that retraining from scratch reproduces the
saved agents exactly (training is deterministic on CPU given a seed).
"""

import sys
from pathlib import Path

import torch

from ultrarl.dqn import DQNAgent


def main(reference: str, candidate: str) -> int:
    ref_dir, cand_dir = Path(reference) / "checkpoints", Path(candidate) / "checkpoints"
    paths = sorted(cand_dir.glob("*.pt"))
    if not paths:
        print(f"no checkpoints under {cand_dir}")
        return 1
    mismatched = []
    for path in paths:
        ref_agent, ref_meta = DQNAgent.load(ref_dir / path.name)
        agent, meta = DQNAgent.load(path)
        same = all(
            torch.equal(a, b)
            for a, b in zip(ref_agent.online.state_dict().values(), agent.online.state_dict().values(), strict=True)
        )
        same = same and ref_meta["checkpoint_episode"] == meta["checkpoint_episode"]
        if not same:
            mismatched.append(path.name)
    print(f"{len(paths) - len(mismatched)}/{len(paths)} checkpoints identical")
    for name in mismatched:
        print(f"  differs: {name}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
