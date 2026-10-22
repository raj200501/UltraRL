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
    ref_names = {p.name for p in ref_dir.glob("*.pt")}
    cand_names = {p.name for p in cand_dir.glob("*.pt")}
    if not ref_names:
        print(f"no reference checkpoints under {ref_dir}")
        return 1
    missing, extra = sorted(ref_names - cand_names), sorted(cand_names - ref_names)
    mismatched = []
    for name in sorted(ref_names & cand_names):
        ref_agent, ref_meta = DQNAgent.load(ref_dir / name)
        agent, meta = DQNAgent.load(cand_dir / name)
        same = all(
            torch.equal(a, b)
            for a, b in zip(ref_agent.online.state_dict().values(), agent.online.state_dict().values(), strict=True)
        )
        if not (same and ref_meta["checkpoint_episode"] == meta["checkpoint_episode"]):
            mismatched.append(name)
    print(f"{len(ref_names & cand_names) - len(mismatched)}/{len(ref_names)} reference checkpoints reproduced exactly")
    for label, names in (("missing", missing), ("unexpected", extra), ("differs", mismatched)):
        for name in names:
            print(f"  {label}: {name}")
    return 1 if (missing or extra or mismatched) else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
