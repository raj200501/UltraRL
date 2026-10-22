"""Check that two results trees contain bitwise-identical checkpoints.

    python scripts/compare_checkpoints.py results/main results_repro/main

Used by `make reproduce` to verify that retraining from scratch reproduces the
saved agents exactly. Training is deterministic on CPU given a seed, but only for
the same package versions and CPU instruction set: torch's CPU kernels round
differently with AVX-512, AVX2 or no vector extensions, and a different rounding
changes the whole trajectory. On other hardware, compare the results
statistically instead (scripts/compare_summaries.py).
"""

import json
import sys
from pathlib import Path

import torch

from ultrarl.dqn import DQNAgent
from ultrarl.run import platform_info


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
    if mismatched:
        info = Path(reference) / "run_info.json"
        recorded = json.loads(info.read_text()).get("platform", "not recorded") if info.exists() else "not recorded"
        print(f"  reference platform: {recorded}\n  this platform:      {platform_info()}")
        print("  bitwise identity needs the same package versions and CPU capability; see compare_summaries.py")
    return 1 if (missing or extra or mismatched) else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
