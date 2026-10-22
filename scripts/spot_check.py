"""Retrain one run from each experiment and check it bitwise against the saved one.

    python scripts/spot_check.py [--out DIR]   # about 5 minutes on 6 cores, 9 on 4

A quick alternative to ``make reproduce`` (which retrains everything): one
(method, seed) per experiment, chosen to cover different rewards and encoders.
Bitwise identity needs the pinned package versions and the CPU instruction set
the results were produced with (see compare_checkpoints.py).
"""

import argparse
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

import torch

from ultrarl.config import load_experiment
from ultrarl.dqn import DQNAgent
from ultrarl.run import platform_info
from ultrarl.train import checkpoint_path, curve_path, train_run

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("main", "outcome_abs", 0),
    ("ablations", "no_cue_terms", 1),
    ("sensitivity", "ma_inv_polyak", 2),
    ("followups", "ma_history3", 3),
    ("encoders", "ma_abs_nopos", 4),
    ("process", "naive_step10", 5),
]


def check(job: tuple[str, str, int, str]) -> tuple[str, bool]:
    experiment, name, seed, out = job
    exp = load_experiment(ROOT / "configs" / f"{experiment}.yaml")
    method = next(m for m in exp.methods if m.name == name)
    saved, fresh = ROOT / "results" / experiment, Path(out) / experiment
    train_run(exp, method, seed, fresh)
    same = curve_path(saved, method, seed).read_text() == curve_path(fresh, method, seed).read_text()
    for which in ("best", "final"):
        (a, meta_a), (b, meta_b) = (DQNAgent.load(checkpoint_path(d, method, seed, which)) for d in (saved, fresh))
        weights = zip(a.online.state_dict().values(), b.online.state_dict().values(), strict=True)
        same &= all(torch.equal(x, y) for x, y in weights)
        same &= meta_a["checkpoint_episode"] == meta_b["checkpoint_episode"]
    return f"{experiment}/{name} seed {seed}", same


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="where to retrain (default: a temporary directory)")
    args = parser.parse_args()
    out = args.out or tempfile.mkdtemp(prefix="ultrarl_spot_check_")
    with mp.get_context("spawn").Pool(min(len(RUNS), max(1, mp.cpu_count() or 1))) as pool:
        results = pool.map(check, [(*run, out) for run in RUNS])
    for label, same in results:
        print(f"{'identical' if same else 'DIFFERS  '}  {label}")
    print(f"this platform: {platform_info()}")
    return 0 if all(same for _, same in results) else 1


if __name__ == "__main__":
    sys.exit(main())
