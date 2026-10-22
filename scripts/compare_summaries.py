"""Compare retrained results with the saved ones statistically, for hardware where
bitwise reproduction is not expected (see compare_checkpoints.py).

    python scripts/compare_summaries.py results/main results_repro/main

Two checks:

1. For every learned method and length setting, the retrained-minus-saved
   difference in success, paired by seed (both trees use the same evaluation
   episodes), with a 95% t-interval. On different hardware each seed follows a
   different training trajectory, so individual numbers move; the intervals show
   by how much.
2. For the main experiment, the paper's headline paired comparisons (the
   Holm-corrected family): whether each keeps its sign and whether its interval
   still excludes zero.

Neither is an equivalence test: with 10 seeds, shifts of about 0.1 in a mean can
go undetected.
"""

import sys
from pathlib import Path

import pandas as pd

from ultrarl.report import PRIMARY, key_numbers
from ultrarl.stats import t_interval

LENGTHS = ("train_like", "heldout_long", "heldout_xlong")


def differences(reference: Path, candidate: Path) -> pd.DataFrame:
    ref, new = (pd.read_csv(d / "eval_per_seed.csv") for d in (reference, candidate))
    learned = set(pd.read_csv(reference / "train_runs.csv").method)
    keys = ["setting", "method", "seed"]
    both = ref[keys + ["success"]].merge(new[keys + ["success"]], on=keys, suffixes=("_saved", "_retrained"))
    both = both[both.setting.isin(LENGTHS) & both.method.str.split("@").str[0].isin(learned)]
    rows = []
    for (setting, method), sub in both.groupby(["setting", "method"]):
        diff = (sub.success_retrained - sub.success_saved).to_numpy()
        low, high = t_interval(diff)
        means = {"saved": sub.success_saved.mean(), "retrained": sub.success_retrained.mean()}
        rows.append({"setting": setting, "method": method, **means, "diff": diff.mean(), "lo": low, "hi": high})
    return pd.DataFrame(rows)


def headline(reference: Path, candidate: Path) -> pd.DataFrame:
    frames = {}
    for label, path in (("saved", reference), ("retrained", candidate)):
        k = key_numbers(pd.read_csv(path / "eval_per_seed.csv"), PRIMARY)
        frames[label] = k.set_index(["a", "b", "setting"])[["mean", "lo", "hi"]]
    out = frames["saved"].join(frames["retrained"], lsuffix="_saved", rsuffix="_retrained").reset_index()
    out["same_sign"] = (out.mean_saved > 0) == (out.mean_retrained > 0)
    out["resolved_saved"] = (out.lo_saved > 0) | (out.hi_saved < 0)
    out["resolved_retrained"] = (out.lo_retrained > 0) | (out.hi_retrained < 0)
    return out


def main(reference: str, candidate: str) -> int:
    ref, cand = Path(reference), Path(candidate)
    frame = differences(ref, cand)
    if frame.empty:
        print("nothing to compare")
        return 1
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    excluded = int(((frame.lo > 0) | (frame.hi < 0)).sum())
    print(f"{excluded} of {len(frame)} intervals exclude zero (about 5% expected by chance)")
    names = set(pd.read_csv(ref / "eval_per_seed.csv").method)
    if {m for a, b, _ in PRIMARY for m in (a, b)} <= names:
        h = headline(ref, cand)
        print(h.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        kept = int((h.same_sign & (h.resolved_saved == h.resolved_retrained)).sum())
        print(f"{kept} of {len(h)} headline comparisons keep their sign and resolution")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
