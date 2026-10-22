"""Compare retrained results with the saved ones statistically, for hardware where
bitwise reproduction is not expected (see compare_checkpoints.py).

    python scripts/compare_summaries.py results/main results_repro/main

For every learned method and length setting, compares per-seed success between
the two trees with Welch's t-test. On different hardware each seed follows a
different training trajectory, so individual numbers move; the comparison asks
whether the means are consistent. About 5% of rows are expected to differ at
p < 0.05 by chance alone.
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

LENGTHS = ("train_like", "heldout_long", "heldout_xlong")


def compare(reference: Path, candidate: Path) -> pd.DataFrame:
    ref, new = (pd.read_csv(d / "eval_per_seed.csv") for d in (reference, candidate))
    learned = set(pd.read_csv(reference / "train_runs.csv").method)
    rows = []
    for (setting, method), a in ref[ref.setting.isin(LENGTHS)].groupby(["setting", "method"]):
        if method.split("@")[0] not in learned:
            continue  # hand-coded policies are deterministic given the episode seeds
        b = new[(new.setting == setting) & (new.method == method)]
        if b.empty:
            continue
        p = stats.ttest_ind(a.success, b.success, equal_var=False).pvalue
        rows.append({"setting": setting, "method": method, "saved": a.success.mean(), "retrained": b.success.mean(),
                     "p": p})  # fmt: skip
    return pd.DataFrame(rows)


def main(reference: str, candidate: str) -> int:
    frame = compare(Path(reference), Path(candidate))
    if frame.empty:
        print("nothing to compare")
        return 1
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"{int((frame.p < 0.05).sum())} of {len(frame)} rows differ at p < 0.05 (about 5% expected by chance)")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
