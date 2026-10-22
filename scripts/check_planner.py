"""Does the planner's restriction (never go back to check an earlier step) bind?

    python scripts/check_planner.py [max_depth]   # default 7; depth 8 takes minutes

Compares the planner's value with an exhaustive search over every
history-dependent policy for each evaluation environment of the main
experiment and every depth from 2 to ``max_depth``.
"""

import sys
from pathlib import Path

from ultrarl.config import load_experiment
from ultrarl.optimal import OptimalPlanner, exhaustive_success, initial_slack

ROOT = Path(__file__).resolve().parents[1]


def main(max_depth: int = 7) -> None:
    exp = load_experiment(ROOT / "configs" / "main.yaml")
    method = next(m for m in exp.methods if m.name == "optimal")
    largest = 0.0
    for setting in exp.eval_settings:
        cfg = exp.method_env(method, setting)
        for depth in range(2, max_depth + 1):
            gap = exhaustive_success(cfg, depth) - OptimalPlanner(cfg, depth).value(0, initial_slack(cfg, depth))
            largest = max(largest, abs(gap))
            print(f"{setting.name:15s} depth {depth}: exhaustive - planner = {gap:+.1e}", flush=True)
    print(f"largest difference: {largest:.1e}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
