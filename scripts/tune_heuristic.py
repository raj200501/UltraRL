"""Pick the budget-aware heuristic's margin on *training* lengths only.

    python scripts/tune_heuristic.py

Uses the learning-curve RNG stream (disjoint from the evaluation stream), so the
choice cannot leak information from the reported evaluation episodes.
"""

from pathlib import Path

from ultrarl.config import load_experiment
from ultrarl.policies import heuristic_act
from ultrarl.rollout import CURVE_STREAM, evaluate

EXPERIMENT = load_experiment(Path(__file__).resolve().parents[1] / "configs" / "main.yaml")


def main() -> None:
    train = next(s for s in EXPERIMENT.eval_settings if s.name == "train_like")
    scores = {}
    for margin in range(7):
        act = heuristic_act(f"budget_aware_{margin}", 0)
        scores[margin] = evaluate(act, EXPERIMENT.env, train, run_seed=0, episodes=2000, stream=CURVE_STREAM)["success"]
        print(f"margin={margin}: train-length success {scores[margin]:.3f}")
    print(f"best margin: {max(scores, key=scores.get)}")


if __name__ == "__main__":
    main()
