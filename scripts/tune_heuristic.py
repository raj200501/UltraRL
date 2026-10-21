"""Pick the budget-aware heuristic's margin on *training* lengths only.

    python scripts/tune_heuristic.py

Uses the learning-curve RNG stream (disjoint from the evaluation stream), so the
choice cannot leak information from the reported evaluation episodes.
"""

from ultrarl.config import EnvConfig, EvalSetting
from ultrarl.policies import build_heuristic
from ultrarl.rollout import CURVE_STREAM, evaluate


def main() -> None:
    train = EvalSetting("train_like", (3, 6))
    scores = {}
    for margin in range(7):
        policy = build_heuristic(f"budget_aware_{margin}", 0)
        act = lambda _obs, observable, mask, p=policy: p(observable, mask)  # noqa: E731
        scores[margin] = evaluate(act, EnvConfig(), train, run_seed=0, episodes=2000, stream=CURVE_STREAM)["success"]
        print(f"margin={margin}: train-length success {scores[margin]:.3f}")
    print(f"best margin: {max(scores, key=scores.get)}")


if __name__ == "__main__":
    main()
