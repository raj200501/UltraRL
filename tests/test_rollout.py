import math

import numpy as np

from ultrarl.config import EnvConfig, EvalSetting
from ultrarl.env import ChainEnv
from ultrarl.rollout import CURVE_STREAM, EVAL_STREAM, TRAIN_STREAM, aggregate, episode_seed, evaluate


def test_streams_produce_independent_random_states():
    # Seeds for different streams must yield different generator states. This
    # guarantees independent sampling, not distinct episodes: on training lengths
    # identical (depth, fault, cue) configurations recur by chance, by design.
    states = set()
    for stream in (TRAIN_STREAM, CURVE_STREAM, EVAL_STREAM):
        for seed in range(5):
            for episode in range(50):
                state = np.random.SeedSequence(episode_seed(stream, seed, episode, "train_like")).generate_state(4)
                states.add(tuple(state))
    assert len(states) == 3 * 5 * 50


def test_evaluate_uses_the_eval_stream(monkeypatch):
    seen = []
    original = ChainEnv.reset

    def recording_reset(self, seed=None, **kwargs):
        seen.append(tuple(seed))
        return original(self, seed=seed, **kwargs)

    monkeypatch.setattr(ChainEnv, "reset", recording_reset)
    evaluate(lambda o, ob, m: 3, EnvConfig(), EvalSetting("train_like", (3, 6)), run_seed=4, episodes=3)
    assert seen == [tuple(episode_seed(EVAL_STREAM, 4, i, "train_like")) for i in range(3)]
    assert all(s[0] == EVAL_STREAM for s in seen)


def test_aggregate_pools_ratio_metrics():
    nan = float("nan")
    base = dict(
        num_actions=1.0,
        incomplete_commit=0.0,
        voluntary_incomplete_commit=0.0,
        commit_at_start=0.0,
        nonrepair_revise_count=0.0,
        cost_spent=1.0,
        cost_adjusted_accuracy=0.0,
        revise_count=0.0,
        depth=3.0,
        fresh_opps_cue0=0.0,
        fresh_verifies_cue0=0.0,
        fresh_opps_cue1=0.0,
        fresh_verifies_cue1=0.0,
    )
    rows = [
        {
            **base,
            "success": 1.0,
            "return": 1.0,
            "verify_count": 3.0,
            "verify_tp": 1.0,
            "fault_present": 1.0,
            "fault_detected": 1.0,
            "success_given_fault": 1.0,
            "success_given_detected": 1.0,
            "fresh_opps_cue1": 2.0,
            "fresh_verifies_cue1": 2.0,
        },
        {
            **base,
            "success": 0.0,
            "return": -1.0,
            "verify_count": 1.0,
            "verify_tp": 0.0,
            "fault_present": 0.0,
            "fault_detected": nan,
            "success_given_fault": nan,
            "success_given_detected": nan,
            "fresh_opps_cue1": 2.0,
        },
    ]
    m = aggregate(rows)
    assert m["success"] == 0.5
    assert m["fault_detected"] == 1.0  # NaN (no fault) episodes are excluded
    assert m["success_given_detected"] == 1.0
    assert m["verify_precision"] == 0.25  # pooled: 1 TP / 4 verifications
    assert m["verify_rate_cue1"] == 0.5  # pooled over decision points
    assert math.isnan(m["verify_rate_cue0"])  # no opportunities
    assert m["clean_success"] == 0.0
