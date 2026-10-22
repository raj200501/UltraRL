import math

from ultrarl.rollout import CURVE_STREAM, EVAL_STREAM, TRAIN_STREAM, aggregate, episode_seed


def test_rng_streams_are_disjoint():
    # Regression: training and evaluation episodes must never share seeds.
    train = {tuple(episode_seed(TRAIN_STREAM, s, e)) for s in range(10) for e in range(1, 2001)}
    for stream in (CURVE_STREAM, EVAL_STREAM):
        for setting in ("train_like", "heldout_long"):
            other = {tuple(episode_seed(stream, s, e, setting)) for s in range(10) for e in range(500)}
            assert train.isdisjoint(other)


def test_aggregate_pools_ratio_metrics():
    nan = float("nan")
    base = dict(
        num_actions=1.0,
        incomplete_commit=0.0,
        voluntary_incomplete_commit=0.0,
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
            "repair_success": 1.0,
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
            "repair_success": nan,
            "fresh_opps_cue1": 2.0,
        },
    ]
    m = aggregate(rows)
    assert m["success"] == 0.5
    assert m["fault_detected"] == 1.0  # NaN (no fault) episodes are excluded
    assert m["verify_precision"] == 0.25  # pooled: 1 TP / 4 verifications
    assert m["verify_rate_cue1"] == 0.5
    assert math.isnan(m["verify_rate_cue0"])  # no opportunities
    assert m["clean_success"] == 0.0
