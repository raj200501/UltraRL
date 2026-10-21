"""Observation encoders.

Two encoders expose the *same* information content relevant to the task but
differ in how position is represented:

``absolute`` (10-d)
    progress ``t / D_max``, budget fraction ``b / B``, depth ``d / D_max``,
    normalised budget slack, unchecked gap ``g / D_max``, cue, revised flag and a
    one-hot of the last verification result. Position and depth features take
    values on longer chains that never occur on training-length chains.

``invariant`` (9-d)
    at-end flag ``[t = d]``, at-start flag ``[t = 0]``, clipped budget slack
    ``clip(b - (d - t), -10, 10) / 10``, clipped gap ``min(g, 3) / 3``, cue,
    revised flag and the verification one-hot. None of these features depends
    on the absolute chain length, so longer chains produce no new feature values
    (as long as the budget slack stays in range).

Budget slack ``b - (d - t)`` is the number of non-THINK actions the agent can
still afford while finishing the chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ultrarl.env import Observable

OBSERVATION_KINDS = ("absolute", "invariant")
_DIMS = {"absolute": 10, "invariant": 9}
SLACK_SCALE = 10.0
GAP_CAP = 3


def observation_dim(kind: str) -> int:
    try:
        return _DIMS[kind]
    except KeyError:
        raise ValueError(f"unknown observation kind {kind!r}; expected one of {OBSERVATION_KINDS}") from None


def _verify_one_hot(o: Observable) -> list[float]:
    one_hot = [0.0, 0.0, 0.0]
    one_hot[int(o.last_verify)] = 1.0
    return one_hot


def encode_observation(kind: str, o: Observable) -> np.ndarray:
    slack = o.remaining_budget - max(0, o.depth - o.step)
    if kind == "absolute":
        features = [
            o.step / max(1, o.max_depth),
            o.remaining_budget / max(1, o.max_budget),
            o.depth / max(1, o.max_depth),
            float(np.clip(0.5 + slack / max(1, 2 * o.max_budget), 0.0, 1.0)),
            o.unchecked_gap / max(1, o.max_depth),
            float(o.cue),
            float(o.revised_recently),
            *_verify_one_hot(o),
        ]
    elif kind == "invariant":
        features = [
            float(o.step >= o.depth),
            float(o.step == 0),
            float(np.clip(slack, -SLACK_SCALE, SLACK_SCALE) / SLACK_SCALE),
            min(o.unchecked_gap, GAP_CAP) / GAP_CAP,
            float(o.cue),
            float(o.revised_recently),
            *_verify_one_hot(o),
        ]
    else:
        raise ValueError(f"unknown observation kind {kind!r}; expected one of {OBSERVATION_KINDS}")
    return np.asarray(features, dtype=np.float32)
