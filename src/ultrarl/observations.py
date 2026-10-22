"""Observation encoders.

All encoders expose the local information a hand-coded checking policy needs
(cue, budget slack, unchecked gap, last check result, revised flag). They differ
in whether, and how, they reveal position and chain length:

``absolute`` (10-d), the original study's encoder
    progress ``t / D_max``, budget fraction ``b / B``, depth ``d / D_max``,
    normalised slack ``clip(0.5 + slack / 2B, 0, 1)``, unchecked gap ``g / D_max``,
    cue, revised flag and a one-hot of the last verification result. Several of
    these depend on the chain length (``B = d + budget_extra``), so longer chains
    produce feature values never seen on training-length chains.

``absolute_nopos`` (8-d), a control
    ``absolute`` without the two position features ``t / D_max`` and
    ``d / D_max``. The remaining length-dependent normalisations are kept, which
    separates the effect of position features from that of the other features.

``invariant`` (9-d)
    at-end flag ``[t = d]``, at-start flag ``[t = 0]``, clipped budget slack
    ``clip(b - (d - t), -10, 10) / 10``, clipped gap ``min(g, 3) / 3``, cue,
    revised flag and the verification one-hot. It drops absolute position and
    length, so the number of remaining steps cannot be recovered, and longer
    chains rarely produce new observations (as long as the slack stays in range).

``ratio`` (10-d)
    the invariant features plus slack per remaining step,
    ``clip(slack / max(1, d - t), 0, 2) / 2``: how thinly the remaining slack must
    be spread. This exposes remaining length in a bounded form (it reveals ``d``
    at ``t = 0``), so unlike ``invariant`` it takes unseen values on longer chains.

``memory`` (10-d)
    the invariant features plus a one-bit memory of whether an UNSAFE check has
    been observed in this episode. Without it, a genuine repair (UNSAFE, REVISE,
    THINK) and a no-op REVISE+THINK after a SAFE check produce identical
    observations.

Budget slack ``b - (d - t)`` is the number of non-THINK actions the agent can
still afford while finishing the chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ultrarl.env import Observable

OBSERVATION_KINDS = ("absolute", "absolute_nopos", "invariant", "ratio", "memory")
_DIMS = {"absolute": 10, "absolute_nopos": 8, "invariant": 9, "ratio": 10, "memory": 10}
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
    if kind in ("absolute", "absolute_nopos"):
        position = [o.step / max(1, o.max_depth), o.depth / max(1, o.max_depth)]
        budget = [
            o.remaining_budget / max(1, o.max_budget),
            float(np.clip(0.5 + slack / max(1, 2 * o.max_budget), 0.0, 1.0)),
        ]
        rest = [o.unchecked_gap / max(1, o.max_depth), float(o.cue), float(o.revised_recently), *_verify_one_hot(o)]
        # "absolute" keeps the original study's feature order
        features = [position[0], budget[0], position[1], budget[1], *rest] if kind == "absolute" else [*budget, *rest]
    elif kind in ("invariant", "ratio", "memory"):
        features = [
            float(o.step >= o.depth),
            float(o.step == 0),
            float(np.clip(slack, -SLACK_SCALE, SLACK_SCALE) / SLACK_SCALE),
            min(o.unchecked_gap, GAP_CAP) / GAP_CAP,
            float(o.cue),
            float(o.revised_recently),
            *_verify_one_hot(o),
        ]
        if kind == "ratio":
            features.append(float(np.clip(slack / max(1, o.depth - o.step), 0.0, 2.0) / 2.0))
        elif kind == "memory":
            features.append(float(o.fault_located))
    else:
        raise ValueError(f"unknown observation kind {kind!r}; expected one of {OBSERVATION_KINDS}")
    return np.asarray(features, dtype=np.float32)
