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

``pending`` (10-d)
    the invariant features plus a bit that is set by an UNSAFE check and cleared
    once the located step has been correctly re-executed. Without it, continuing
    past a caught fault leads to the same observations as being on track; the
    ``memory`` bit does not separate these either, because it stays set after a
    repair. An agent that remembers its own checks and undos could compute it.

``history`` (49-d)
    the invariant features plus a one-hot code for each of the last 8 actions
    (THINK, VERIFY with a SAFE or UNSAFE result, REVISE), oldest first and
    zero-padded (COMMIT has a code too, but ends the episode, so it never
    appears). Often enough to infer a pending repair, but an UNSAFE result can
    leave the window while the repair is still pending, and the agent has to
    learn to use it.

``history3`` (24-d), a control
    ``history`` with only the last 3 actions. Enough to see a caught fault until
    the repair or for two steps after skipping it, but it reveals little about
    position or progress (only through padding in the first three steps), and
    windows this short all occur on training-length chains.

Budget slack ``b - (d - t)`` is the number of non-THINK actions the agent can
still afford while finishing the chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ultrarl.env import Observable

OBSERVATION_KINDS = ("absolute", "absolute_nopos", "invariant", "ratio", "memory", "pending", "history", "history3")
HISTORY_STEPS, HISTORY_CODES = 8, 5  # must match ultrarl.env.HISTORY_LENGTH and the THINK..COMMIT codes
_WINDOWS = {"history": HISTORY_STEPS, "history3": 3}
_DIMS = {"absolute": 10, "absolute_nopos": 8, "invariant": 9, "ratio": 10, "memory": 10, "pending": 10,
         **{kind: 9 + steps * HISTORY_CODES for kind, steps in _WINDOWS.items()}}  # fmt: skip
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
    slack = o.slack
    if kind in ("absolute", "absolute_nopos"):
        position = [o.step / max(1, o.max_depth), o.depth / max(1, o.max_depth)]
        budget = [
            o.remaining_budget / max(1, o.max_budget),
            float(np.clip(0.5 + slack / max(1, 2 * o.max_budget), 0.0, 1.0)),
        ]
        rest = [o.unchecked_gap / max(1, o.max_depth), float(o.cue), float(o.revised_recently), *_verify_one_hot(o)]
        # "absolute" keeps the original study's feature order
        features = [position[0], budget[0], position[1], budget[1], *rest] if kind == "absolute" else [*budget, *rest]
    elif kind in ("invariant", "ratio", "memory", "pending", *_WINDOWS):
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
        elif kind == "pending":
            features.append(float(o.repair_pending))
        elif kind in _WINDOWS:
            steps = _WINDOWS[kind]
            recent = list(o.recent_events[-steps:])
            padded = [0] * (steps - len(recent)) + recent
            for code in padded:
                one_hot = [0.0] * HISTORY_CODES
                if code:
                    one_hot[code - 1] = 1.0
                features.extend(one_hot)
    else:
        raise ValueError(f"unknown observation kind {kind!r}; expected one of {OBSERVATION_KINDS}")
    return np.asarray(features, dtype=np.float32)
