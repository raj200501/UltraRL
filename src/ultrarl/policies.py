"""Hand-coded reference policies.

Every heuristic acts only on :class:`ultrarl.env.Observable` (the same
information available to the learned agents) plus the action mask.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ultrarl.env import Action, Observable, VerifyStatus

Policy = Callable[[Observable, np.ndarray], int]


def _think_or_commit(o: Observable, mask: np.ndarray) -> int:
    if o.step < o.depth and mask[Action.THINK]:
        return Action.THINK
    return Action.COMMIT


def always_think(o: Observable, mask: np.ndarray) -> int:
    """Never check anything: execute the chain and commit."""
    return _think_or_commit(o, mask)


def verify_every_step(o: Observable, mask: np.ndarray) -> int:
    """Check every newly executed step; repair on UNSAFE."""
    if o.last_verify == VerifyStatus.UNSAFE and mask[Action.REVISE]:
        return Action.REVISE
    if o.step > 0 and not o.step_checked and mask[Action.VERIFY]:
        return Action.VERIFY
    return _think_or_commit(o, mask)


def cue_triggered(o: Observable, mask: np.ndarray) -> int:
    """Check a newly executed step only if its cue fired; repair on UNSAFE."""
    if o.last_verify == VerifyStatus.UNSAFE and mask[Action.REVISE]:
        return Action.REVISE
    if o.step > 0 and o.cue == 1 and not o.step_checked and mask[Action.VERIFY]:
        return Action.VERIFY
    return _think_or_commit(o, mask)


REPAIR_COST = 2  # REVISE + re-THINK


def budget_aware(margin: int) -> Policy:
    """Check cued steps whenever a repair stays affordable, and also check un-cued
    steps while the budget slack exceeds that by ``margin``."""

    def policy(o: Observable, mask: np.ndarray) -> int:
        if o.last_verify == VerifyStatus.UNSAFE and mask[Action.REVISE]:
            return Action.REVISE
        slack = o.remaining_budget - (o.depth - o.step)
        affordable = slack >= 1 + REPAIR_COST
        wanted = o.cue == 1 or slack >= 1 + REPAIR_COST + margin
        if o.step > 0 and not o.step_checked and affordable and wanted and mask[Action.VERIFY]:
            return Action.VERIFY
        return _think_or_commit(o, mask)

    return policy


class RandomPolicy:
    """Uniform over available actions."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, o: Observable, mask: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(mask)))


def build_heuristic(name: str, seed: int) -> Policy:
    if name == "random":
        return RandomPolicy(seed)
    table: dict[str, Policy] = {
        "always_think": always_think,
        "verify_every_step": verify_every_step,
        "cue_triggered": cue_triggered,
    }
    if name.startswith("budget_aware_"):
        return budget_aware(int(name.removeprefix("budget_aware_")))
    try:
        return table[name]
    except KeyError:
        raise ValueError(f"unknown heuristic {name!r}") from None
