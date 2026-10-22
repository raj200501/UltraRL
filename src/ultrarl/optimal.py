"""Success-maximising reference policy, computed exactly by dynamic programming.

Only the success probability matters for this reference, not any training
reward. We restrict attention to policies that decide whether to check a step
right after executing it and never go back to check an earlier one; going back
costs at least two extra units (undo and redo) on top of the check.

Under that restriction the problem has a small sufficient statistic. If any
already-executed step held the fault unchecked, the episode is lost whatever
the agent does next, so decisions only matter on the event that every executed
step is clean. On that event the cues seen so far are equally likely under every
remaining fault position (and under no fault), so the posterior over the fault
is the prior restricted to the unexecuted steps. The value therefore depends
only on the number of executed steps ``k`` and the budget slack ``s`` (budget
left beyond what finishing the chain needs):

    V(d, s) = 1                                   for s >= 0
    V(k, s) = sum_c max(skip_c, check_c)          for k < d, with
    skip_c  = P(no fault at k, c) V(k+1, s)
    check_c = P(fault at k, c) [s - c_V - c_R - c_T >= 0]
              + P(no fault at k, c) [s >= c_V] V(k+1, s - c_V)

where the probabilities are conditional on steps 0..k-1 being clean and ``c`` is
step k's cue. A caught fault is repaired by undoing and redoing the step, after
which the rest of the chain is clean.

:func:`exhaustive_success` checks the restriction: it searches over every
history-dependent policy, including ones that undo steps to check earlier
ones, by enumerating the hidden (fault, cues) configurations. It is exponential
in the depth and only meant for short chains (tests use depth <= 4;
``scripts/check_planner.py`` goes further).
"""

from __future__ import annotations

import itertools
from functools import cache, lru_cache

import numpy as np

from ultrarl.config import EnvConfig
from ultrarl.env import Action, Observable, VerifyStatus, fault_position_probabilities


def fault_prior(cfg: EnvConfig, depth: int) -> np.ndarray:
    """P(fault at position i) for i < depth; the remaining mass is 'no fault'."""
    return cfg.p_fault * fault_position_probabilities(cfg, depth)


TIE_TOLERANCE = 1e-12


class OptimalPlanner:
    """Exact values and check decisions for one environment config and chain depth."""

    def __init__(self, cfg: EnvConfig, depth: int) -> None:
        self.cfg, self.depth = cfg, depth
        prior = fault_prior(cfg, depth)
        clean_mass = (1.0 - cfg.p_fault) + np.cumsum(prior[::-1])[::-1]  # P(fault >= k or none)
        self.p_fault_here = prior / clean_mass  # P(fault at k | steps before k are clean)
        self.value = lru_cache(maxsize=None)(self._value)

    def _options(self, k: int, s: int, cue: int) -> tuple[float, float]:
        cfg = self.cfg
        hit = cfg.cue_hit_rate if cue else 1.0 - cfg.cue_hit_rate
        false_alarm = cfg.cue_false_positive_rate if cue else 1.0 - cfg.cue_false_positive_rate
        fault, clean = self.p_fault_here[k] * hit, (1.0 - self.p_fault_here[k]) * false_alarm
        skip = clean * self.value(k + 1, s)
        check = 0.0
        if s >= cfg.verify_cost:
            repair_affordable = s - cfg.verify_cost - cfg.revise_cost - cfg.think_cost >= 0
            check = fault * repair_affordable + clean * self.value(k + 1, s - cfg.verify_cost)
        return skip, check

    def _value(self, k: int, s: int) -> float:
        if s < 0:
            return 0.0
        if k == self.depth:
            return 1.0
        return sum(max(self._options(k, s, cue)) for cue in (0, 1))

    def should_check(self, executed: int, slack: int, cue: int) -> bool:
        """Check the step just executed (``executed`` steps done) with this cue?

        Ties (both choices optimal) go to skipping, with a tolerance so that the
        choice does not depend on floating-point rounding."""
        skip, check = self._options(executed - 1, slack, cue)
        return check > skip + TIE_TOLERANCE


def initial_slack(cfg: EnvConfig, depth: int) -> int:
    return depth + cfg.budget_extra - depth * cfg.think_cost


def optimal_success(cfg: EnvConfig, depth_range: tuple[int, int]) -> float:
    """Expected success of the optimal policy, with depths drawn uniformly from the range."""
    low, high = depth_range
    return float(np.mean([OptimalPlanner(cfg, d).value(0, initial_slack(cfg, d)) for d in range(low, high + 1)]))


def optimal_policy(cfg: EnvConfig):
    """The planner's policy, acting on :class:`Observable` like the other heuristics."""
    planners: dict[int, OptimalPlanner] = {}

    def policy(o: Observable, mask: np.ndarray) -> int:
        if o.last_verify == VerifyStatus.UNSAFE and mask[Action.REVISE]:
            return Action.REVISE
        if o.step > 0 and not o.step_checked and not o.fault_located and mask[Action.VERIFY]:
            if o.depth not in planners:
                planners[o.depth] = OptimalPlanner(cfg, o.depth)
            planner = planners[o.depth]
            slack = o.slack
            if planner.should_check(o.step, slack, o.cue):
                return Action.VERIFY
        if o.step < o.depth and mask[Action.THINK]:
            return Action.THINK
        return Action.COMMIT

    return policy


def exhaustive_success(cfg: EnvConfig, depth: int) -> float:
    """Best success over all history-dependent policies (exhaustive; short chains only).

    An information state is the set of hidden configurations (fault position or
    none, and the cue of every step) consistent with what the agent has seen,
    each with a flag for whether its running value is currently wrong.
    """
    prior = fault_prior(cfg, depth)
    configs = []
    for fault in [None, *range(depth)]:
        p_fault = 1.0 - cfg.p_fault if fault is None else prior[fault]
        for cues in itertools.product((0, 1), repeat=depth):
            p = p_fault
            for i, cue in enumerate(cues):
                rate = cfg.cue_hit_rate if i == fault else cfg.cue_false_positive_rate
                p *= rate if cue else 1.0 - rate
            configs.append((p, fault, cues))

    @cache
    def value(step: int, budget: int, located: bool, checked: frozenset, group: frozenset) -> float:
        best = sum(configs[c][0] for c, wrong in group if step == depth and not wrong)  # COMMIT
        if step < depth and budget >= cfg.think_cost:  # THINK, then observe the new step's cue
            by_cue: dict[int, set] = {}
            for c, wrong in group:
                fault, cues = configs[c][1], configs[c][2]
                by_cue.setdefault(cues[step], set()).add((c, wrong or (step == fault and not located)))
            after = budget - cfg.think_cost
            best = max(best, sum(value(step + 1, after, located, checked, frozenset(g)) for g in by_cue.values()))
        if step > 0 and budget >= cfg.verify_cost and step not in checked:  # VERIFY the last step
            by_result: dict[bool, set] = {}
            for c, wrong in group:
                by_result.setdefault(step - 1 == configs[c][1] and wrong, set()).add((c, wrong))
            after, seen = budget - cfg.verify_cost, checked | {step}
            outcomes = [value(step, after, located or unsafe, seen, frozenset(g)) for unsafe, g in by_result.items()]
            best = max(best, sum(outcomes))
        if step > 0 and budget >= cfg.revise_cost:  # REVISE: undoing the faulty step clears the error
            undone = frozenset((c, wrong and step - 1 > configs[c][1]) for c, wrong in group)
            best = max(best, value(step - 1, budget - cfg.revise_cost, located, checked, undone))
        return best

    start = frozenset((i, False) for i in range(len(configs)))
    return value(0, depth + cfg.budget_extra, False, frozenset(), start)
