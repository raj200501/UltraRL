"""Budgeted arithmetic-chain environment for studying *when* to verify.

An episode is a chain of ``depth`` arithmetic transitions ``s_0 -> s_1 -> ... -> s_d``.
With probability ``p_fault`` exactly one transition is *faulty*: until the fault
has been detected, executing that transition applies a corrupted operation, so
the agent's running value silently diverges from the true value. Each transition
carries a noisy one-bit *cue* that fires with probability ``cue_hit_rate`` on the
faulty transition and ``cue_false_positive_rate`` on every other transition.

Actions
    THINK   execute the next transition                      (costs ``think_cost``)
    VERIFY  exactly check the most recently executed step    (costs ``verify_cost``)
    REVISE  undo the most recently executed step             (costs ``revise_cost``)
    COMMIT  end the episode (free)

VERIFY is a *local* check: it reports UNSAFE iff the most recent transition is
the faulty one and was executed with the corrupted operation. The first UNSAFE
report marks the fault as located, after which re-executing that transition
(REVISE then THINK) applies the correct operation. An episode succeeds iff the
agent commits after executing all ``depth`` transitions with the correct value.

Every non-COMMIT action costs at least one unit of a fixed budget
(``depth + budget_extra``), so episodes always terminate. Rewards are computed
by :mod:`ultrarl.rewards`; observations by :mod:`ultrarl.observations`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from ultrarl.config import EnvConfig
from ultrarl.observations import encode_observation, observation_dim
from ultrarl.rewards import StepEvent, compute_reward


class Action(IntEnum):
    THINK = 0
    VERIFY = 1
    REVISE = 2
    COMMIT = 3


class VerifyStatus(IntEnum):
    NONE = 0  # last action was not VERIFY
    SAFE = 1
    UNSAFE = 2


NUM_ACTIONS = len(Action)

Op = tuple[str, int]


def apply_op(value: int, op: Op) -> int:
    kind, operand = op
    if kind == "add":
        return value + operand
    if kind == "sub":
        return value - operand
    if kind == "mul":
        return value * operand
    raise ValueError(f"unknown op kind: {kind}")


@dataclass(frozen=True)
class Transition:
    safe_op: Op
    faulty_op: Op


@dataclass(frozen=True)
class Observable:
    """Everything a policy is allowed to condition on (no hidden state)."""

    step: int
    depth: int
    remaining_budget: int
    max_budget: int
    cue: int
    last_verify: VerifyStatus
    revised_recently: bool
    unchecked_gap: int
    step_checked: bool
    budget_extra: int
    max_depth: int


@dataclass
class EpisodeStats:
    """Counters accumulated over one episode; converted to metrics at the end."""

    total_return: float = 0.0
    num_actions: int = 0
    cost_spent: int = 0
    verify_count: int = 0
    revise_count: int = 0
    verify_tp: int = 0
    # Cue-conditioned verification: decision points where VERIFY was available
    # and the current step had not yet been checked ("fresh" opportunities).
    fresh_opps: list[int] = field(default_factory=lambda: [0, 0])
    fresh_verifies: list[int] = field(default_factory=lambda: [0, 0])


class ChainEnv:
    """Single-fault arithmetic chain with exact local verification and a budget."""

    def __init__(self, config: EnvConfig, seed: int | Sequence[int] = 0) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.obs_dim = observation_dim(config.observation)
        self.num_actions = NUM_ACTIONS
        self.depth = 0
        self.done = True

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        seed: int | Sequence[int] | None = None,
        depth_range: tuple[int, int] | None = None,
        fault_index: int | None | str = "sample",
    ) -> np.ndarray:
        """Start a new episode.

        Args:
            seed: optional seed (int or sequence of ints) for this episode's RNG.
            depth_range: inclusive ``(low, high)`` range for the chain depth;
                defaults to ``config.train_depth_range``.
            fault_index: ``"sample"`` (default) draws the fault from the prior,
                ``None`` forces a fault-free chain, an int forces a 0-based fault
                position (useful in tests).
        """
        cfg = self.cfg
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        low, high = depth_range or cfg.train_depth_range
        self.depth = int(self.rng.integers(low, high + 1))

        if fault_index == "sample":
            self.fault_index = self._sample_fault_index() if self.rng.random() < cfg.p_fault else None
        else:
            self.fault_index = None if fault_index is None else int(fault_index)
            if self.fault_index is not None and not 0 <= self.fault_index < self.depth:
                raise ValueError(f"fault_index {self.fault_index} outside chain of depth {self.depth}")

        start = int(self.rng.integers(2, 7))
        self.true_values = [start]
        self.transitions: list[Transition] = []
        for _ in range(self.depth):
            transition = self._sample_transition(self.true_values[-1])
            self.transitions.append(transition)
            self.true_values.append(apply_op(self.true_values[-1], transition.safe_op))
        # cues[t] is the cue for transition t-1 (the step that produced value t);
        # cues[0] is a dummy 0 for the start state.
        self.cues = [0] + [
            int(self.rng.random() < (cfg.cue_hit_rate if t == self.fault_index else cfg.cue_false_positive_rate))
            for t in range(self.depth)
        ]

        self.step_idx = 0
        self.beliefs = [start]
        self.max_budget = self.depth + cfg.budget_extra
        self.remaining_budget = self.max_budget
        self.checked: set[int] = set()
        self.fault_located = False
        self.detection_step: int | None = None
        self.last_verify = VerifyStatus.NONE
        self.last_action: Action | None = None
        self.revised_recently = False
        self.done = False
        self.success = False
        self.incomplete_commit = False
        self.voluntary_incomplete = False
        self.stats = EpisodeStats()
        return self.observation()

    def _sample_fault_index(self) -> int:
        if self.cfg.fault_profile == "uniform":
            return int(self.rng.integers(0, self.depth))
        # "middle": weights peak at the centre of the chain.
        positions = np.arange(self.depth, dtype=float)
        centre = (self.depth - 1) / 2.0
        weights = 1.0 + self.cfg.fault_peak_weight * (1.0 - np.abs(positions - centre) / max(1.0, centre))
        weights = np.clip(weights, 0.1, None)
        return int(self.rng.choice(self.depth, p=weights / weights.sum()))

    def _sample_transition(self, value: int) -> Transition:
        options: list[Op] = [("add", int(self.rng.integers(1, 5)))]
        if value > 2:
            options.append(("sub", int(self.rng.integers(1, min(5, value)))))
        if abs(value) <= 16:
            options.append(("mul", int(self.rng.choice([2, 3]))))
        safe = options[int(self.rng.integers(0, len(options)))]
        kind, operand = safe
        faulty: Op = (kind, operand + 1) if kind in ("add", "sub") else ("add", operand)
        if apply_op(value, faulty) == apply_op(value, safe):
            faulty = ("add", operand + 2)
        return Transition(safe_op=safe, faulty_op=faulty)

    # ------------------------------------------------------------- accessors
    def action_cost(self, action: Action) -> int:
        cfg = self.cfg
        return {
            Action.THINK: cfg.think_cost,
            Action.VERIFY: cfg.verify_cost,
            Action.REVISE: cfg.revise_cost,
            Action.COMMIT: 0,
        }[Action(action)]

    @property
    def belief_valid(self) -> bool:
        return self.beliefs[self.step_idx] == self.true_values[self.step_idx]

    @property
    def unchecked_gap(self) -> int:
        """Steps executed since the most recently checked step index."""
        return max(0, self.step_idx - max(self.checked, default=0))

    def action_mask(self) -> np.ndarray:
        cfg = self.cfg
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[Action.COMMIT] = True
        if self.done:
            return mask
        budget = self.remaining_budget
        mask[Action.THINK] = self.step_idx < self.depth and budget >= cfg.think_cost
        mask[Action.VERIFY] = cfg.allow_verify and self.step_idx > 0 and budget >= cfg.verify_cost
        mask[Action.REVISE] = cfg.allow_revise and self.step_idx > 0 and budget >= cfg.revise_cost
        return mask

    def observable(self) -> Observable:
        return Observable(
            step=self.step_idx,
            depth=self.depth,
            remaining_budget=self.remaining_budget,
            max_budget=self.max_budget,
            cue=self.cues[self.step_idx],
            last_verify=self.last_verify,
            revised_recently=self.revised_recently,
            unchecked_gap=self.unchecked_gap,
            step_checked=self.step_idx in self.checked,
            budget_extra=self.cfg.budget_extra,
            max_depth=self.cfg.max_depth,
        )

    def observation(self) -> np.ndarray:
        return encode_observation(self.cfg.observation, self.observable())

    # ------------------------------------------------------------------ step
    def step(self, action: int | Action) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        action = Action(action)
        if not self.action_mask()[action]:
            raise ValueError(f"action {action.name} is not available in the current state")

        self.action_mask_before_commit = self.action_mask()
        self._record_cue_selectivity(action)
        gap_before = self.unchecked_gap
        event = StepEvent(action=action.name, depth=self.depth, cost=self.action_cost(action))

        if action == Action.THINK:
            self._think(event)
        elif action == Action.VERIFY:
            self._verify(event)
        elif action == Action.REVISE:
            self._revise()
        else:
            self.done = True
            self.success = self.step_idx == self.depth and self.belief_valid
            self.incomplete_commit = self.step_idx < self.depth
            self.voluntary_incomplete = self.incomplete_commit and bool(self.action_mask_before_commit[Action.THINK])
            event.success = self.success

        self.remaining_budget -= event.cost
        self.stats.cost_spent += event.cost
        event.gap_before = gap_before
        event.gap_after = self.unchecked_gap
        reward = compute_reward(self.cfg, event)

        self.last_action = action
        self.revised_recently = action == Action.REVISE
        if action != Action.VERIFY:
            self.last_verify = VerifyStatus.NONE
        self.stats.total_return += reward
        self.stats.num_actions += 1

        info: dict[str, Any] = {"event": event}
        if self.done:
            info["metrics"] = self.episode_metrics()
        return self.observation(), reward, self.done, info

    def _think(self, event: StepEvent) -> None:
        t = self.step_idx
        transition = self.transitions[t]
        corrupted = t == self.fault_index and not self.fault_located
        op = transition.faulty_op if corrupted else transition.safe_op
        self.beliefs.append(apply_op(self.beliefs[-1], op))
        self.step_idx += 1
        event.think_valid = self.belief_valid

    def _verify(self, event: StepEvent) -> None:
        t = self.step_idx
        unsafe = (t - 1) == self.fault_index and not self.belief_valid
        self.last_verify = VerifyStatus.UNSAFE if unsafe else VerifyStatus.SAFE
        self.stats.verify_count += 1
        event.cue = self.cues[t]
        event.verify_unsafe = unsafe
        event.verify_first = t not in self.checked
        event.belief_valid = self.belief_valid
        if not event.verify_first:
            return
        self.checked.add(t)
        if unsafe:
            self.stats.verify_tp += 1
            self.fault_located = True
            self.detection_step = t

    def _revise(self) -> None:
        self.stats.revise_count += 1
        self.step_idx -= 1
        self.beliefs.pop()

    def _record_cue_selectivity(self, action: Action) -> None:
        if self.step_idx == 0 or self.step_idx in self.checked or not self.action_mask()[Action.VERIFY]:
            return
        cue = self.cues[self.step_idx]
        self.stats.fresh_opps[cue] += 1
        self.stats.fresh_verifies[cue] += int(action == Action.VERIFY)

    # --------------------------------------------------------------- metrics
    def episode_metrics(self) -> dict[str, float]:
        """Per-episode metrics. NaN marks metrics undefined for this episode."""
        s = self.stats
        fault = self.fault_index is not None
        nan = float("nan")
        return {
            "success": float(self.success),
            "return": s.total_return,
            "num_actions": float(s.num_actions),
            "cost_spent": float(s.cost_spent),
            "cost_adjusted_accuracy": float(self.success) - self.cfg.cost_adjust_weight * s.cost_spent,
            # committed before executing the whole chain (voluntarily = THINK was still affordable)
            "incomplete_commit": float(self.incomplete_commit),
            "voluntary_incomplete_commit": float(self.voluntary_incomplete),
            "verify_count": float(s.verify_count),
            "revise_count": float(s.revise_count),
            "verify_tp": float(s.verify_tp),
            "depth": float(self.depth),
            "fault_present": float(fault),
            # P(fault detected by an UNSAFE verification | fault present)
            "fault_detected": float(self.fault_located) if fault else nan,
            # P(success | fault present): requires detect -> revise -> redo
            "repair_success": float(self.success) if fault else nan,
            "fresh_opps_cue0": float(s.fresh_opps[0]),
            "fresh_opps_cue1": float(s.fresh_opps[1]),
            "fresh_verifies_cue0": float(s.fresh_verifies[0]),
            "fresh_verifies_cue1": float(s.fresh_verifies[1]),
        }
