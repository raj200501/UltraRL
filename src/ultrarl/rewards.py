"""Reward schemes.

All schemes share the same *base* reward

    r_base = +R_correct * [COMMIT and success] + R_incorrect * [COMMIT and not success]
             - step_penalty * cost(a)            (if ``budget_penalty``)

and differ only in the per-step *process* term added on top:

``outcome_only``
    no process term.
``naive_process``
    +naive_progress_bonus for a THINK that leaves the running value correct, and
    +naive_verify_bonus for any first-time informative verification. Note that the
    progress bonus uses the *hidden* correctness of the running value.
``monitorability_aware``
    rewards informative verification and penalises unverified stretches:

    +ma_safe_verify_bonus    first-time VERIFY of a correct step
    +ma_fault_verify_bonus   first-time VERIFY that catches the fault (or, with
                             ``ma_fault_bonus_on_repair``, the first correct
                             re-execution of the located fault)
    +ma_cue_verify_bonus     informative VERIFY of a step whose cue fired
    -ma_low_cue_penalty      informative SAFE VERIFY of a step whose cue did not fire
    -ma_waste_verify_penalty uninformative VERIFY (repeat, or check downstream of an
                             undetected fault)
    -ma_gap_penalty * (g' - g) / depth   change in the unchecked gap g

Here ``g = t - max(checked)`` is the number of steps executed since the
highest-index checked step. The gap term telescopes to
``-ma_gap_penalty * g_T / depth`` over an episode: (up to discounting) a
potential-based term plus a terminal penalty that depends only on the distance
from the last *checked index* to the end, not on how many steps are unchecked.
One check of the final step cancels it, and every action that lowers ``g``
(including an uninformative check or a REVISE) is refunded.
"""

from __future__ import annotations

from dataclasses import dataclass

from ultrarl.config import EnvConfig

REWARD_SCHEMES = ("outcome_only", "naive_process", "monitorability_aware")


@dataclass
class StepEvent:
    """What happened during one environment step (inputs to the reward)."""

    action: str
    depth: int
    cost: int
    success: bool = False
    think_valid: bool = False
    verify_first: bool = False
    verify_unsafe: bool = False
    belief_valid: bool = False
    cue: int = 0
    gap_before: int = 0
    gap_after: int = 0
    repaired: bool = False  # THINK correctly re-executed the located fault

    @property
    def safe_check(self) -> bool:
        """First-time verification confirming a correct step."""
        return self.action == "VERIFY" and self.verify_first and not self.verify_unsafe and self.belief_valid

    @property
    def fault_catch(self) -> bool:
        """First-time verification that catches the fault."""
        return self.action == "VERIFY" and self.verify_first and self.verify_unsafe

    @property
    def informative_verify(self) -> bool:
        return self.safe_check or self.fault_catch


def base_reward(cfg: EnvConfig, event: StepEvent) -> float:
    reward = 0.0
    if cfg.budget_penalty:
        reward -= cfg.step_penalty * event.cost
    if event.action == "COMMIT":
        reward += cfg.correct_reward if event.success else cfg.incorrect_reward
    return reward


def process_reward(cfg: EnvConfig, event: StepEvent) -> float:
    scheme = cfg.reward_scheme
    if scheme == "outcome_only":
        return 0.0
    if scheme == "naive_process":
        reward = 0.0
        if event.action == "THINK" and event.think_valid:
            reward += cfg.naive_progress_bonus
        if event.informative_verify:
            reward += cfg.naive_verify_bonus
        return reward
    if scheme == "monitorability_aware":
        reward = 0.0
        if event.safe_check:
            reward += cfg.ma_safe_verify_bonus
        if event.repaired if cfg.ma_fault_bonus_on_repair else event.fault_catch:
            reward += cfg.ma_fault_verify_bonus
        if event.informative_verify:
            if event.cue == 1:
                reward += cfg.ma_cue_verify_bonus
            elif not event.fault_catch:
                reward -= cfg.ma_low_cue_penalty
        if event.action == "VERIFY" and not event.informative_verify:
            reward -= cfg.ma_waste_verify_penalty
        reward -= cfg.ma_gap_penalty * (event.gap_after - event.gap_before) / max(1, event.depth)
        return reward
    raise ValueError(f"unknown reward scheme {scheme!r}; expected one of {REWARD_SCHEMES}")


def compute_reward(cfg: EnvConfig, event: StepEvent) -> float:
    return base_reward(cfg, event) + process_reward(cfg, event)
