# Original environment implementation, vendored from commit 81ec8cb (src/envs.py) and reformatted only
# so that scripts/check_legacy_equivalence.py works without git history. Not used by ultrarl.
# ruff: noqa
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class Action(IntEnum):
    THINK = 0
    VERIFY = 1
    REVISE = 2
    COMMIT = 3


class VerifyStatus(IntEnum):
    NOT_USED = 0
    SAFE = 1
    UNSAFE = 2


ACTION_NAMES = {
    Action.THINK: "THINK",
    Action.VERIFY: "VERIFY",
    Action.REVISE: "REVISE",
    Action.COMMIT: "COMMIT",
}


@dataclass
class TransitionSpec:
    safe_op: tuple[str, int]
    faulty_op: tuple[str, int]
    safe_description: str
    faulty_description: str


def apply_op(value: int, op: tuple[str, int]) -> int:
    kind, operand = op
    if kind == "add":
        return value + operand
    if kind == "sub":
        return value - operand
    if kind == "mul":
        return value * operand
    raise ValueError(f"Unknown op kind: {kind}")


def op_to_text(op: tuple[str, int]) -> str:
    kind, operand = op
    if kind == "add":
        return f"+{operand}"
    if kind == "sub":
        return f"-{operand}"
    if kind == "mul":
        return f"*{operand}"
    raise ValueError(f"Unknown op kind: {kind}")


class ProcessReasoningEnv:
    """Tiny symbolic reasoning environment with exact local verification."""

    def __init__(self, config: dict[str, Any], seed: int = 0) -> None:
        self.config = dict(config)
        self.train_depth_range = tuple(self.config.get("train_depth_range", [3, 6]))
        self.max_depth = int(self.config.get("max_depth", 9))
        self.p_fault = float(self.config.get("p_fault", 0.35))
        self.default_budget_extra = int(self.config.get("budget_extra", 3))
        self.fault_profile = str(self.config.get("fault_profile", "middle"))
        self.fault_peak_weight = float(self.config.get("fault_peak_weight", 2.0))
        self.use_risk_hints = bool(self.config.get("use_risk_hints", True))
        self.hint_reliability = float(self.config.get("hint_reliability", 0.85))
        self.hint_false_positive = float(self.config.get("hint_false_positive", 0.15))
        self.verify_cost = int(self.config.get("verify_cost", 1))
        self.revise_cost = int(self.config.get("revise_cost", 1))
        self.think_cost = int(self.config.get("think_cost", 1))
        self.correct_reward = float(self.config.get("correct_reward", 1.0))
        self.incorrect_reward = float(self.config.get("incorrect_reward", -1.0))
        self.invalid_action_penalty = float(self.config.get("invalid_action_penalty", 0.1))
        self.step_penalty = float(self.config.get("step_penalty", 0.04))
        self.cost_adjust_weight = float(self.config.get("cost_adjust_weight", 0.05))
        self.naive_progress_bonus = float(self.config.get("naive_progress_bonus", 0.015))
        self.naive_verify_bonus = float(self.config.get("naive_verify_bonus", 0.04))
        self.ma_safe_verify_bonus = float(self.config.get("ma_safe_verify_bonus", 0.05))
        self.ma_fault_verify_bonus = float(self.config.get("ma_fault_verify_bonus", 0.15))
        self.ma_waste_verify_penalty = float(self.config.get("ma_waste_verify_penalty", 0.035))
        self.ma_opaque_penalty = float(self.config.get("ma_opaque_penalty", 0.25))
        self.ma_hint_verify_bonus = float(self.config.get("ma_hint_verify_bonus", 0.15))
        self.ma_low_hint_penalty = float(self.config.get("ma_low_hint_penalty", 0.02))
        self.reward_scheme = str(self.config.get("reward_scheme", "outcome_only"))
        self.allow_verify = bool(self.config.get("allow_verify", True))
        self.allow_revise = bool(self.config.get("allow_revise", True))
        self.budget_penalty = bool(self.config.get("budget_penalty", True))
        self.rng = np.random.default_rng(seed)
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self.depth = 0
        self.remaining_budget = 0
        self.max_budget = 0
        self.fault_index: int | None = None
        self.transitions: list[TransitionSpec] = []
        self.risk_hints: list[int] = []
        self.true_values: list[int] = []
        self.belief_values: list[int] = []
        self.current_step = 0
        self.done = False
        self.success = False
        self.last_verify_status = VerifyStatus.NOT_USED
        self.last_action: Action | None = None
        self.revised_recently = False
        self.total_return = 0.0
        self.total_action_cost = 0
        self.total_actions = 0
        self.verify_count = 0
        self.revise_count = 0
        self.verify_tp = 0
        self.verify_fp = 0
        self.trusted_prefix = 0
        self.verified_safe_steps: set[int] = set()
        self.checked_steps: set[int] = set()
        self.first_unsafe_verify_step: int | None = None
        self.commit_step: int | None = None
        self.case_description = ""

    def clone_with(self, overrides: dict[str, Any]) -> "ProcessReasoningEnv":
        merged = dict(self.config)
        merged.update(overrides)
        return ProcessReasoningEnv(merged)

    @property
    def observation_dim(self) -> int:
        return 10

    @property
    def action_dim(self) -> int:
        return 4

    def discrete_state(self) -> tuple[int, int, int, int, int]:
        return (
            self.current_step,
            self.remaining_budget,
            int(self.last_verify_status),
            int(self.revised_recently),
            self.depth,
        )

    def action_cost(self, action: int | Action) -> int:
        action = Action(action)
        if action == Action.THINK:
            return self.think_cost
        if action == Action.VERIFY:
            return self.verify_cost
        if action == Action.REVISE:
            return self.revise_cost
        return 0

    def unresolved_fault_distance(self) -> int:
        if self.fault_index is None:
            return 0
        if self.first_unsafe_verify_step is not None:
            return 0
        fault_pos = self.fault_index + 1
        return max(0, self.current_step - fault_pos)

    def is_current_state_valid(self) -> bool:
        return self.belief_values[self.current_step] == self.true_values[self.current_step]

    def available_actions(self) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[Action.COMMIT] = True
        if not self.done and self.current_step < self.depth and self.remaining_budget >= self.think_cost:
            mask[Action.THINK] = True
        if not self.done and self.allow_verify and self.current_step > 0 and self.remaining_budget >= self.verify_cost:
            mask[Action.VERIFY] = True
        if not self.done and self.allow_revise and self.current_step > 0 and self.remaining_budget >= self.revise_cost:
            mask[Action.REVISE] = True
        return mask

    def _sample_transition(self, current_value: int) -> TransitionSpec:
        options: list[tuple[str, int]] = [("add", int(self.rng.integers(1, 5)))]
        if current_value > 2:
            upper = int(min(5, current_value))
            options.append(("sub", int(self.rng.integers(1, upper))))
        if abs(current_value) <= 16:
            options.append(("mul", int(self.rng.choice([2, 3]))))
        safe_op = options[int(self.rng.integers(0, len(options)))]
        if safe_op[0] == "add":
            faulty_op = ("add", safe_op[1] + 1)
        elif safe_op[0] == "sub":
            faulty_op = ("sub", safe_op[1] + 1)
        else:
            faulty_op = ("add", safe_op[1])
        if apply_op(current_value, faulty_op) == apply_op(current_value, safe_op):
            faulty_op = ("add", safe_op[1] + 2)
        return TransitionSpec(
            safe_op=safe_op,
            faulty_op=faulty_op,
            safe_description=op_to_text(safe_op),
            faulty_description=op_to_text(faulty_op),
        )

    def _build_case_description(self, start_value: int) -> str:
        parts = [f"start={start_value}"]
        for step, transition in enumerate(self.transitions, start=1):
            if self.fault_index == step - 1:
                parts.append(f"s{step}:{transition.safe_description} (faulty branch {transition.faulty_description})")
            else:
                parts.append(f"s{step}:{transition.safe_description}")
        return " | ".join(parts)

    def _sample_fault_index(self) -> int:
        if self.fault_profile == "uniform":
            return int(self.rng.integers(0, self.depth))
        positions = np.arange(self.depth, dtype=float)
        center = (self.depth - 1) / 2.0
        denom = max(1.0, center)
        weights = 1.0 + self.fault_peak_weight * (1.0 - (np.abs(positions - center) / denom))
        weights = np.clip(weights, 0.1, None)
        weights = weights / weights.sum()
        return int(self.rng.choice(np.arange(self.depth), p=weights))

    def _sample_risk_hints(self) -> list[int]:
        hints = [0]
        for step in range(self.depth):
            if not self.use_risk_hints:
                hints.append(0)
                continue
            is_fault_step = self.fault_index is not None and step == self.fault_index
            hit_prob = self.hint_reliability if is_fault_step else self.hint_false_positive
            hints.append(int(self.rng.random() < hit_prob))
        return hints

    def reset(
        self,
        seed: int | None = None,
        depth_range: tuple[int, int] | None = None,
        budget: int | None = None,
        fault_index: int | None | str = "sample",
        p_fault: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_episode_state()
        depth_low, depth_high = depth_range or self.train_depth_range
        self.depth = int(self.rng.integers(depth_low, depth_high + 1))
        sampled_fault_prob = self.p_fault if p_fault is None else float(p_fault)
        if fault_index == "sample":
            if self.rng.random() < sampled_fault_prob:
                self.fault_index = self._sample_fault_index()
            else:
                self.fault_index = None
        elif fault_index is None:
            self.fault_index = None
        else:
            self.fault_index = int(fault_index)
        start_value = int(self.rng.integers(2, 7))
        self.true_values = [start_value]
        self.belief_values = [start_value]
        current_true_value = start_value
        self.transitions = []
        for _ in range(self.depth):
            transition = self._sample_transition(current_true_value)
            self.transitions.append(transition)
            current_true_value = apply_op(current_true_value, transition.safe_op)
            self.true_values.append(current_true_value)
        self.risk_hints = self._sample_risk_hints()
        self.case_description = self._build_case_description(start_value)
        self.max_budget = budget if budget is not None else self.depth + self.default_budget_extra
        self.remaining_budget = self.max_budget
        info = {
            "depth": self.depth,
            "budget": self.remaining_budget,
            "fault_index": None if self.fault_index is None else self.fault_index + 1,
            "case_description": self.case_description,
        }
        return self._make_obs(), info

    def _make_obs(self) -> np.ndarray:
        verify_one_hot = [0.0, 0.0, 0.0]
        verify_one_hot[int(self.last_verify_status)] = 1.0
        budget_slack = self.remaining_budget - max(0, self.depth - self.current_step)
        normalized_budget_slack = np.clip(
            0.5 + (budget_slack / max(1, 2 * self.max_budget)),
            0.0,
            1.0,
        )
        last_checked_step = max(self.checked_steps) if self.checked_steps else 0
        unchecked_gap = max(0, self.current_step - last_checked_step)
        current_hint = float(self.risk_hints[self.current_step]) if self.risk_hints else 0.0
        return np.asarray(
            [
                self.current_step / max(1, self.max_depth),
                self.remaining_budget / max(1, self.max_budget),
                self.depth / max(1, self.max_depth),
                normalized_budget_slack,
                unchecked_gap / max(1, self.max_depth),
                current_hint,
                float(self.revised_recently),
                *verify_one_hot,
            ],
            dtype=np.float32,
        )

    def _transition_is_faulty(self, transition_index: int) -> bool:
        return self.fault_index is not None and transition_index == self.fault_index

    def _handle_think(self) -> dict[str, Any]:
        event = {
            "invalid": False,
            "think_valid_progress": False,
            "extended_trusted_prefix": False,
            "fault_hit": False,
        }
        if self.current_step >= self.depth:
            event["invalid"] = True
            return event
        transition_index = self.current_step
        transition = self.transitions[transition_index]
        current_belief = self.belief_values[-1]
        if self._transition_is_faulty(transition_index) and self.first_unsafe_verify_step is None:
            if transition_index < self.trusted_prefix:
                next_belief = apply_op(current_belief, transition.safe_op)
            else:
                next_belief = apply_op(current_belief, transition.faulty_op)
                event["fault_hit"] = True
        elif self._transition_is_faulty(transition_index) and self.first_unsafe_verify_step is not None:
            next_belief = apply_op(current_belief, transition.safe_op)
        else:
            next_belief = apply_op(current_belief, transition.safe_op)
        self.belief_values.append(next_belief)
        self.current_step += 1
        self.remaining_budget -= self.think_cost
        self.total_action_cost += self.think_cost
        if self.is_current_state_valid():
            event["think_valid_progress"] = True
        if (
            self._transition_is_faulty(transition_index)
            and self.first_unsafe_verify_step is not None
            and transition_index == self.trusted_prefix
        ):
            self.trusted_prefix += 1
            event["extended_trusted_prefix"] = True
        return event

    def _handle_verify(self) -> dict[str, Any]:
        event = {
            "invalid": False,
            "safe_verify_bonus": False,
            "fault_verify_bonus": False,
            "wasted_verify": False,
        }
        if self.current_step == 0:
            event["invalid"] = True
            return event
        transition_index = self.current_step - 1
        already_checked = self.current_step in self.checked_steps
        self.remaining_budget -= self.verify_cost
        self.total_action_cost += self.verify_cost
        self.verify_count += 1
        current_valid = self.is_current_state_valid()
        if already_checked:
            if self._transition_is_faulty(transition_index) and not current_valid:
                self.last_verify_status = VerifyStatus.UNSAFE
            else:
                self.last_verify_status = VerifyStatus.SAFE
            event["wasted_verify"] = True
            return event

        self.checked_steps.add(self.current_step)
        if self._transition_is_faulty(transition_index) and not current_valid:
            self.last_verify_status = VerifyStatus.UNSAFE
            self.first_unsafe_verify_step = self.current_step
            self.verify_tp += 1
            event["fault_verify_bonus"] = True
        else:
            self.last_verify_status = VerifyStatus.SAFE
            extends_prefix = current_valid and transition_index == self.trusted_prefix
            if extends_prefix:
                self.trusted_prefix += 1
                self.verified_safe_steps.add(transition_index)
            if current_valid:
                event["safe_verify_bonus"] = True
            else:
                event["wasted_verify"] = True
            self.verify_fp += 1
        return event

    def _handle_revise(self) -> dict[str, Any]:
        event = {"invalid": False}
        if self.current_step == 0:
            event["invalid"] = True
            return event
        self.remaining_budget -= self.revise_cost
        self.total_action_cost += self.revise_cost
        self.revise_count += 1
        self.current_step -= 1
        self.belief_values.pop()
        return event

    def _handle_commit(self) -> dict[str, Any]:
        self.commit_step = self.current_step
        self.done = True
        self.success = self.current_step == self.depth and self.is_current_state_valid()
        return {"success": self.success}

    def _base_reward(self, action: Action, invalid: bool) -> float:
        reward = 0.0
        if self.budget_penalty:
            reward -= self.step_penalty * self.action_cost(action)
        if invalid:
            reward -= self.invalid_action_penalty
        if action == Action.COMMIT:
            reward += self.correct_reward if self.success else self.incorrect_reward
        return reward

    def step(self, action: int | Action) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode already finished")
        action = Action(action)
        last_checked_prev = max(self.checked_steps) if self.checked_steps else 0
        unchecked_gap_prev = max(0, self.current_step - last_checked_prev)
        trusted_prev = self.trusted_prefix
        if action != Action.COMMIT and self.remaining_budget < self.action_cost(action):
            event = {"invalid": True}
            reward = self._base_reward(action, invalid=True)
        else:
            if action == Action.THINK:
                event = self._handle_think()
            elif action == Action.VERIFY:
                event = self._handle_verify()
            elif action == Action.REVISE:
                event = self._handle_revise()
            else:
                event = self._handle_commit()
            reward = self._base_reward(action, invalid=bool(event.get("invalid", False)))
        if action == Action.THINK:
            self.last_verify_status = VerifyStatus.NOT_USED
        elif action == Action.REVISE and self.last_verify_status == VerifyStatus.NOT_USED:
            self.last_verify_status = VerifyStatus.NOT_USED
        if self.reward_scheme == "naive_process":
            if event.get("think_valid_progress", False):
                reward += self.naive_progress_bonus
            if event.get("safe_verify_bonus", False) or event.get("fault_verify_bonus", False):
                reward += self.naive_verify_bonus
        elif self.reward_scheme == "monitorability_aware":
            if event.get("safe_verify_bonus", False):
                reward += self.ma_safe_verify_bonus
            if event.get("fault_verify_bonus", False):
                reward += self.ma_fault_verify_bonus
            if action == Action.VERIFY and not event.get("invalid", False) and not event.get("wasted_verify", False):
                current_hint = self.risk_hints[self.current_step] if self.risk_hints else 0
                if current_hint == 1:
                    reward += self.ma_hint_verify_bonus
                elif not event.get("fault_verify_bonus", False):
                    reward -= self.ma_low_hint_penalty
            if (
                action == Action.VERIFY
                and not event.get("safe_verify_bonus", False)
                and not event.get("fault_verify_bonus", False)
            ):
                reward -= self.ma_waste_verify_penalty
            last_checked_curr = max(self.checked_steps) if self.checked_steps else 0
            unchecked_gap_curr = max(0, self.current_step - last_checked_curr)
            reward -= self.ma_opaque_penalty * ((unchecked_gap_curr - unchecked_gap_prev) / max(1, self.depth))
        self.total_actions += 1
        self.total_return += reward
        self.last_action = action
        self.revised_recently = action == Action.REVISE
        if action == Action.COMMIT:
            self.done = True
        if action in {Action.THINK, Action.REVISE}:
            self.last_verify_status = VerifyStatus.NOT_USED
        if action != Action.VERIFY and action != Action.REVISE:
            self.revised_recently = False
        info = {
            "action_name": ACTION_NAMES[action],
            "reward": reward,
            "trusted_prefix_before": trusted_prev,
            "trusted_prefix_after": self.trusted_prefix,
            "current_step": self.current_step,
            "remaining_budget": self.remaining_budget,
            "last_verify_status": self.last_verify_status.name,
            "current_state_valid": self.is_current_state_valid(),
            "fault_index": None if self.fault_index is None else self.fault_index + 1,
            "case_description": self.case_description,
        }
        if self.done:
            info["episode_metrics"] = self.get_episode_metrics()
        return self._make_obs(), reward, self.done, info

    def get_episode_metrics(self) -> dict[str, Any]:
        fault_present = self.fault_index is not None
        fault_step = None if self.fault_index is None else self.fault_index + 1
        localization_score = float("nan")
        if fault_present and not self.success:
            diagnosis_step = self.first_unsafe_verify_step or self.commit_step or self.current_step
            if diagnosis_step < fault_step:
                localization_score = 0.0
            else:
                denom = max(1, self.depth - fault_step)
                localization_score = float(np.clip(1.0 - ((diagnosis_step - fault_step) / denom), 0.0, 1.0))
        verify_precision = float("nan")
        if self.verify_count > 0:
            verify_precision = self.verify_tp / self.verify_count
        verify_recall = float("nan")
        if fault_present:
            verify_recall = 1.0 if self.verify_tp > 0 else 0.0
        return {
            "success": int(self.success),
            "episode_return": self.total_return,
            "total_actions": self.total_actions,
            "verify_count": self.verify_count,
            "revise_count": self.revise_count,
            "remaining_budget": self.remaining_budget,
            "used_budget": self.max_budget - self.remaining_budget,
            "depth": self.depth,
            "fault_present": int(fault_present),
            "fault_step": fault_step if fault_step is not None else -1,
            "failure_localization_score": localization_score,
            "verification_precision": verify_precision,
            "verification_recall": verify_recall,
            "verify_tp": self.verify_tp,
            "verify_fp": self.verify_fp,
            "cost_adjusted_accuracy": int(self.success) - self.cost_adjust_weight * self.total_action_cost,
            "commit_step": self.commit_step if self.commit_step is not None else self.current_step,
            "trusted_prefix": self.trusted_prefix,
            "case_description": self.case_description,
        }


def sanity_check_env() -> None:
    config = {
        "train_depth_range": [3, 3],
        "max_depth": 6,
        "budget_extra": 6,
        "p_fault": 1.0,
        "reward_scheme": "monitorability_aware",
    }
    env = ProcessReasoningEnv(config, seed=7)
    obs, info = env.reset(seed=9, fault_index=1, budget=10)
    assert info["fault_index"] == 2
    assert obs.shape[0] == env.observation_dim
    env.step(Action.THINK)
    env.step(Action.VERIFY)
    env.step(Action.THINK)
    _, _, _, info = env.step(Action.VERIFY)
    assert info["last_verify_status"] == "UNSAFE"
    env.step(Action.REVISE)
    env.step(Action.THINK)
    env.step(Action.THINK)
    _, _, done, info = env.step(Action.COMMIT)
    assert done
    assert info["episode_metrics"]["success"] == 1

    env = ProcessReasoningEnv(config, seed=7)
    env.reset(seed=9, fault_index=1, budget=10)
    env.step(Action.THINK)
    env.step(Action.THINK)
    env.step(Action.THINK)
    _, _, _, info = env.step(Action.COMMIT)
    assert info["episode_metrics"]["success"] == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()
    if args.sanity:
        sanity_check_env()
        print("env_sanity_ok")


if __name__ == "__main__":
    main()
