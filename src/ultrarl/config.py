"""Typed configuration objects and YAML loading."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EnvConfig:
    # Task distribution
    train_depth_range: tuple[int, int] = (3, 6)
    max_depth: int = 9  # normaliser for the absolute observation encoder only
    p_fault: float = 0.8
    fault_profile: str = "middle"  # "middle" or "uniform"
    fault_peak_weight: float = 5.0
    cue_hit_rate: float = 0.85
    cue_false_positive_rate: float = 0.15
    # Budget and action costs
    budget_extra: int = 6
    think_cost: int = 1
    verify_cost: int = 1
    revise_cost: int = 1
    allow_verify: bool = True
    allow_revise: bool = True
    # Observation encoder ("absolute" or "invariant")
    observation: str = "absolute"
    # Base reward
    reward_scheme: str = "outcome_only"
    correct_reward: float = 1.0
    incorrect_reward: float = -1.2
    budget_penalty: bool = True
    step_penalty: float = 0.04
    cost_adjust_weight: float = 0.05  # metric only, not a reward
    # Naive process reward
    naive_progress_bonus: float = 0.015
    naive_verify_bonus: float = 0.02
    # Monitorability-aware process reward
    ma_safe_verify_bonus: float = 0.02
    ma_fault_verify_bonus: float = 0.25
    ma_cue_verify_bonus: float = 0.18
    ma_low_cue_penalty: float = 0.02
    ma_waste_verify_penalty: float = 0.06
    ma_gap_penalty: float = 0.30

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_depth_range", tuple(self.train_depth_range))
        low, high = self.train_depth_range
        if not 1 <= low <= high:
            raise ValueError(f"invalid train_depth_range {self.train_depth_range}")
        if min(self.think_cost, self.verify_cost, self.revise_cost) < 1:
            raise ValueError("action costs must be >= 1 so that episodes terminate")
        for name in ("p_fault", "cue_hit_rate", "cue_false_positive_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be a probability")
        if self.fault_profile not in ("middle", "uniform"):
            raise ValueError(f"unknown fault_profile {self.fault_profile!r}")

    def replace(self, **overrides: Any) -> EnvConfig:
        unknown = set(overrides) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise KeyError(f"unknown EnvConfig fields: {sorted(unknown)}")
        return dataclasses.replace(self, **overrides)


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 20_000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    hidden_size: int = 64
    replay_size: int = 25_000
    batch_size: int = 64
    warmup_steps: int = 500
    target_update_interval: int = 200  # hard target sync period (env steps); used when target_tau == 0
    target_tau: float = 0.0  # > 0: Polyak-average the target network every step instead
    epsilon_start: float = 0.5
    epsilon_end: float = 0.02
    epsilon_decay_fraction: float = 0.85
    grad_clip: float = 1.0
    curve_interval: int = 1_000  # episodes between greedy learning-curve evaluations
    curve_episodes: int = 300  # validation episodes per learning-curve point and setting
    # Checkpoint selection uses only this curve setting, which must lie inside the
    # training length range so that held-out lengths never influence selection.
    select_setting: str = "train_like"


@dataclass(frozen=True)
class EvalSetting:
    name: str
    depth_range: tuple[int, int]
    env_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Method:
    name: str
    display_name: str
    kind: str  # "heuristic" or "dqn"
    env_overrides: dict[str, Any] = field(default_factory=dict)
    train_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    env: EnvConfig
    train: TrainConfig
    seeds: tuple[int, ...]
    methods: tuple[Method, ...]
    eval_settings: tuple[EvalSetting, ...]
    eval_episodes: int
    curve_settings: tuple[EvalSetting, ...]

    @property
    def learned_methods(self) -> tuple[Method, ...]:
        return tuple(m for m in self.methods if m.kind == "dqn")

    def method_train(self, method: Method) -> TrainConfig:
        return dataclasses.replace(self.train, **method.train_overrides)

    def method_env(self, method: Method, setting: EvalSetting | None = None) -> EnvConfig:
        env = self.env.replace(**method.env_overrides)
        if setting is not None:
            env = env.replace(**setting.env_overrides)
        return env


def _setting(raw: dict[str, Any]) -> EvalSetting:
    raw = dict(raw)
    return EvalSetting(
        name=raw.pop("name"),
        depth_range=tuple(raw.pop("depth_range")),
        env_overrides=raw.pop("env_overrides", {}),
    )


def load_experiment(path: str | Path, *, smoke: bool = False) -> ExperimentConfig:
    """Load an experiment YAML. ``smoke=True`` applies the file's ``smoke`` overrides."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if smoke:
        for section, overrides in raw.get("smoke", {}).items():
            if isinstance(raw.get(section), dict):
                raw[section] = {**raw[section], **overrides}
            else:
                raw[section] = overrides
    env = EnvConfig(**raw.get("env", {}))
    train = TrainConfig(**raw.get("train", {}))
    methods = tuple(
        Method(
            name=m["name"],
            display_name=m.get("display_name", m["name"]),
            kind=m.get("kind", "dqn"),
            env_overrides=m.get("env_overrides", {}),
            train_overrides=m.get("train_overrides", {}),
        )
        for m in raw["methods"]
    )
    names = [m.name for m in methods]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate method names in {path}")
    for method in methods:  # validate overrides eagerly
        env.replace(**method.env_overrides)
        dataclasses.replace(train, **method.train_overrides)
    evaluation = raw["evaluation"]
    exp = ExperimentConfig(
        name=raw.get("name", Path(path).stem),
        env=env,
        train=train,
        seeds=tuple(int(s) for s in raw["seeds"]),
        methods=methods,
        eval_settings=tuple(_setting(s) for s in evaluation["settings"]),
        eval_episodes=int(evaluation["episodes"]),
        curve_settings=tuple(_setting(s) for s in evaluation.get("curve_settings", [])),
    )
    _check_selection_setting(exp)
    return exp


def _check_selection_setting(exp: ExperimentConfig) -> None:
    if not exp.learned_methods:
        return
    for method in exp.learned_methods:
        name = exp.method_train(method).select_setting
        matches = [s for s in exp.curve_settings if s.name == name]
        if not matches:
            raise ValueError(f"select_setting {name!r} is not one of the curve_settings")
        low, high = matches[0].depth_range
        train_low, train_high = exp.method_env(method).train_depth_range
        if low < train_low or high > train_high:
            raise ValueError(f"select_setting {name!r} uses lengths outside the training range")
