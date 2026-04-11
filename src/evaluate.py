from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents import DQNAgent, build_heuristic
from src.envs import ProcessReasoningEnv
from src.utils import choose_device, ensure_dir, load_yaml, mean_ignore_nan, merge_dict, sem


HEURISTIC_METHODS = [
    {"name": "random", "display_name": "Random"},
    {"name": "always_think_commit", "display_name": "Always THINK"},
    {"name": "verify_every_step", "display_name": "Verify Every Step"},
    {"name": "hint_triggered", "display_name": "Hint-Triggered Heuristic"},
]


def build_method_specs(default_cfg: dict[str, Any], ablations_cfg: dict[str, Any], include_ablations: bool) -> list[dict[str, Any]]:
    methods = list(HEURISTIC_METHODS)
    methods.extend(dict(item) for item in default_cfg.get("rl_variants", []))
    if include_ablations:
        methods.extend(dict(item) for item in ablations_cfg.get("train_variants", []))
    return methods


def setting_to_overrides(setting: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "p_fault",
        "verify_cost",
        "budget_extra",
        "allow_verify",
        "allow_revise",
        "budget_penalty",
        "hint_reliability",
        "hint_false_positive",
    ]
    return {key: setting[key] for key in keys if key in setting}


def evaluate_policy(
    method: dict[str, Any],
    seed: int,
    setting: dict[str, Any],
    config: dict[str, Any],
    device,
    episodes_per_seed: int,
    checkpoint_dir: Path,
) -> list[dict[str, Any]]:
    env_config = merge_dict(config["env"], method.get("env_overrides", {}))
    env_config = merge_dict(env_config, setting_to_overrides(setting))
    env = ProcessReasoningEnv(env_config, seed=seed)
    depth_range = tuple(setting["depth_range"])
    if method["name"] in {item["name"] for item in HEURISTIC_METHODS}:
        policy = build_heuristic(method["name"], seed)
        hidden_size = 0
        parameter_count = 0
    else:
        checkpoint_path = checkpoint_dir / f"{method['name']}_seed{seed}.pt"
        agent, metadata = DQNAgent.load(
            checkpoint_path,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_size=int(config["training"]["hidden_size"]),
            device=device,
        )
        policy = agent
        hidden_size = metadata.get("hidden_size", 0)
        parameter_count = metadata.get("parameter_count", 0)
    rows: list[dict[str, Any]] = []
    for episode_idx in range(episodes_per_seed):
        obs, _ = env.reset(
            seed=seed * 1_000_000 + episode_idx,
            depth_range=depth_range,
            p_fault=setting.get("p_fault"),
        )
        done = False
        info = {}
        while not done:
            action_mask = env.available_actions()
            if method["name"] in {item["name"] for item in HEURISTIC_METHODS}:
                action = policy.act(obs, env, action_mask)
            else:
                action = policy.act(obs, action_mask, epsilon=0.0, rng=np.random.default_rng(seed), deterministic=True)
            obs, _, done, info = env.step(action)
        metrics = dict(info["episode_metrics"])
        metrics.update(
            {
                "method": method["name"],
                "display_name": method.get("display_name", method["name"]),
                "seed": seed,
                "setting": setting["name"],
                "depth_min": depth_range[0],
                "depth_max": depth_range[1],
                "verify_cost_setting": env_config["verify_cost"],
                "budget_extra_setting": env_config["budget_extra"],
                "fault_probability_setting": setting.get("p_fault", env_config["p_fault"]),
                "hidden_size": hidden_size,
                "parameter_count": parameter_count,
            }
        )
        rows.append(metrics)
    return rows


def build_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["setting", "method", "display_name", "seed"]
    for keys, group in df.groupby(group_cols, sort=False):
        setting, method, display_name, seed = keys
        fault_group = group[group["fault_present"] == 1]
        train_like = setting == "train_like"
        rows.append(
            {
                "setting": setting,
                "method": method,
                "display_name": display_name,
                "seed": seed,
                "success_rate": group["success"].mean(),
                "avg_return": group["episode_return"].mean(),
                "avg_total_actions": group["total_actions"].mean(),
                "avg_verify_count": group["verify_count"].mean(),
                "avg_revise_count": group["revise_count"].mean(),
                "success_under_fixed_budget": group["success"].mean(),
                "failure_localization_score": mean_ignore_nan(group["failure_localization_score"]),
                "verification_precision": group["verify_tp"].sum() / max(1, group["verify_count"].sum()),
                "verification_recall": fault_group["verification_recall"].mean() if len(fault_group) else np.nan,
                "cost_adjusted_accuracy": group["cost_adjusted_accuracy"].mean(),
                "generalization_gap_proxy": np.nan if train_like else np.nan,
                "fault_present_rate": group["fault_present"].mean(),
                "verify_cost_setting": group["verify_cost_setting"].iloc[0],
                "budget_extra_setting": group["budget_extra_setting"].iloc[0],
                "fault_probability_setting": group["fault_probability_setting"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def build_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_names = [
        "success_rate",
        "avg_return",
        "avg_total_actions",
        "avg_verify_count",
        "avg_revise_count",
        "success_under_fixed_budget",
        "failure_localization_score",
        "verification_precision",
        "verification_recall",
        "cost_adjusted_accuracy",
    ]
    for (setting, method, display_name), group in seed_summary.groupby(["setting", "method", "display_name"], sort=False):
        row = {
            "setting": setting,
            "method": method,
            "display_name": display_name,
            "verify_cost_setting": group["verify_cost_setting"].iloc[0],
            "budget_extra_setting": group["budget_extra_setting"].iloc[0],
            "fault_probability_setting": group["fault_probability_setting"].iloc[0],
        }
        for metric in metric_names:
            row[f"{metric}_mean"] = mean_ignore_nan(group[metric])
            row[f"{metric}_se"] = sem(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablations", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--output-dir", default="results/eval")
    parser.add_argument("--all-episodes-path", default="results/eval_all_episodes.csv")
    parser.add_argument("--seed-summary-path", default="results/eval_seed_summary.csv")
    parser.add_argument("--summary-path", default="results/eval_summary.csv")
    args = parser.parse_args()

    config = load_yaml(args.config)
    ablations_cfg = load_yaml(args.ablations) if args.ablations else {}
    include_ablations = args.mode == "full"
    methods = build_method_specs(config, ablations_cfg, include_ablations)
    eval_cfg = config["evaluation"]
    settings = eval_cfg["settings"]
    episodes_per_seed = int(eval_cfg["smoke_episodes_per_seed"] if args.mode == "smoke" else eval_cfg["episodes_per_seed"])
    seeds = config["training"]["smoke_seeds"] if args.mode == "smoke" else config["training"]["seeds"]
    device = choose_device(prefer_mps=bool(config["training"].get("prefer_mps", True)))

    checkpoint_dir = Path(args.checkpoint_dir)
    eval_dir = ensure_dir(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    for setting in settings:
        setting_rows = []
        print(f"[eval] setting={setting['name']} episodes_per_seed={episodes_per_seed}")
        for method in methods:
            for seed in seeds:
                setting_rows.extend(
                    evaluate_policy(
                        method=method,
                        seed=int(seed),
                        setting=setting,
                        config=config,
                        device=device,
                        episodes_per_seed=episodes_per_seed,
                        checkpoint_dir=checkpoint_dir,
                    )
                )
        setting_df = pd.DataFrame(setting_rows)
        setting_df.to_csv(eval_dir / f"{setting['name']}.csv", index=False)
        all_rows.extend(setting_rows)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(args.all_episodes_path, index=False)
    seed_summary = build_seed_summary(all_df)
    summary = build_summary(seed_summary)
    seed_summary.to_csv(args.seed_summary_path, index=False)
    summary.to_csv(args.summary_path, index=False)


if __name__ == "__main__":
    main()
