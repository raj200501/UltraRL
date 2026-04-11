from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from src.agents import DQNAgent, build_heuristic
from src.envs import Action, ProcessReasoningEnv
from src.evaluate import HEURISTIC_METHODS, setting_to_overrides
from src.utils import ensure_dir, format_pm, load_yaml, merge_dict, sem

os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir("results/mpl_cache")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



DISPLAY_ORDER = [
    "Random",
    "Always THINK",
    "Verify Every Step",
    "Hint-Triggered Heuristic",
    "Outcome-only RL",
    "Naive Process RL",
    "Monitorability-aware RL",
]

COLORS = {
    "Random": "#6b7280",
    "Always THINK": "#4b5563",
    "Verify Every Step": "#b45309",
    "Hint-Triggered Heuristic": "#7c2d12",
    "Outcome-only RL": "#1d4ed8",
    "Naive Process RL": "#0f766e",
    "Monitorability-aware RL": "#b91c1c",
    "Monitorability RL w/o Budget Penalty": "#7c3aed",
    "Monitorability RL w/o VERIFY": "#be123c",
    "Monitorability RL w/o REVISE": "#065f46",
    "Monitorability RL w/o Monitorability Term": "#92400e",
}


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {name: idx for idx, name in enumerate(DISPLAY_ORDER)}
    fallback = len(order_map)
    return df.sort_values(by="display_name", key=lambda column: column.map(order_map).fillna(fallback))


def build_main_methods(config: dict[str, Any]) -> list[dict[str, Any]]:
    methods = [dict(item) for item in HEURISTIC_METHODS]
    methods.extend(dict(item) for item in config.get("rl_variants", []))
    return methods


def build_selectivity_summary(config: dict[str, Any], checkpoint_dir: Path) -> pd.DataFrame:
    heuristic_names = {item["name"] for item in HEURISTIC_METHODS}
    methods = build_main_methods(config)
    selected_settings = {
        item["name"]: item
        for item in config["evaluation"]["settings"]
        if item["name"] in {"train_like", "heldout_long"}
    }
    rows: list[dict[str, Any]] = []
    for method in methods:
        for setting_name, setting in selected_settings.items():
            for seed in config["training"]["seeds"]:
                env_config = merge_dict(config["env"], method.get("env_overrides", {}))
                env_config = merge_dict(env_config, setting_to_overrides(setting))
                env = ProcessReasoningEnv(env_config, seed=int(seed))
                policy = (
                    build_heuristic(method["name"], int(seed))
                    if method["name"] in heuristic_names
                    else DQNAgent.load(
                        checkpoint_dir / f"{method['name']}_seed{seed}.pt",
                        obs_dim=env.observation_dim,
                        action_dim=env.action_dim,
                        hidden_size=int(config["training"]["hidden_size"]),
                        device=torch.device("cpu"),
                    )[0]
                )
                for episode_idx in range(int(config["evaluation"]["episodes_per_seed"])):
                    obs, _ = env.reset(
                        seed=int(seed) * 1_000_000 + episode_idx,
                        depth_range=tuple(setting["depth_range"]),
                        p_fault=setting.get("p_fault"),
                    )
                    done = False
                    while not done:
                        action_mask = env.available_actions()
                        if method["name"] in heuristic_names:
                            action = policy.act(obs, env, action_mask)
                        else:
                            action = policy.act(
                                obs,
                                action_mask,
                                epsilon=0.0,
                                rng=np.random.default_rng(int(seed)),
                                deterministic=True,
                            )
                        if bool(action_mask[Action.VERIFY]):
                            rows.append(
                                {
                                    "setting": setting_name,
                                    "method": method["name"],
                                    "display_name": method.get("display_name", method["name"]),
                                    "seed": int(seed),
                                    "cue_value": int(env.risk_hints[env.current_step]) if env.risk_hints else 0,
                                    "choose_verify": int(action == Action.VERIFY),
                                }
                            )
                        obs, _, done, _ = env.step(action)
    raw = pd.DataFrame(rows)
    summary_rows = []
    for (setting, display_name, seed, cue_value), group in raw.groupby(
        ["setting", "display_name", "seed", "cue_value"], sort=False
    ):
        summary_rows.append(
            {
                "setting": setting,
                "display_name": display_name,
                "seed": seed,
                "cue_value": cue_value,
                "verify_rate": float(group["choose_verify"].mean()),
                "support": int(len(group)),
            }
        )
    seed_df = pd.DataFrame(summary_rows)
    agg_rows = []
    for (setting, display_name, cue_value), group in seed_df.groupby(["setting", "display_name", "cue_value"], sort=False):
        agg_rows.append(
            {
                "setting": setting,
                "display_name": display_name,
                "cue_value": cue_value,
                "verify_rate_mean": float(group["verify_rate"].mean()),
                "verify_rate_se": sem(group["verify_rate"]),
                "support_mean": float(group["support"].mean()),
            }
        )
    return pd.DataFrame(agg_rows)


def plot_tradeoff(summary: pd.DataFrame, figures_dir: Path) -> None:
    subset = summary[summary["setting"].str.startswith("verify_cost_")].copy()
    subset = subset[subset["display_name"].isin(DISPLAY_ORDER)]
    subset = ordered(subset)
    plt.figure(figsize=(7.0, 4.2))
    for display_name, group in subset.groupby("display_name", sort=False):
        group = group.sort_values("verify_cost_setting")
        plt.plot(
            group["verify_cost_setting"],
            group["cost_adjusted_accuracy_mean"],
            marker="o",
            linewidth=2.0,
            color=COLORS[display_name],
            label=display_name,
        )
    plt.xlabel("Verification Cost")
    plt.ylabel("Cost-adjusted Accuracy")
    plt.title("Held-out Tradeoff Under Higher Verification Cost")
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "tradeoff_verify_cost.pdf")
    plt.savefig(figures_dir / "tradeoff_verify_cost.png", dpi=240)
    plt.close()


def plot_bar_comparison(summary: pd.DataFrame, figures_dir: Path) -> None:
    subset = ordered(summary[summary["setting"] == "train_like"].copy())
    subset = subset[subset["display_name"].isin(DISPLAY_ORDER)]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))
    x = range(len(subset))
    success_colors = [COLORS[name] for name in subset["display_name"]]
    axes[0].bar(x, subset["success_rate_mean"], yerr=subset["success_rate_se"], color=success_colors, capsize=3)
    axes[0].set_ylabel("Success Rate")
    axes[0].set_xticks(list(x), subset["display_name"], rotation=35, ha="right")
    axes[0].set_title("Train-like Success")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)

    axes[1].bar(
        x,
        subset["failure_localization_score_mean"],
        yerr=subset["failure_localization_score_se"],
        color=success_colors,
        capsize=3,
    )
    axes[1].set_ylabel("Failure Localization")
    axes[1].set_xticks(list(x), subset["display_name"], rotation=35, ha="right")
    axes[1].set_title("Train-like Failed-episode Localization")
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(figures_dir / "method_comparison.pdf")
    plt.savefig(figures_dir / "method_comparison.png", dpi=240)
    plt.close(fig)


def plot_generalization(summary: pd.DataFrame, figures_dir: Path) -> None:
    subset = summary[summary["setting"].isin(["train_like", "heldout_long"])].copy()
    subset = subset[subset["display_name"].isin(DISPLAY_ORDER)]
    subset = ordered(subset)
    setting_order = {"train_like": 0, "heldout_long": 1}
    plt.figure(figsize=(7.0, 4.1))
    for display_name, group in subset.groupby("display_name", sort=False):
        group = group.sort_values("setting", key=lambda col: col.map(setting_order))
        plt.plot(
            [setting_order[item] for item in group["setting"]],
            group["success_rate_mean"],
            marker="o",
            linewidth=2.0,
            color=COLORS[display_name],
            label=display_name,
        )
    plt.xticks([0, 1], ["Train-like Depths", "Held-out Longer Depths"])
    plt.ylabel("Success Rate")
    plt.title("Generalization from Short to Longer Chains")
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "generalization.pdf")
    plt.savefig(figures_dir / "generalization.png", dpi=240)
    plt.close()


def plot_verification(summary: pd.DataFrame, figures_dir: Path) -> None:
    subset = ordered(summary[summary["setting"] == "train_like"].copy())
    subset = subset[subset["display_name"].isin(DISPLAY_ORDER)]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    x = range(len(subset))
    colors = [COLORS[name] for name in subset["display_name"]]
    axes[0].bar(x, subset["verification_precision_mean"], yerr=subset["verification_precision_se"], color=colors, capsize=3)
    axes[0].set_ylabel("Verification Precision")
    axes[0].set_xticks(list(x), subset["display_name"], rotation=35, ha="right")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[1].bar(x, subset["verification_recall_mean"], yerr=subset["verification_recall_se"], color=colors, capsize=3)
    axes[1].set_ylabel("Verification Recall")
    axes[1].set_xticks(list(x), subset["display_name"], rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(figures_dir / "verification_diagnostics.pdf")
    plt.savefig(figures_dir / "verification_diagnostics.png", dpi=240)
    plt.close(fig)


def plot_selectivity(selectivity_summary: pd.DataFrame, figures_dir: Path) -> None:
    order = [name for name in DISPLAY_ORDER if name in set(selectivity_summary["display_name"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), sharey=True)
    width = 0.38
    cue_colors = {0: "#9ca3af", 1: "#111827"}
    cue_labels = {0: "Cue = 0", 1: "Cue = 1"}
    for axis, setting_name, title in zip(
        axes,
        ["train_like", "heldout_long"],
        ["Train-like Depths", "Held-out Longer Depths"],
    ):
        subset = selectivity_summary[selectivity_summary["setting"] == setting_name].copy()
        x = np.arange(len(order))
        for cue_value, offset in [(0, -width / 2), (1, width / 2)]:
            cue_subset = (
                subset[subset["cue_value"] == cue_value]
                .set_index("display_name")
                .reindex(order)
                .reset_index()
            )
            axis.bar(
                x + offset,
                cue_subset["verify_rate_mean"].fillna(0.0),
                width=width,
                yerr=cue_subset["verify_rate_se"].fillna(0.0),
                color=cue_colors[cue_value],
                capsize=3,
                label=cue_labels[cue_value],
            )
        axis.set_title(title)
        axis.set_xticks(x, order, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel(r"$P(\mathrm{VERIFY}\mid\mathrm{cue})$")
    axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "selectivity_diagnostics.pdf")
    plt.savefig(figures_dir / "selectivity_diagnostics.png", dpi=240)
    plt.close(fig)


def write_main_table(summary: pd.DataFrame, tables_dir: Path) -> None:
    train_df = ordered(summary[summary["setting"] == "train_like"].copy())
    heldout_df = ordered(summary[summary["setting"] == "heldout_long"].copy())
    train_df = train_df[train_df["display_name"].isin(DISPLAY_ORDER)]
    heldout_df = heldout_df[heldout_df["display_name"].isin(DISPLAY_ORDER)]
    table_rows = []
    for display_name in train_df["display_name"]:
        train_row = train_df[train_df["display_name"] == display_name].iloc[0]
        heldout_row = heldout_df[heldout_df["display_name"] == display_name].iloc[0]
        table_rows.append(
            {
                "Method": display_name,
                "TrainSucc": format_pm(train_row["success_rate_mean"], train_row["success_rate_se"]),
                "TrainLoc": format_pm(train_row["failure_localization_score_mean"], train_row["failure_localization_score_se"]),
                "HeldSucc": format_pm(heldout_row["success_rate_mean"], heldout_row["success_rate_se"]),
                "HeldLoc": format_pm(heldout_row["failure_localization_score_mean"], heldout_row["failure_localization_score_se"]),
                "HeldCostAdj": format_pm(heldout_row["cost_adjusted_accuracy_mean"], heldout_row["cost_adjusted_accuracy_se"]),
            }
        )
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(tables_dir / "main_results.csv", index=False)
    tex_lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & TrainSucc & TrainLoc & HeldSucc & HeldLoc & HeldCostAdj \\\\",
        "\\midrule",
    ]
    for _, row in table_df.iterrows():
        tex_lines.append(
            f"{row['Method']} & {row['TrainSucc']} & {row['TrainLoc']} & {row['HeldSucc']} & {row['HeldLoc']} & {row['HeldCostAdj']} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "main_results.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def write_ablation_table(summary: pd.DataFrame, tables_dir: Path) -> None:
    subset = summary[
        summary["display_name"].isin(
            [
                "Monitorability-aware RL",
                "Monitorability RL w/o Budget Penalty",
                "Monitorability RL w/o VERIFY",
                "Monitorability RL w/o REVISE",
                "Monitorability RL w/o Monitorability Term",
            ]
        )
        & (summary["setting"] == "heldout_long")
    ].copy()
    subset = subset.sort_values("display_name")
    subset.to_csv(tables_dir / "ablation_results.csv", index=False)


def write_cue_shift_table(summary: pd.DataFrame, tables_dir: Path) -> None:
    subset = summary[
        (summary["setting"].isin(["cue_clean", "cue_noisy"]))
        & (summary["display_name"].isin(
            [
                "Hint-Triggered Heuristic",
                "Outcome-only RL",
                "Naive Process RL",
                "Monitorability-aware RL",
            ]
        ))
    ].copy()
    subset = ordered(subset)
    rows = []
    for display_name in subset["display_name"].drop_duplicates():
        clean_row = subset[(subset["display_name"] == display_name) & (subset["setting"] == "cue_clean")].iloc[0]
        noisy_row = subset[(subset["display_name"] == display_name) & (subset["setting"] == "cue_noisy")].iloc[0]
        rows.append(
            {
                "Method": display_name,
                "CleanSucc": format_pm(clean_row["success_rate_mean"], clean_row["success_rate_se"]),
                "NoisySucc": format_pm(noisy_row["success_rate_mean"], noisy_row["success_rate_se"]),
                "CleanLoc": format_pm(clean_row["failure_localization_score_mean"], clean_row["failure_localization_score_se"]),
                "NoisyLoc": format_pm(noisy_row["failure_localization_score_mean"], noisy_row["failure_localization_score_se"]),
            }
        )
    table_df = pd.DataFrame(rows)
    table_df.to_csv(tables_dir / "cue_shift_results.csv", index=False)
    tex_lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & CleanSucc & NoisySucc & CleanLoc & NoisyLoc \\\\",
        "\\midrule",
    ]
    for _, row in table_df.iterrows():
        tex_lines.append(
            f"{row['Method']} & {row['CleanSucc']} & {row['NoisySucc']} & {row['CleanLoc']} & {row['NoisyLoc']} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "cue_shift_results.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--summary-path", default="results/eval_summary.csv")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    args = parser.parse_args()
    config = load_yaml(args.config)
    summary = pd.read_csv(args.summary_path)
    figures_dir = ensure_dir("paper/figures")
    tables_dir = ensure_dir("paper/tables")
    checkpoint_dir = Path(args.checkpoint_dir)
    selectivity_summary = build_selectivity_summary(config, checkpoint_dir)
    selectivity_summary.to_csv(tables_dir / "selectivity_results.csv", index=False)

    plot_tradeoff(summary, figures_dir)
    plot_bar_comparison(summary, figures_dir)
    plot_generalization(summary, figures_dir)
    plot_verification(summary, figures_dir)
    plot_selectivity(selectivity_summary, figures_dir)
    write_main_table(summary, tables_dir)
    write_ablation_table(summary, tables_dir)
    write_cue_shift_table(summary, tables_dir)


if __name__ == "__main__":
    main()
