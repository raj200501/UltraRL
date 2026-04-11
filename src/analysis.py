from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from src.utils import ensure_dir, format_pm, load_yaml

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--summary-path", default="results/eval_summary.csv")
    args = parser.parse_args()
    load_yaml(args.config)
    summary = pd.read_csv(args.summary_path)
    figures_dir = ensure_dir("paper/figures")
    tables_dir = ensure_dir("paper/tables")

    plot_tradeoff(summary, figures_dir)
    plot_bar_comparison(summary, figures_dir)
    plot_generalization(summary, figures_dir)
    plot_verification(summary, figures_dir)
    write_main_table(summary, tables_dir)
    write_ablation_table(summary, tables_dir)


if __name__ == "__main__":
    main()
