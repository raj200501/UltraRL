"""Figures and LaTeX tables for the paper, built only from files under ``results/``.

Usage::

    python -m ultrarl.report --results results --paper paper
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ultrarl.stats import bootstrap_ci, paired_difference  # noqa: E402

# Categorical slots in fixed order (one hue per reward scheme); observation
# encoder is a secondary encoding (line style / hatch), heuristics are neutral.
SCHEME_COLOR = {"outcome": "#2a78d6", "naive": "#eb6834", "ma": "#1baf7a"}
SCHEME_LABEL = {"outcome": "Outcome-only", "naive": "Naive process", "ma": "Monitorability-aware"}
OBS_STYLE = {"inv": "-", "abs": "--"}
OBS_LABEL = {"inv": "invariant obs.", "abs": "absolute obs."}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HEURISTIC_STYLE = {
    "cue_triggered": dict(color="#52514e", linestyle=":", marker="s"),
    "budget_aware_1": dict(color="#9a9893", linestyle=":", marker="D"),
    "verify_every_step": dict(color="#9a9893", linestyle=":", marker="^"),
    "always_think": dict(color="#c3c2b7", linestyle=":", marker="v"),
}
LEARNED = [f"{s}_{o}" for o in ("inv", "abs") for s in ("outcome", "naive", "ma")]
PANELS = [("train_like", "Train lengths (3–6)"), ("heldout_long", "Held-out lengths (7–9)")]
LENGTH_SETTINGS = [("train_like", "3–6\n(train)"), ("heldout_long", "7–9"), ("heldout_xlong", "10–14")]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.5,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
    }
)


def split(method: str) -> tuple[str, str]:
    scheme, obs = method.rsplit("_", 1)
    return scheme, obs


def label(method: str) -> str:
    scheme, obs = split(method)
    return f"{SCHEME_LABEL[scheme]} ({OBS_LABEL[obs]})"


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def row(summary: pd.DataFrame, setting: str, method: str) -> pd.Series:
    sel = summary[(summary.setting == setting) & (summary.method == method)]
    if len(sel) != 1:
        raise KeyError(f"no unique row for setting={setting} method={method}")
    return sel.iloc[0]


# --------------------------------------------------------------------- figures
def fig_length_generalization(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = np.arange(len(LENGTH_SETTINGS))
    for method in LEARNED:
        scheme, obs = split(method)
        r = [row(summary, s, method) for s, _ in LENGTH_SETTINGS]
        y = np.array([v["success"] for v in r])
        err = np.array([[v["success"] - v["success_lo"], v["success_hi"] - v["success"]] for v in r]).T
        ax.errorbar(
            x, y, yerr=err, color=SCHEME_COLOR[scheme], linestyle=OBS_STYLE[obs], marker="o", markersize=4,
            linewidth=1.6, capsize=2, label=label(method),
        )  # fmt: skip
    for name, text in (("cue_triggered", "Cue-triggered heuristic"), ("budget_aware_1", "Budget-aware heuristic")):
        y = [row(summary, s, name)["success"] for s, _ in LENGTH_SETTINGS]
        ax.plot(x, y, linewidth=1.2, markersize=4, label=text, **HEURISTIC_STYLE[name])
    never = np.mean([row(summary, s, "always_think")["success"] for s, _ in LENGTH_SETTINGS])
    ax.axhline(never, color=INK_2, linewidth=0.8, linestyle=(0, (1, 3)))
    ax.annotate("never checks", (x[-1], never), xytext=(-2, 3), textcoords="offset points", ha="right",
                fontsize=6.5, color=INK_2)  # fmt: skip
    ax.set_xticks(x, [lab for _, lab in LENGTH_SETTINGS])
    ax.set_xlabel("Chain length")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, out, "length_generalization")


def load_curves(results: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((results / "curves").glob("*.csv"))]
    return pd.concat(frames, ignore_index=True)


def curve_band(curves: pd.DataFrame, method: str, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sub = curves[curves.method == method]
    episodes = np.sort(sub.episode.unique())
    means, los, his = [], [], []
    for ep in episodes:
        vals = sub.loc[sub.episode == ep, column].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals)
        means.append(np.nanmean(vals))
        los.append(lo)
        his.append(hi)
    return episodes, np.array(means), np.array(los), np.array(his)


def fig_learning_curves(curves: pd.DataFrame, summary: pd.DataFrame, out: Path, methods: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (setting, title) in zip(axes, PANELS, strict=True):
        for method in methods:
            scheme, obs = split(method)
            ep, mean, lo, hi = curve_band(curves, method, f"{setting}_success")
            ax.plot(
                ep / 1000,
                mean,
                color=SCHEME_COLOR[scheme],
                linestyle=OBS_STYLE[obs],
                linewidth=1.5,
                label=label(method),
            )
            ax.fill_between(ep / 1000, lo, hi, color=SCHEME_COLOR[scheme], alpha=0.12, linewidth=0)
        ref = row(summary, setting, "cue_triggered")["success"]
        style = {k: v for k, v in HEURISTIC_STYLE["cue_triggered"].items() if k != "marker"}
        ax.axhline(ref, linewidth=1.0, label="Cue-triggered heuristic", **style)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Training episodes (thousands)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Greedy success rate")
    axes[1].legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, out, "learning_curves")


def fig_churn(curves: pd.DataFrame, out: Path, methods: tuple[str, ...] = ("ma_inv", "ma_abs")) -> None:
    """Per-seed greedy success during training: policies switch abruptly between solutions."""
    fig, axes = plt.subplots(1, len(methods), figsize=(7.0, 2.4), sharey=True)
    for ax, method in zip(axes, methods, strict=True):
        scheme, _ = split(method)
        sub = curves[curves.method == method]
        for _, run in sub.groupby("seed"):
            run = run.sort_values("episode")
            ax.plot(run.episode / 1000, run.train_like_success, color=SCHEME_COLOR[scheme], linewidth=0.8, alpha=0.55)
        ax.axhline(COLLAPSE_THRESHOLD, color=INK_2, linewidth=0.8, linestyle=(0, (1, 3)))
        ax.set_title(label(method), fontsize=8.5)
        ax.set_xlabel("Training episodes (thousands)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Greedy success, lengths 3–6")
    save(fig, out, "policy_churn")


def fig_selectivity(summary: pd.DataFrame, out: Path) -> None:
    methods = LEARNED + ["cue_triggered", "verify_every_step"]
    names = [label(m) if m in LEARNED else m.replace("_", " ").capitalize() + " (heuristic)" for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (setting, title) in zip(axes, PANELS, strict=True):
        for yi, method in zip(y, methods, strict=True):
            r = row(summary, setting, method)
            c0, c1 = r["verify_rate_cue0"], r["verify_rate_cue1"]
            color = SCHEME_COLOR[split(method)[0]] if method in LEARNED else INK_2
            ax.plot([c0, c1], [yi, yi], color=GRID, linewidth=2.0, zorder=1)
            ax.scatter([c0], [yi], s=22, facecolor="white", edgecolor=color, linewidth=1.2, zorder=2)
            ax.scatter([c1], [yi], s=22, color=color, zorder=3)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("P(VERIFY | unchecked step)")
    axes[0].set_yticks(y, names, fontsize=7)
    axes[0].scatter([], [], s=22, facecolor="white", edgecolor=INK_2, label="unchecked step, cue = 0")
    axes[0].scatter([], [], s=22, color=INK_2, label="unchecked step, cue = 1")
    fig.legend(fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.62, 1.04))
    save(fig, out, "cue_selectivity")


# ---------------------------------------------------------------------- tables
def fmt(r: pd.Series, metric: str, digits: int = 2) -> str:
    v, lo, hi = r[metric], r[f"{metric}_lo"], r[f"{metric}_hi"]
    if np.isnan(v):
        return "--"
    return f"{v:.{digits}f}\\,{{\\scriptsize[{lo:.{digits}f}, {hi:.{digits}f}]}}"


def latex_table(header: list[str], rows: list[list[str]], align: str) -> str:
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [r[0] if len(r) == 1 else " & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def display(summary: pd.DataFrame, method: str) -> str:
    return summary.loc[summary.method == method, "display_name"].iloc[0]


FINAL = "@final"


def table_main(summary: pd.DataFrame, out: Path, suffix: str = "", name: str = "main_results") -> None:
    heuristics = ["random", "always_think", "verify_every_step", "cue_triggered", "budget_aware_1"]
    methods = heuristics + [m + suffix for m in LEARNED]
    rows = []
    for m in methods:
        if m == LEARNED[0] + suffix:
            rows.append(["\\midrule"])
        rows.append(
            [
                display(summary, m).removesuffix(FINAL),
                fmt(row(summary, "train_like", m), "success"),
                fmt(row(summary, "heldout_long", m), "success"),
                fmt(row(summary, "heldout_xlong", m), "success"),
                fmt(row(summary, "heldout_long", m), "fault_detected"),
                fmt(row(summary, "heldout_long", m), "cost_adjusted_accuracy"),
            ]
        )
    header = ["Method", "Success 3--6", "Success 7--9", "Success 10--14", "Detect 7--9", "Cost-adj. 7--9"]
    (out / f"{name}.tex").write_text(latex_table(header, rows, "lccccc"), encoding="utf-8")


SHIFTS = [
    ("heldout_long", "default"),
    ("fault_0.6", "$p_f{=}0.6$"),
    ("fault_0.9", "$p_f{=}0.9$"),
    ("verify_cost_2", "$c_V{=}2$"),
    ("verify_cost_3", "$c_V{=}3$"),
    ("budget_extra_4", "slack 4"),
    ("budget_extra_8", "slack 8"),
    ("cue_clean", "clean cue"),
    ("cue_noisy", "noisy cue"),
]


def table_shifts(summary: pd.DataFrame, out: Path) -> None:
    methods = ["always_think", "cue_triggered", "budget_aware_1", "outcome_inv", "naive_inv", "ma_abs", "ma_inv"]
    rows = []
    for m in methods:
        rows.append([display(summary, m)] + [f"{row(summary, s, m)['success']:.2f}" for s, _ in SHIFTS])
    header = ["Method"] + [lab for _, lab in SHIFTS]
    (out / "shift_results.tex").write_text(latex_table(header, rows, "l" + "c" * len(SHIFTS)), encoding="utf-8")


def table_ablation(summary: pd.DataFrame, out: Path, name: str = "ablation_results") -> None:
    rows = []
    for m in [m for m in summary.method.drop_duplicates() if not m.endswith(FINAL)]:
        rows.append(
            [
                display(summary, m),
                fmt(row(summary, "train_like", m), "success"),
                fmt(row(summary, "heldout_long", m), "success"),
                fmt(row(summary, "heldout_long", m), "fault_detected"),
                fmt(row(summary, "heldout_long", m), "verify_rate_cue1"),
                fmt(row(summary, "heldout_long", m), "verify_rate_cue0"),
            ]
        )
    header = ["Variant", "Success 3--6", "Success 7--9", "Detect 7--9", "$P(V\\mid c{=}1)$", "$P(V\\mid c{=}0)$"]
    (out / f"{name}.tex").write_text(latex_table(header, rows, "lccccc"), encoding="utf-8")


COLLAPSE_THRESHOLD = 0.35  # greedy success at or below "never verify" (~0.2-0.3)


def curve_stats(curves: pd.DataFrame, per_seed: pd.DataFrame, burn_in: int = 5000) -> dict[str, float]:
    """Training-stability statistics quoted in the text, keyed for ``write_macros``."""
    out: dict[str, float] = {}
    late = curves[curves.episode > burn_in]
    for method, sub in late.groupby("method"):
        collapsed = sub.train_like_success <= COLLAPSE_THRESHOLD
        out[f"main/{method}/collapse_rate"] = float(collapsed.mean())
        jumps = sub.sort_values("episode").groupby("seed").train_like_success.apply(lambda x: x.diff().abs().mean())
        out[f"main/{method}/mean_abs_change"] = float(jumps.mean())
        selected = per_seed[(per_seed.method == method) & (per_seed.setting == "train_like")].success
        out[f"main/{method}/discovery_rate"] = float((selected > 0.5).mean())
        out[f"main/{method}/discovered_seeds"] = float((selected > 0.5).sum())
    return out


def key_numbers(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Paired (by seed) differences quoted in the text."""
    comparisons = [
        ("ma_inv", "ma_abs", "heldout_long"),
        ("ma_inv", "ma_abs", "heldout_xlong"),
        ("ma_inv", "outcome_inv", "train_like"),
        ("ma_inv", "naive_inv", "train_like"),
        ("ma_abs", "outcome_abs", "train_like"),
        ("ma_inv", "cue_triggered", "train_like"),
        ("ma_inv", "cue_triggered", "heldout_long"),
    ]
    rows = []
    for a, b, setting in comparisons:
        d = paired_difference(per_seed, "success", a, b, setting)
        rows.append({"a": a, "b": b, "setting": setting, "metric": "success", **d})
    return pd.DataFrame(rows)


MACRO_HEADER = r"""% Generated by `python -m ultrarl.report`; do not edit by hand.
% \val{exp/method/setting/metric} -> mean over seeds; \ci{...} -> mean [95% bootstrap CI].
\makeatletter
\newcommand{\val}[1]{\ifcsname ur@#1\endcsname\csname ur@#1\endcsname\else\errmessage{UltraRL: no result named #1}\fi}
\newcommand{\ci}[1]{\val{#1}\,{\scriptsize[\val{#1/lo}, \val{#1/hi}]}}
\makeatother
"""


def _macro(key: str, value: float, digits: int = 2) -> str:
    text = "--" if np.isnan(value) else f"{value:.{digits}f}"
    return f"\\expandafter\\def\\csname ur@{key}\\endcsname{{{text}}}"


def _pct_macro(key: str, value: float) -> str:
    return f"\\expandafter\\def\\csname ur@{key}/pct\\endcsname{{{100 * value:.0f}\\%}}"


def write_macros(path: Path, summaries: dict[str, pd.DataFrame], extras: dict[str, float]) -> None:
    lines = [MACRO_HEADER]
    for exp, summary in summaries.items():
        metrics = [c for c in summary.columns if f"{c}_lo" in summary.columns]
        for _, r in summary.iterrows():
            for metric in metrics:
                key = f"{exp}/{r['method']}/{r['setting']}/{metric}"
                lines += [
                    _macro(key, r[metric]),
                    _macro(f"{key}/lo", r[f"{metric}_lo"]),
                    _macro(f"{key}/hi", r[f"{metric}_hi"]),
                    _pct_macro(key, r[metric]),
                ]
    for key, value in extras.items():
        lines.append(_macro(key, value))
        if key.endswith("_rate"):
            lines.append(_pct_macro(key, value))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--paper", default="paper")
    args = parser.parse_args(argv)
    results, paper = Path(args.results), Path(args.paper)
    figures, tables = paper / "figures", paper / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    main_dir = results / "main"
    summary = pd.read_csv(main_dir / "eval_summary.csv")
    per_seed = pd.read_csv(main_dir / "eval_per_seed.csv")
    curves = load_curves(main_dir)

    fig_length_generalization(summary, figures)
    fig_learning_curves(curves, summary, figures, LEARNED)
    fig_selectivity(summary, figures)
    fig_churn(curves, figures)
    table_main(summary, tables)
    table_main(summary, tables, suffix=FINAL, name="main_results_final")
    table_shifts(summary, tables)
    comparisons = key_numbers(per_seed)
    comparisons.to_csv(main_dir / "key_comparisons.csv", index=False)

    summaries = {"main": summary}
    extras = curve_stats(curves, per_seed)
    for _, c in comparisons.iterrows():
        key = f"diff/{c['a']}-{c['b']}/{c['setting']}/{c['metric']}"
        extras.update({key: c["mean"], f"{key}/lo": c["lo"], f"{key}/hi": c["hi"]})
    for name in ("ablations", "sensitivity"):
        path = results / name / "eval_summary.csv"
        if path.exists():
            summaries[name] = pd.read_csv(path)
            table_ablation(summaries[name], tables, name=f"{name}_results")
    write_macros(paper / "numbers.tex", summaries, extras)
    print(f"[report] wrote figures to {figures}, tables to {tables} and {paper / 'numbers.tex'}")


if __name__ == "__main__":
    main()
