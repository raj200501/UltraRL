"""Figures and LaTeX tables for the paper, built only from files under ``results/``.

Usage::

    python -m ultrarl.report --results results --paper paper
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ultrarl.config import EnvConfig, load_experiment  # noqa: E402
from ultrarl.optimal import OptimalPlanner  # noqa: E402
from ultrarl.stats import FINAL_SUFFIX, holm, paired_difference, t_interval  # noqa: E402

# Categorical slots in fixed order (one hue per reward scheme); observation
# encoder is a secondary encoding (line style / hatch), heuristics are neutral.
SCHEME_COLOR = {"outcome": "#2a78d6", "naive": "#eb6834", "ma": "#1baf7a"}
SCHEME_LABEL = {"outcome": "Outcome-only", "naive": "Naive process", "ma": "Monitorability-aware"}
OBS_STYLE = {"inv": "-", "abs": "--"}
OBS_LABEL = {"inv": "invariant obs.", "abs": "absolute obs."}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HEURISTIC_STYLE = {
    "optimal": dict(color=INK, linestyle="-.", marker="*"),
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
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})  # reproducible bytes
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
    references = (
        ("optimal", "Exact planner (knows length, no back-checking)"),
        ("cue_triggered", "Cue-triggered heuristic"),
        ("budget_aware_1", "Budget-aware heuristic"),
    )
    for name, text in references:
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
        lo, hi = t_interval(vals)
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
def fig_optimal_decisions(cfg: EnvConfig, out: Path, depths: range = range(3, 15)) -> None:
    """Where the exact planner checks a step it has just executed, by remaining steps
    and budget slack: the fraction of chain lengths for which checking is optimal."""
    max_remaining, max_slack = depths.stop - 2, cfg.budget_extra
    planners = {d: OptimalPlanner(cfg, d) for d in depths}
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.1), sharey=True)
    for ax, cue in zip(axes, (0, 1), strict=True):
        grid = np.full((max_slack + 1, max_remaining + 1), np.nan)
        for slack in range(max_slack + 1):
            for remaining in range(max_remaining + 1):
                decisions = [p.should_check(d - remaining, slack, cue) for d, p in planners.items() if d > remaining]
                grid[slack, remaining] = np.mean(decisions)
        image = ax.imshow(grid, origin="lower", cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"cue {'on' if cue else 'off'}", fontsize=8)
        ax.set_xlabel("Steps remaining")
        ax.grid(False)
    axes[0].set_ylabel("Budget slack")
    fig.colorbar(image, ax=axes, fraction=0.04, pad=0.02, label="P(check is optimal)")
    save(fig, out, "optimal_decisions")


def fmt(r: pd.Series, metric: str, digits: int = 2) -> str:
    v, lo, hi = r[metric], r[f"{metric}_lo"], r[f"{metric}_hi"]
    if np.isnan(v):
        return "--"
    return f"{_number(v, digits)}\\,{{\\scriptsize[{_number(lo, digits, True)}, {_number(hi, digits, True)}]}}"


def latex_table(header: list[str], rows: list[list[str]], align: str) -> str:
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [r[0] if len(r) == 1 else " & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def display(summary: pd.DataFrame, method: str) -> str:
    return summary.loc[summary.method == method, "display_name"].iloc[0]


FINAL = FINAL_SUFFIX


def table_main(summary: pd.DataFrame, out: Path, suffix: str = "", name: str = "main_results") -> None:
    heuristics = ["random", "always_think", "verify_every_step", "cue_triggered", "budget_aware_1", "optimal"]
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


def pm(r: pd.Series, metric: str) -> str:
    """Mean and half-width of its (symmetric) t-interval."""
    return f"{r[metric]:.2f}\\,{{\\scriptsize$\\pm${r[metric] - r[metric + '_lo']:.2f}}}"


def table_shifts(summary: pd.DataFrame, out: Path) -> None:
    methods = [
        "always_think", "cue_triggered", "budget_aware_1", "optimal", "outcome_inv", "naive_inv", "ma_abs", "ma_inv",
    ]  # fmt: skip
    rows = []
    for m in methods:
        rows.append([display(summary, m)] + [pm(row(summary, s, m), "success") for s, _ in SHIFTS])
    header = ["Method"] + [lab for _, lab in SHIFTS]
    (out / "shift_results.tex").write_text(latex_table(header, rows, "l" + "c" * len(SHIFTS)), encoding="utf-8")


def table_ablation(summary: pd.DataFrame, out: Path, name: str = "ablation_results", suffix: str = "") -> None:
    rows = []
    for base in [m for m in summary.method.drop_duplicates() if not m.endswith(FINAL)]:
        m = base + suffix
        rows.append(
            [
                display(summary, base),
                fmt(row(summary, "train_like", m), "success"),
                fmt(row(summary, "heldout_long", m), "success"),
                fmt(row(summary, "heldout_xlong", m), "success"),
                f"{row(summary, 'heldout_long', m)['verify_rate_cue1']:.2f}",
                f"{row(summary, 'heldout_long', m)['verify_rate_cue0']:.2f}",
            ]
        )
    header = ["Variant", "Success 3--6", "Success 7--9", "Success 10--14", "$P(V\\mid c{=}1)$", "$P(V\\mid c{=}0)$"]
    (out / f"{name}.tex").write_text(latex_table(header, rows, "lccccc"), encoding="utf-8")


COLLAPSE_THRESHOLD = 0.35  # a margin above never-check success (Always THINK: ~0.20 on training lengths)
LOW_DETECTION = 0.2  # collapsed checkpoints below this detection rate check, but almost never catch a fault
NO_REPAIR_DETECTION = 0.5  # collapsed checkpoints at or above this detection rate still check but fail to repair
BURN_IN = 5000


def curve_stats(
    curves: pd.DataFrame, per_seed: pd.DataFrame, burn_in: int = BURN_IN, prefix: str = "main"
) -> dict[str, float | str]:
    """Training-stability statistics quoted in the text, keyed for ``write_macros``.

    A late checkpoint (after ``burn_in`` episodes) is *collapsed* if its greedy
    success on training lengths is at most ``COLLAPSE_THRESHOLD``. Collapsed
    checkpoints are split by their fault-detection rate into *low-detection*
    (detection < 0.2: the policy checks too little, or in the wrong places, to catch
    faults; the ones we inspected check only the final step) and *detect-without-repair*
    (detection >= 0.5: it still catches faults but does not complete the repair).
    """
    out: dict[str, float | str] = {}
    late = curves[curves.episode > burn_in]
    for method, sub in late.groupby("method"):
        sub = sub.assign(
            collapsed=sub.train_like_success <= COLLAPSE_THRESHOLD,
            detect=sub.train_like_fault_detected,
        )
        sub = sub.assign(
            lowdetect=sub.collapsed & (sub.detect < LOW_DETECTION),
            norepair=sub.collapsed & (sub.detect >= NO_REPAIR_DETECTION),
        )
        for name in ("collapsed", "lowdetect", "norepair"):
            key = {
                "collapsed": "collapse_rate",
                "lowdetect": "collapse_lowdetect_rate",
                "norepair": "collapse_norepair_rate",
            }
            per_seed_rate = sub.groupby("seed")[name].mean().to_numpy(dtype=float)
            low, high = t_interval(per_seed_rate)
            out[f"{prefix}/{method}/{key[name]}"] = float(sub[name].mean())
            out[f"{prefix}/{method}/{key[name]}/lo"] = max(0.0, low)
            out[f"{prefix}/{method}/{key[name]}/hi"] = min(1.0, high)
        jumps = sub.sort_values("episode").groupby("seed").train_like_success.apply(lambda x: x.diff().abs().mean())
        out[f"{prefix}/{method}/mean_abs_change"] = float(jumps.mean())
        everything = curves[curves.method == method]
        peaks = everything.groupby("seed").train_like_success.max()
        out[f"{prefix}/{method}/seeds_peak_above_half"] = str(int((peaks > 0.5).sum()))
        # Acquisition and retention: when a seed first exceeds 0.5, and how often it collapses afterwards.
        first = everything[everything.train_like_success > 0.5].groupby("seed").episode.min()
        out[f"{prefix}/{method}/first_above_half_median"] = f"{first.median():,.0f}" if len(first) else "--"
        after = [
            float((s.train_like_success[s.episode > first[seed]] <= COLLAPSE_THRESHOLD).mean())
            for seed, s in everything.groupby("seed")
            if seed in first.index and (s.episode > first[seed]).any()
        ]
        if after:
            low, high = t_interval(np.array(after))
            out[f"{prefix}/{method}/collapse_after_first_rate"] = float(np.mean(after))
            out[f"{prefix}/{method}/collapse_after_first_rate/lo"] = max(0.0, low)
            out[f"{prefix}/{method}/collapse_after_first_rate/hi"] = min(1.0, high)
        selected = per_seed[(per_seed.method == method) & (per_seed.setting == "train_like")].success
        out[f"{prefix}/{method}/seeds_selected_above_half"] = str(int((selected > 0.5).sum()))
    return out


def unrepaired_loss_macros(per_seed: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Fraction of episodes lost to a fault that was caught but not repaired, per method and setting:
    P(fault) * P(caught | fault) * P(failure | caught), computed per seed and averaged."""
    loss = per_seed.fault_rate * per_seed.fault_detected * (1.0 - per_seed.success_given_detected)
    means = per_seed.assign(loss=loss).groupby(["method", "setting"]).loss.mean()
    return {f"{prefix}/{m}/{s}/unrepaired_rate": float(v) for (m, s), v in means.items()}


def selection_macros(train_runs: pd.DataFrame, prefix: str) -> dict[str, str]:
    """When the selected checkpoints come from: median episode, and how many are within the burn-in."""
    out: dict[str, str] = {}
    for method, sub in train_runs.groupby("method"):
        out[f"{prefix}/{method}/selected_episode_median"] = f"{sub.selected_episode.median():,.0f}"
        out[f"{prefix}/{method}/selected_early"] = str(int((sub.selected_episode <= BURN_IN).sum()))
        if "selection_success" in sub:
            out[f"{prefix}/{method}/selection_success_median"] = f"{sub.selection_success.median():.2f}"
    return out


COMPARISONS = [
    # encoder effect
    ("ma_inv", "ma_abs", "train_like"),
    ("ma_inv", "ma_abs", "heldout_long"),
    ("ma_inv", "ma_abs", "heldout_xlong"),
    ("outcome_inv", "outcome_abs", "heldout_long"),
    ("naive_inv", "naive_abs", "heldout_long"),
    # reward effect under each encoder
    ("ma_abs", "outcome_abs", "train_like"),
    ("ma_inv", "outcome_inv", "train_like"),
    ("ma_inv", "outcome_inv", "heldout_long"),
    ("ma_inv", "outcome_inv", "heldout_xlong"),
    ("ma_inv", "naive_inv", "train_like"),
    ("ma_inv", "naive_inv", "heldout_long"),
    ("ma_inv", "naive_inv", "heldout_xlong"),
    ("naive_inv", "outcome_inv", "train_like"),
    ("naive_inv", "outcome_inv", "heldout_long"),
    ("naive_inv", "outcome_inv", "heldout_xlong"),
    # learned vs hand-coded
    ("ma_inv", "cue_triggered", "train_like"),
    ("ma_inv", "cue_triggered", "heldout_long"),
    ("ma_inv", "cue_triggered", "heldout_xlong"),
    ("ma_inv", "budget_aware_1", "heldout_long"),
    # the same comparisons for last checkpoints
    ("ma_inv@final", "ma_abs@final", "heldout_long"),
    ("ma_inv@final", "ma_abs@final", "heldout_xlong"),
    ("ma_inv@final", "outcome_inv@final", "train_like"),
    ("ma_inv@final", "outcome_inv@final", "heldout_long"),
    ("ma_inv@final", "naive_inv@final", "heldout_long"),
    ("naive_inv@final", "outcome_inv@final", "heldout_long"),
    ("ma_inv@final", "cue_triggered", "train_like"),
    ("ma_inv@final", "cue_triggered", "heldout_long"),
]


# The comparisons behind the paper's headline claims, chosen after the results were
# known (not pre-registered); their p-values are Holm-adjusted as one family. All
# other comparisons are exploratory and reported unadjusted.
PRIMARY = [
    ("ma_inv", "ma_abs", "heldout_long"),
    ("ma_inv", "ma_abs", "heldout_xlong"),
    ("ma_abs", "outcome_abs", "train_like"),
    ("ma_inv", "outcome_inv", "train_like"),
    ("ma_inv", "outcome_inv", "heldout_long"),
    ("ma_inv", "outcome_inv", "heldout_xlong"),
]


def key_numbers(per_seed: pd.DataFrame, comparisons: list[tuple[str, str, str]] = COMPARISONS) -> pd.DataFrame:
    """Paired (by seed) differences in success quoted in the text."""
    rows = []
    for a, b, setting in comparisons:
        d = paired_difference(per_seed, "success", a, b, setting)
        rows.append(
            {"a": a, "b": b, "setting": setting, "metric": "success", "primary": (a, b, setting) in PRIMARY, **d}
        )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = np.nan
    primary = frame.primary
    if primary.any():
        frame.loc[primary, "p_holm"] = holm(frame.loc[primary, "p_value"].tolist())
    return frame


def comparison_macros(comparisons: pd.DataFrame, prefix: str = "diff") -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for _, c in comparisons.iterrows():
        key = f"{prefix}/{c['a']}-{c['b']}/{c['setting']}/{c['metric']}"
        out.update({key: c["mean"], f"{key}/lo": c["lo"], f"{key}/hi": c["hi"]})
        out[f"{key}/npos"] = str(int(c["n_positive"]))
        out[f"{key}/n"] = str(int(c["n"]))
        p = c["p_value"]
        out[f"{key}/p"] = "<0.001" if p < 0.001 else f"{p:.3f}"
        if c.get("primary", False):
            out[f"{key}/pholm"] = f"{c['p_holm']:.3f}"
    return out


def table_behaviour(summary: pd.DataFrame, out: Path, setting: str = "heldout_long") -> None:
    """Behavioural diagnostics of the learned agents and two references."""
    methods = ["cue_triggered", "budget_aware_1", "optimal", *LEARNED]
    rows = []
    for m in methods:
        r = row(summary, setting, m)
        rows.append(
            [
                display(summary, m),
                f"{r['verify_count']:.2f}",
                f"{r['nonrepair_revise_count']:.2f}",
                f"{r['fault_detected']:.2f}",
                f"{r['success_given_detected']:.2f}",
                f"{r['voluntary_incomplete_commit']:.2f}",
                f"{r['commit_at_start']:.2f}",
            ]
        )
    header = [
        "Method",
        "Checks/ep.",
        "Non-repair REVISE/ep.",
        "Detect",
        "Success $\\mid$ detect",
        "Early commit",
        "Commit at $t{=}0$",
    ]
    (out / f"behaviour_{setting}.tex").write_text(latex_table(header, rows, "lcccccc"), encoding="utf-8")


FOLLOWUP_REFERENCE = "ma_inv"


def table_followups(
    main_summary: pd.DataFrame, summary: pd.DataFrame, stats: dict[str, float | str], out: Path
) -> None:
    """Follow-up variants next to the reference agent: peak (selected) and last-checkpoint
    success, how often late checkpoints collapse, and wasted REVISEs on long chains."""
    combined = pd.concat([main_summary, summary], ignore_index=True)
    rows = []
    variants = [FOLLOWUP_REFERENCE] + [
        m for m in summary.method.drop_duplicates() if not m.endswith(FINAL) and m.startswith("ma_")
    ]
    for m in variants:
        prefix = "main" if m == FOLLOWUP_REFERENCE else "followups"
        rows.append(
            [
                display(combined, m),
                fmt(row(combined, "train_like", m), "success"),
                fmt(row(combined, "heldout_long", m), "success"),
                fmt(row(combined, "heldout_xlong", m), "success"),
                fmt(row(combined, "heldout_long", m + FINAL), "success"),
                f"{100 * stats[f'{prefix}/{m}/collapse_rate']:.0f}\\%",
                f"{row(combined, 'heldout_xlong', m)['nonrepair_revise_count']:.2f}",
            ]
        )
    header = [
        "Variant",
        "Success 3--6",
        "Success 7--9",
        "Success 10--14",
        "Last ckpt.\\ 7--9",
        "Collapsed",
        "Wasted REVISE",
    ]
    (out / "followup_results.tex").write_text(latex_table(header, rows, "lcccccc"), encoding="utf-8")


def table_pending(main_summary: pd.DataFrame, summary: pd.DataFrame, stats: dict[str, float | str], out: Path) -> None:
    """Each reward with the invariant observation, with the repair-pending bit added, and with
    the last few actions added instead (if those runs exist)."""
    combined = pd.concat([main_summary, summary], ignore_index=True)
    rows = []
    variants = [
        ("inv", "main", ""),
        ("pending", "followups", " + pending bit"),
        ("history", "followups", " + last 8 actions"),
    ]
    for scheme in SCHEMES:
        for obs, prefix, suffix in variants:
            m = f"{scheme}_{obs}"
            if m not in set(combined.method):
                continue
            pct = [f"{100 * stats[f'{prefix}/{m}/{k}']:.0f}\\%" for k in ("collapse_rate", "collapse_norepair_rate",
                                                                          "collapse_lowdetect_rate")]  # fmt: skip
            rows.append(
                [
                    SCHEME_LABEL[scheme] + suffix,
                    *pct,
                    fmt(row(combined, "train_like", m), "success"),
                    fmt(row(combined, "heldout_long", m), "success"),
                    fmt(row(combined, "heldout_xlong", m), "success"),
                    fmt(row(combined, "heldout_long", m + FINAL), "success"),
                    fmt(row(combined, "heldout_xlong", m + FINAL), "success"),
                ]
            )
    header = ["Reward and observation", "Collapsed", "no repair", "low detection", "Success 3--6", "Success 7--9",
              "Success 10--14", "Last ckpt.\\ 7--9", "Last ckpt.\\ 10--14"]  # fmt: skip
    (out / "pending_results.tex").write_text(latex_table(header, rows, "lcccccccc"), encoding="utf-8")


def table_encoders(
    main_summary: pd.DataFrame, encoders: pd.DataFrame, stats: dict[str, float | str], out: Path
) -> None:
    """The position-feature control next to the two main encoders (monitorability-aware reward)."""
    combined = pd.concat([main_summary, encoders], ignore_index=True)
    rows = []
    for m, kind in (("ma_abs", "absolute"), ("ma_abs_nopos", "absolute_nopos"), ("ma_inv", "invariant")):
        rows.append(
            [
                display(combined, m),
                fmt(row(combined, "train_like", m), "success"),
                fmt(row(combined, "heldout_long", m), "success"),
                fmt(row(combined, "heldout_xlong", m), "success"),
                f"{100 * stats[f'coverage/{kind}/heldout_xlong/unseen_rate']:.0f}\\%",
                f"{row(combined, 'heldout_xlong', m)['commit_at_start']:.2f}",
            ]
        )
    header = ["Encoder", "Success 3--6", "Success 7--9", "Success 10--14", "Unseen obs.\\ 10--14", "Commit at $t{=}0$"]
    (out / "encoder_results.tex").write_text(latex_table(header, rows, "lccccc"), encoding="utf-8")


def table_pilot(pilot: Path, out: Path) -> None:
    """The archived two-seed hyperparameter pilot (results/pilot), selected checkpoints."""
    rows = []
    for stage in sorted(pilot.glob("stage*")):
        summary = pd.read_csv(stage / "eval_summary.csv")
        for method in [m for m in summary.method.drop_duplicates() if not m.endswith(FINAL)]:
            scheme, obs, *rest = method.split("_")
            lr = "$3{\\cdot}10^{-4}$" if any(t in ("lr0.0003", "lr3e4") for t in rest) else "$10^{-3}$"
            sync = next((f"hard, {t[2:]}" for t in rest if t.startswith("tu")), "Polyak $\\tau{=}0.005$")
            rows.append([SCHEME_LABEL[scheme], obs, lr, sync] + [
                f"{row(summary, s, method)['success']:.2f}" for s in ("train_like", "heldout_long")
            ])  # fmt: skip
    header = ["Reward", "Obs.", "Learning rate", "Target update", "Success 3--6", "Success 7--9"]
    (out / "pilot_results.tex").write_text(latex_table(header, rows, "llllcc"), encoding="utf-8")


LEARNER_SETTINGS = [("", "default (main experiment)"), ("lr3e-4", "lr $3{\\cdot}10^{-4}$"),
                    ("sync1000", "hard sync 1000"), ("polyak", "Polyak $\\tau{=}0.005$"),
                    ("40k", "40k episodes")]  # fmt: skip
SCHEMES = ("outcome", "naive", "ma")


def table_sensitivity_rewards(
    main_summary: pd.DataFrame, summary: pd.DataFrame, stats: dict[str, float | str], out: Path
) -> None:
    """The reward comparison (invariant encoder) under each learner setting: how often late
    checkpoints collapse, and selected-checkpoint success at lengths 7--9."""
    rows = []
    for tag, text in LEARNER_SETTINGS:
        source, prefix = (main_summary, "main") if not tag else (summary, "sensitivity")
        names = [f"{s}_inv" + (f"_{tag}" if tag else "") for s in SCHEMES]
        if not any(m in set(source.method) for m in names):
            continue
        collapse = [
            f"{100 * stats[f'{prefix}/{m}/collapse_rate']:.0f}\\%" if f"{prefix}/{m}/collapse_rate" in stats else "--"
            for m in names
        ]
        success = [fmt(row(source, "heldout_long", m), "success") if m in set(source.method) else "--" for m in names]
        rows.append([text, *collapse, *success])
    header = ["Learner", "Collapsed: outcome", "naive", "MA", "Success 7--9: outcome", "naive", "MA"]
    (out / "sensitivity_rewards.tex").write_text(latex_table(header, rows, "lcccccc"), encoding="utf-8")


PROBE_METRICS = (
    "think_rate", "q_think", "q_revise", "return_think", "return_revise", "q_on_track", "return_on_track",
    "q_start", "return_start", "aliased_rate",
)  # fmt: skip
PROBE_DERIVED = ("q_gap", "true_gap", "start_bias", "think_error", "revise_error", "think_bias", "revise_bias")


def with_probe_derived(probe: pd.DataFrame) -> pd.DataFrame:
    """Learned and true gaps between the two actions; each value's raw error (learned minus
    true); and its excess error, relative to the checkpoint's error at the first decision
    point of an episode (its general offset, measured where no hidden state differs)."""
    start_bias = probe.q_start - probe.return_start
    think_error, revise_error = probe.q_think - probe.return_think, probe.q_revise - probe.return_revise
    return probe.assign(
        q_gap=probe.q_revise - probe.q_think,
        true_gap=probe.return_revise - probe.return_think,
        start_bias=start_bias,
        think_error=think_error,
        revise_error=revise_error,
        think_bias=think_error - start_bias,
        revise_bias=revise_error - start_bias,
    )


def probe_macros(probe: pd.DataFrame) -> dict[str, float | str]:
    """Mean over seeds (and t-interval) of the post-UNSAFE value probe (scripts/q_probe.py)."""
    out: dict[str, float | str] = {}
    for (method, which), sub in with_probe_derived(probe).groupby(["method", "checkpoint"]):
        name = method + ("" if which == "best" else FINAL)
        out[f"probe/{name}/n_seeds"] = str(int(sub.think_rate.notna().sum()))
        for metric in PROBE_METRICS + PROBE_DERIVED:
            low, high = t_interval(sub[metric].to_numpy(dtype=float))
            out.update({f"probe/{name}/{metric}": float(sub[metric].mean()), f"probe/{name}/{metric}/lo": low,
                        f"probe/{name}/{metric}/hi": high})  # fmt: skip
    return out


PROBE_ROWS = [
    ("main", "outcome_inv", "Outcome-only"),
    ("main", "naive_inv", "Naive process"),
    ("main", "ma_inv", "Monitorability-aware (MA)"),
    ("followups", "ma_memory", "MA + memory bit"),
    ("followups", "ma_repair", "MA, bonus paid on repair"),
    ("followups", "ma_memory_repair", "MA + memory bit, bonus on repair"),
    ("followups", "outcome_pending", "Outcome-only + pending bit"),
    ("followups", "naive_pending", "Naive process + pending bit"),
    ("followups", "ma_pending", "MA + pending bit"),
    ("followups", "outcome_history", "Outcome-only + last 8 actions"),
    ("followups", "naive_history", "Naive process + last 8 actions"),
    ("followups", "ma_history", "MA + last 8 actions"),
]


def _pair(a: float, b: float, signed: bool = False) -> str:
    def text(x: float) -> str:
        return ("+" if signed and x >= 0.005 else "") + _number(x, 2)

    return f"{text(a)} / {text(b)}"


def probe_range_macros(probe: pd.DataFrame, methods: tuple[str, ...] = ("outcome_inv", "naive_inv", "ma_inv")):
    """Smallest and largest per-agent mean of the excess errors, over the main invariant agents."""
    means = with_probe_derived(probe).groupby(["method", "checkpoint"]).mean(numeric_only=True)
    out: dict[str, float] = {}
    for which in ("best", "final"):
        for metric in ("think_bias", "revise_bias"):
            values = [means.loc[(m, which), metric] for m in methods if (m, which) in means.index]
            out[f"probe/inv_{which}/{metric}/min"], out[f"probe/inv_{which}/{metric}/max"] = min(values), max(values)
    return out


def table_probe(
    probe: pd.DataFrame, summaries: dict[str, pd.DataFrame], stats: dict[str, float | str], out: Path
) -> None:
    """Post-UNSAFE value probe of the selected checkpoints, next to how often late checkpoints
    detect faults without repairing them."""
    means = with_probe_derived(probe).groupby(["method", "checkpoint"]).mean(numeric_only=True)
    rows = []
    for experiment, m, text in PROBE_ROWS:
        if (m, "best") not in means.index:
            continue
        best, final = means.loc[(m, "best")], means.loc[(m, "final")]
        rows.append([
            text,
            f"{100 * stats[f'{experiment}/{m}/collapse_norepair_rate']:.0f}\\%",
            f"{100 * best.think_rate:.0f}\\% / {100 * final.think_rate:.0f}\\%",
            _pair(best.revise_error, best.think_error, signed=True),
            _number(best.start_bias, 2),
            _pair(best.revise_bias, best.think_bias, signed=True),
            _pair(final.revise_bias, final.think_bias, signed=True),
        ])  # fmt: skip
    header = ["Agent", "Detect, no repair", "THINK after \\textsc{unsafe} (sel.\\ / last)", "Error R / T",
              "Offset", "Excess error R / T", "Excess, last ckpt.\\ R / T"]  # fmt: skip
    (out / "probe_results.tex").write_text(latex_table(header, rows, "lcccccc"), encoding="utf-8")


def success_range(summary: pd.DataFrame, key: str, pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Smallest and largest success (as rounded percentages) over (method, setting) pairs."""
    values = [row(summary, s, m)["success"] for m, s in pairs]
    return {f"{key}/min/pct": f"{100 * min(values):.0f}", f"{key}/max/pct": f"{100 * max(values):.0f}",
            f"{key}/min": f"{min(values):.2f}", f"{key}/max": f"{max(values):.2f}"}  # fmt: skip


# Ablation variants that keep the per-action cost and some payment for informative checks.
CHECK_PAYING = ["no_cue_terms", "no_gap_term", "no_waste_penalty", "no_check_bonuses", "no_monitor_terms",
                "cue_bonus_only"]  # fmt: skip
LENGTHS = ("train_like", "heldout_long", "heldout_xlong")

MACRO_HEADER = r"""% Generated by `python -m ultrarl.report`; do not edit by hand.
% \val{exp/method/setting/metric} -> mean over seeds; \ci{...} -> mean [95% Student-t interval over seeds].
% Only the results that the paper's sources reference are written.
\makeatletter
\newcommand{\val}[1]{\ifcsname ur@#1\endcsname\csname ur@#1\endcsname\else\errmessage{UltraRL: no result named #1}\fi}
\newcommand{\ci}[1]{\val{#1}\,{\scriptsize[\val{#1/lo}, \val{#1/hi}]}}
\makeatother
"""


def _number(value: float, digits: int, keep_sign: bool = False) -> str:
    """Rounded value with a real minus sign. A mean that rounds to -0.00 prints as 0.00; an
    interval bound keeps its sign (``keep_sign``) so a bound just below zero stays visible."""
    if np.isnan(value):
        return "--"
    text = f"{value:.{digits}f}"
    if text.startswith("-"):
        text = text[1:] if float(text) == 0.0 and not keep_sign else "\\ensuremath{-}" + text[1:]
    return text


def _macro(key: str, value: float, digits: int = 2) -> str:
    text = _number(value, digits, keep_sign=key.endswith(("/lo", "/hi")))
    return f"\\expandafter\\def\\csname ur@{key}\\endcsname{{{text}}}"


def _pct_macro(key: str, value: float) -> str:
    return f"\\expandafter\\def\\csname ur@{key}/pct\\endcsname{{{_number(100 * value, 0)}\\%}}"


def used_macros(paper: Path) -> set[str] | None:
    """Result keys referenced by \\val or \\ci in the paper's sources (None if there are none)."""
    sources = [paper / "main.tex", *sorted((paper / "sections").glob("*.tex"))]
    keys: set[str] = set()
    for source in (f for f in sources if f.exists()):
        for command, key in re.findall(r"\\(val|ci)\{([^}]+)\}", source.read_text(encoding="utf-8")):
            keys.add(key)
            if command == "ci":
                keys.update({f"{key}/lo", f"{key}/hi"})
    return keys or None


def write_macros(
    path: Path, summaries: dict[str, pd.DataFrame], extras: dict[str, float | str], used: set[str] | None = None
) -> None:
    """Write every result as a macro, or only the ``used`` ones."""
    lines = [MACRO_HEADER]
    for exp, summary in summaries.items():
        metrics = [c for c in summary.columns if f"{c}_lo" in summary.columns and not c.endswith("_n")]
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
        if isinstance(value, str):
            lines.append(f"\\expandafter\\def\\csname ur@{key}\\endcsname{{{value}}}")
            continue
        lines.append(_macro(key, value))
        if key.endswith(("_rate", "_rate/lo", "_rate/hi")):
            lines.append(_pct_macro(key, value))
    if used is not None:
        lines = [lines[0]] + [x for x in lines[1:] if re.search(r"ur@(.+?)\\endcsname", x).group(1) in used]
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
    fig_optimal_decisions(load_experiment(main_dir / "config.yaml").env, figures)
    table_main(summary, tables)
    table_main(summary, tables, suffix=FINAL, name="main_results_final")
    table_shifts(summary, tables)
    table_behaviour(summary, tables, "heldout_long")
    table_behaviour(summary, tables, "heldout_xlong")
    comparisons = key_numbers(per_seed)
    comparisons.to_csv(main_dir / "key_comparisons.csv", index=False)

    summaries = {"main": summary}
    extras = curve_stats(curves, per_seed)
    extras.update(comparison_macros(comparisons))
    extras.update(unrepaired_loss_macros(per_seed, "main"))
    for method in ("cue_triggered", "optimal"):
        extras.update(success_range(summary, f"main/{method}/lengths", [(method, s) for s in LENGTHS]))
    for name in ("ablations", "sensitivity"):
        path = results / name / "eval_summary.csv"
        if path.exists():
            summaries[name] = pd.read_csv(path)
            shown = summaries[name]
            if name == "sensitivity":  # the encoder comparison; the reward comparison has its own table
                shown = shown[shown.method.str.startswith(("ma_abs", "ma_inv"))]
            table_ablation(shown, tables, name=f"{name}_results")
    if "ablations" in summaries:
        table_ablation(summaries["ablations"], tables, name="ablations_results_final", suffix=FINAL)
        ablation_seeds = pd.read_csv(results / "ablations" / "eval_per_seed.csv")
        extras.update(curve_stats(load_curves(results / "ablations"), ablation_seeds, prefix="ablations"))
        variants = [m for m in summaries["ablations"].method.unique() if m != "ma_inv" and not m.endswith(FINAL)]
        pairs = [("ma_inv", m, s) for m in variants for s in ("train_like", "heldout_long", "heldout_xlong")]
        pairs += [(f"ma_inv{FINAL}", f"{m}{FINAL}", s) for m in variants if m != "no_revise" for s in LENGTHS]
        extras.update(comparison_macros(key_numbers(ablation_seeds, pairs), prefix="abl"))
        paying = [(m, "train_like") for m in CHECK_PAYING]
        extras.update(success_range(summaries["ablations"], "ablations/check_paying/train_like", paying))
        paying = [(m + FINAL, "train_like") for m in CHECK_PAYING]
        extras.update(success_range(summaries["ablations"], f"ablations/check_paying{FINAL}/train_like", paying))
        # variants against the main experiment's outcome-only agents (same evaluation episodes per seed)
        own = ablation_seeds.method.isin(variants + [m + FINAL for m in variants])
        joint = pd.concat([per_seed, ablation_seeds[own]], ignore_index=True)
        pairs = [(m, "outcome_inv", s) for m in ("no_cue_terms", "no_monitor_terms") for s in LENGTHS]
        pairs += [(f"no_cue_terms{FINAL}", f"outcome_inv{FINAL}", "heldout_xlong")]
        extras.update(comparison_macros(key_numbers(joint, pairs)))
    if "sensitivity" in summaries:
        sensitivity_seeds = pd.read_csv(results / "sensitivity" / "eval_per_seed.csv")
        tags = sorted({m.removeprefix("ma_inv_") for m in summaries["sensitivity"].method if m.startswith("ma_inv_")})
        tags = [t for t in tags if not t.endswith(FINAL)]
        methods = set(summaries["sensitivity"].method)
        pairs = [
            (f"ma_inv_{t}", f"ma_abs_{t}", s)
            for t in tags
            if f"ma_abs_{t}" in methods
            for s in ("heldout_long", "heldout_xlong")
        ]
        for t in tags:
            for a, b in (("ma", "outcome"), ("ma", "naive"), ("naive", "outcome")):
                if {f"{a}_inv_{t}", f"{b}_inv_{t}"} <= methods:
                    pairs += [(f"{a}_inv_{t}", f"{b}_inv_{t}", s) for s in LENGTHS]
        extras.update(comparison_macros(key_numbers(sensitivity_seeds, pairs), prefix="sens"))
        extras.update(curve_stats(load_curves(results / "sensitivity"), sensitivity_seeds, prefix="sensitivity"))
        if {f"{s}_inv_{t}" for s in SCHEMES for t in ("lr3e-4", "sync1000", "polyak")} <= methods:
            table_sensitivity_rewards(summary, summaries["sensitivity"], extras, tables)
    if (results / "pilot").exists():
        table_pilot(results / "pilot", tables)
    probe_path = results / "q_probe.csv"
    if probe_path.exists():
        probe = pd.read_csv(probe_path)
        extras.update(probe_macros(probe))
        extras.update(probe_range_macros(probe))
    coverage_path = results / "observation_coverage.csv"
    if coverage_path.exists():
        for _, c in pd.read_csv(coverage_path).iterrows():
            extras[f"coverage/{c['observation']}/{c['setting']}/unseen_rate"] = c["unseen_fraction"]
    followup_dir = results / "followups"
    if (followup_dir / "eval_summary.csv").exists():
        summaries["followups"] = pd.read_csv(followup_dir / "eval_summary.csv")

        joint = pd.concat([per_seed, pd.read_csv(followup_dir / "eval_per_seed.csv")], ignore_index=True)
        followup_methods = [m for m in summaries["followups"].method.unique() if not m.endswith(FINAL)]
        lengths = ("train_like", "heldout_long", "heldout_xlong")
        pairs = [(m, FOLLOWUP_REFERENCE, s) for m in followup_methods for s in lengths]
        pairs += [(m + FINAL, FOLLOWUP_REFERENCE + FINAL, s) for m in followup_methods for s in lengths]
        extras.update(comparison_macros(key_numbers(joint, pairs)))
        followup_seeds = pd.read_csv(followup_dir / "eval_per_seed.csv")
        extras.update(curve_stats(load_curves(followup_dir), followup_seeds, prefix="followups"))
        pending = [m for m in followup_methods if m.endswith("_pending")]
        pairs = [(m, m.replace("_pending", "_inv"), s) for m in pending for s in lengths]
        pairs += [(m + FINAL, m.replace("_pending", "_inv") + FINAL, s) for m in pending for s in lengths]
        if {"ma_pending", "outcome_pending"} <= set(pending):
            pairs += [(f"ma_pending{x}", f"outcome_pending{x}", s) for x in ("", FINAL) for s in lengths]
        pairs += [(f"{m}{FINAL}", "cue_triggered", s) for m in pending for s in ("heldout_long", "heldout_xlong")]
        history = [m for m in followup_methods if m.endswith("_history")]
        pairs += [(m + x, m.replace("_history", y) + x, s) for m in history for y in ("_inv", "_pending")
                  for x in ("", FINAL) for s in lengths]  # fmt: skip
        extras.update(comparison_macros(key_numbers(joint, pairs)))
        table_followups(summary, summaries["followups"], extras, tables)
        if {f"{s}_pending" for s in SCHEMES} <= set(followup_methods):
            table_pending(summary, summaries["followups"], extras, tables)
    encoder_dir = results / "encoders"
    if (encoder_dir / "eval_summary.csv").exists():
        summaries["encoders"] = pd.read_csv(encoder_dir / "eval_summary.csv")
        joint = pd.concat([per_seed, pd.read_csv(encoder_dir / "eval_per_seed.csv")], ignore_index=True)
        lengths = ("train_like", "heldout_long", "heldout_xlong")
        pairs = [("ma_abs_nopos", ref, s) for ref in ("ma_abs", "ma_inv") for s in lengths]
        extras.update(comparison_macros(key_numbers(joint, pairs)))
        extras.update(unrepaired_loss_macros(pd.read_csv(encoder_dir / "eval_per_seed.csv"), "encoders"))
        if "coverage/absolute_nopos/heldout_xlong/unseen_rate" in extras:
            table_encoders(summary, summaries["encoders"], extras, tables)
    if probe_path.exists() and "followups" in summaries:
        table_probe(probe, summaries, extras, tables)
    for name in summaries:
        runs = results / name / "train_runs.csv"
        if runs.exists():
            extras.update(selection_macros(pd.read_csv(runs), name))
    write_macros(paper / "numbers.tex", summaries, extras, used_macros(paper))
    print(f"[report] wrote figures to {figures}, tables to {tables} and {paper / 'numbers.tex'}")


if __name__ == "__main__":
    main()
