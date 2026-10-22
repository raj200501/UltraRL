"""Run an experiment end to end: train every learned method, then evaluate all methods.

Usage::

    python -m ultrarl.run --config configs/main.yaml --out results/main
    python -m ultrarl.run --config configs/main.yaml --out /tmp/smoke --smoke

Training is resumable: runs whose checkpoint already exists are skipped unless
``--retrain`` is given. Outputs under ``--out``:

    checkpoints/<method>_seed<k>.pt   trained agents
    curves/<method>_seed<k>.csv       greedy learning curves
    train_runs.csv                    one row per training run
    eval_per_seed.csv                 one row per (setting, method, seed)
    eval_summary.csv                  mean and 95% bootstrap CI over seeds
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ultrarl.config import ExperimentConfig, load_experiment
from ultrarl.dqn import DQNAgent
from ultrarl.policies import build_heuristic
from ultrarl.rollout import evaluate
from ultrarl.stats import summarize
from ultrarl.train import checkpoint_path, greedy_act, train_run

SUMMARY_METRICS = [
    "success",
    "return",
    "cost_adjusted_accuracy",
    "num_actions",
    "verify_count",
    "revise_count",
    "incomplete_commit",
    "voluntary_incomplete_commit",
    "fault_detected",
    "repair_success",
    "clean_success",
    "verify_precision",
    "verify_rate_cue0",
    "verify_rate_cue1",
]


FINAL_SUFFIX = "@final"


def _train_job(job: tuple[str, bool, str, int, str]) -> dict[str, Any]:
    config_path, smoke, method_name, seed, out_dir = job
    exp = load_experiment(config_path, smoke=smoke)
    method = next(m for m in exp.methods if m.name == method_name)
    result = train_run(exp, method, seed, Path(out_dir))
    print(f"[train] done {method_name} seed={seed} ({result['wall_seconds']}s)", flush=True)
    return result


def _eval_job(job: tuple[str, bool, str, int, str]) -> list[dict[str, Any]]:
    config_path, smoke, method_name, seed, out_dir = job
    torch.set_num_threads(1)
    exp = load_experiment(config_path, smoke=smoke)
    method = next(m for m in exp.methods if m.name == method_name)
    # Learned methods are evaluated twice: the checkpoint selected on train-length
    # validation (reported as ``method``) and the final one (``method@final``).
    variants = []
    if method.kind == "dqn":
        for which, suffix in (("best", ""), ("final", FINAL_SUFFIX)):
            agent, _ = DQNAgent.load(checkpoint_path(Path(out_dir), method, seed, which))
            variants.append((method.name + suffix, method.display_name + suffix, lambda a=agent: greedy_act(a)))
    else:

        def make_act():
            policy = build_heuristic(method.name, seed)
            return lambda _obs, observable, mask: policy(observable, mask)

        variants.append((method.name, method.display_name, make_act))

    rows = []
    for name, display_name, make_act in variants:
        for setting in exp.eval_settings:
            metrics = evaluate(make_act(), exp.method_env(method, setting), setting, seed, exp.eval_episodes)
            rows.append(
                {"setting": setting.name, "method": name, "display_name": display_name, "seed": seed, **metrics}
            )
    return rows


def _pool_map(fn, jobs: list, workers: int) -> list:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(job) for job in jobs]
    with mp.get_context("spawn").Pool(workers) as pool:
        return list(pool.imap_unordered(fn, jobs))


def train_all(exp: ExperimentConfig, config_path: str, smoke: bool, out: Path, workers: int, retrain: bool) -> None:
    jobs = [
        (config_path, smoke, m.name, s, str(out))
        for m in exp.learned_methods
        for s in exp.seeds
        if retrain or not checkpoint_path(out, m, s, "final").exists()
    ]
    print(f"[train] {len(jobs)} runs to train ({workers} workers)", flush=True)
    _pool_map(_train_job, jobs, workers)
    write_manifest(exp, out)


def write_manifest(exp: ExperimentConfig, out: Path) -> pd.DataFrame:
    """One row per trained run, rebuilt from checkpoint metadata (robust to interrupted runs)."""
    rows = []
    for method in exp.learned_methods:
        for seed in exp.seeds:
            path = checkpoint_path(out, method, seed, "best")
            if path.exists():
                _, meta = DQNAgent.load(path)
                rows.append(
                    {
                        "method": method.name,
                        "seed": seed,
                        "episodes": meta["episodes"],
                        "env_steps": meta["env_steps"],
                        "parameter_count": meta["parameter_count"],
                        "selected_episode": meta["checkpoint_episode"],
                        "selection_success": meta["selection_success"],
                        "wall_seconds": meta.get("wall_seconds", float("nan")),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "train_runs.csv", index=False)
    return frame


def evaluate_all(exp: ExperimentConfig, config_path: str, smoke: bool, out: Path, workers: int) -> pd.DataFrame:
    jobs = [(config_path, smoke, m.name, s, str(out)) for m in exp.methods for s in exp.seeds]
    print(
        f"[eval] {len(jobs)} (method, seed) jobs x {len(exp.eval_settings)} settings x {exp.eval_episodes} episodes",
        flush=True,
    )
    rows = [row for chunk in _pool_map(_eval_job, jobs, workers) for row in chunk]
    method_order = {m.name: i for i, m in enumerate(exp.methods)}
    method_order.update({m.name + FINAL_SUFFIX: len(method_order) + i for i, m in enumerate(exp.learned_methods)})
    setting_order = {s.name: i for i, s in enumerate(exp.eval_settings)}
    per_seed = pd.DataFrame(rows)
    per_seed["_setting"] = per_seed["setting"].map(setting_order)
    per_seed["_method"] = per_seed["method"].map(method_order)
    per_seed = per_seed.sort_values(["_setting", "_method", "seed"]).drop(columns=["_setting", "_method"])
    per_seed.to_csv(out / "eval_per_seed.csv", index=False)
    summary = summarize(per_seed, ["setting", "method", "display_name"], SUMMARY_METRICS)
    summary.to_csv(out / "eval_summary.csv", index=False)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true", help="apply the config's tiny 'smoke' overrides")
    parser.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--retrain", action="store_true", help="retrain even if checkpoints exist")
    args = parser.parse_args(argv)

    exp = load_experiment(args.config, smoke=args.smoke)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out / "config.yaml")
    start = time.time()
    if args.stage in ("train", "all"):
        train_all(exp, args.config, args.smoke, out, args.workers, args.retrain)
    if args.stage in ("eval", "all"):
        evaluate_all(exp, args.config, args.smoke, out, args.workers)
    print(f"[run] {exp.name} finished in {time.time() - start:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
