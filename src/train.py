from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.agents import DQNAgent, ReplayBuffer
from src.envs import ProcessReasoningEnv
from src.utils import choose_device, ensure_dir, load_yaml, merge_dict, set_global_seed


def build_rl_specs(default_cfg: dict[str, Any], ablations_cfg: dict[str, Any], include_ablations: bool) -> list[dict[str, Any]]:
    specs = [dict(item) for item in default_cfg.get("rl_variants", [])]
    if include_ablations:
        specs.extend(dict(item) for item in ablations_cfg.get("train_variants", []))
    return specs


def train_one_run(
    spec: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    episodes: int,
    checkpoint_dir: Path,
    log_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    training_cfg = config["training"]
    env_config = merge_dict(config["env"], spec.get("env_overrides", {}))
    set_global_seed(seed)
    env = ProcessReasoningEnv(env_config, seed=seed)
    agent = DQNAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        hidden_size=int(training_cfg["hidden_size"]),
        device=device,
    )
    optimizer = torch.optim.Adam(agent.online_net.parameters(), lr=float(training_cfg["learning_rate"]))
    replay = ReplayBuffer(capacity=int(training_cfg["replay_size"]))
    rng = np.random.default_rng(seed)

    gamma = float(training_cfg["gamma"])
    batch_size = int(training_cfg["batch_size"])
    warmup_steps = int(training_cfg["warmup_steps"])
    updates_per_step = int(training_cfg["updates_per_step"])
    target_update_interval = int(training_cfg["target_update_interval"])
    epsilon_start = float(training_cfg["epsilon_start"])
    epsilon_end = float(training_cfg["epsilon_end"])
    epsilon_decay_fraction = float(training_cfg["epsilon_decay_fraction"])
    log_interval = int(training_cfg["log_interval"])

    train_rows: list[dict[str, Any]] = []
    rolling_returns: list[float] = []
    rolling_success: list[float] = []
    rolling_losses: list[float] = []
    global_step = 0

    for episode in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed * 100_000 + episode)
        done = False
        epsilon_progress = min(1.0, episode / max(1, int(episodes * epsilon_decay_fraction)))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * epsilon_progress
        episode_losses: list[float] = []
        while not done:
            action_mask = env.available_actions()
            action = agent.act(obs, action_mask, epsilon=epsilon, rng=rng)
            next_obs, reward, done, info = env.step(action)
            next_mask = env.available_actions()
            replay.push(obs, action, reward, next_obs, next_mask, done)
            obs = next_obs
            global_step += 1

            if len(replay) >= max(batch_size, warmup_steps):
                for _ in range(updates_per_step):
                    batch = replay.sample(batch_size, rng)
                    loss = agent.update(optimizer=optimizer, batch=batch, gamma=gamma)
                    episode_losses.append(loss)
                    rolling_losses.append(loss)
                if global_step % target_update_interval == 0:
                    agent.sync_target()

        metrics = info["episode_metrics"]
        row = {
            "variant": spec["name"],
            "seed": seed,
            "episode": episode,
            "epsilon": epsilon,
            "avg_loss": float(np.mean(episode_losses)) if episode_losses else np.nan,
            "replay_size": len(replay),
            **metrics,
        }
        train_rows.append(row)
        rolling_returns.append(metrics["episode_return"])
        rolling_success.append(metrics["success"])
        if len(rolling_returns) > log_interval:
            rolling_returns.pop(0)
            rolling_success.pop(0)
        if len(rolling_losses) > 5 * log_interval:
            rolling_losses = rolling_losses[-5 * log_interval :]
        if episode % log_interval == 0 or episode == episodes:
            mean_return = sum(rolling_returns) / len(rolling_returns)
            mean_success = sum(rolling_success) / len(rolling_success)
            mean_loss = float(np.mean(rolling_losses)) if rolling_losses else float("nan")
            print(
                f"[train] {spec['name']} seed={seed} episode={episode}/{episodes} "
                f"return={mean_return:.3f} success={mean_success:.3f} loss={mean_loss:.4f}"
            )

    agent.sync_target()
    checkpoint_path = checkpoint_dir / f"{spec['name']}_seed{seed}.pt"
    metadata = {
        "variant": spec["name"],
        "display_name": spec.get("display_name", spec["name"]),
        "seed": seed,
        "env_config": env_config,
        "algorithm": "masked_double_dqn",
        "hidden_size": int(training_cfg["hidden_size"]),
        "parameter_count": agent.parameter_count(),
    }
    agent.save(checkpoint_path, metadata)
    train_path = log_dir / f"{spec['name']}_seed{seed}.csv"
    pd.DataFrame(train_rows).to_csv(train_path, index=False)
    return {
        "variant": spec["name"],
        "seed": seed,
        "display_name": spec.get("display_name", spec["name"]),
        "checkpoint_path": str(checkpoint_path),
        "train_log_path": str(train_path),
        "parameter_count": agent.parameter_count(),
        "algorithm": "masked_double_dqn",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablations", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--log-dir", default="results/train_logs")
    parser.add_argument("--manifest-path", default="results/train_manifest.csv")
    args = parser.parse_args()

    config = load_yaml(args.config)
    ablations_cfg = load_yaml(args.ablations) if args.ablations else {}
    include_ablations = args.mode == "full"
    specs = build_rl_specs(config, ablations_cfg, include_ablations)
    training_cfg = config["training"]
    seeds = training_cfg["smoke_seeds"] if args.mode == "smoke" else training_cfg["seeds"]
    episodes = int(training_cfg["smoke_episodes"] if args.mode == "smoke" else training_cfg["episodes"])
    device = choose_device(prefer_mps=bool(training_cfg.get("prefer_mps", True)))
    print(f"[train] mode={args.mode} episodes={episodes} device={device}")

    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    log_dir = ensure_dir(args.log_dir)
    manifests = []
    for spec in specs:
        for seed in seeds:
            manifests.append(
                train_one_run(
                    spec=spec,
                    seed=int(seed),
                    config=config,
                    episodes=episodes,
                    checkpoint_dir=checkpoint_dir,
                    log_dir=log_dir,
                    device=device,
                )
            )
    pd.DataFrame(manifests).to_csv(args.manifest_path, index=False)


if __name__ == "__main__":
    main()
