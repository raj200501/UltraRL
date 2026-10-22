"""Train one masked Double DQN agent and record a greedy learning curve."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ultrarl.config import ExperimentConfig, Method
from ultrarl.dqn import DQNAgent, ReplayBuffer
from ultrarl.env import ChainEnv
from ultrarl.rollout import CURVE_STREAM, TRAIN_STREAM, episode_seed, evaluate

AGENT_STREAM = 3  # exploration and replay sampling


def epsilon_at(episode: int, episodes: int, start: float, end: float, decay_fraction: float) -> float:
    progress = min(1.0, episode / max(1, int(episodes * decay_fraction)))
    return start + (end - start) * progress


def greedy_act(agent: DQNAgent):
    return lambda obs, _observable, mask: agent.act(obs, mask)


def checkpoint_path(out_dir: Path, method: Method, seed: int, which: str = "best") -> Path:
    """``which="best"``: selected on train-length validation; ``"final"``: end of training."""
    if which not in ("best", "final"):
        raise ValueError(f"unknown checkpoint kind {which!r}")
    return out_dir / "checkpoints" / f"{method.name}_seed{seed}_{which}.pt"


def train_run(exp: ExperimentConfig, method: Method, seed: int, out_dir: Path, verbose: bool = False) -> dict[str, Any]:
    """Train ``method`` with ``seed``; write checkpoint and learning curve under ``out_dir``."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    tc = exp.method_train(method)
    env_config = exp.method_env(method)
    env = ChainEnv(env_config)
    agent = DQNAgent(env.obs_dim, env.num_actions, tc.hidden_size)
    optimizer = torch.optim.Adam(agent.online.parameters(), lr=tc.learning_rate)
    replay = ReplayBuffer(tc.replay_size, env.obs_dim, env.num_actions)
    rng = np.random.default_rng([AGENT_STREAM, seed])

    curve: list[dict[str, Any]] = []
    best_state, best_score, best_episode = copy.deepcopy(agent.online.state_dict()), -1.0, 0
    recent_success: list[float] = []
    global_step = 0
    start_time = time.time()
    for episode in range(1, tc.episodes + 1):
        epsilon = epsilon_at(episode, tc.episodes, tc.epsilon_start, tc.epsilon_end, tc.epsilon_decay_fraction)
        obs = env.reset(seed=episode_seed(TRAIN_STREAM, seed, episode))
        done = False
        while not done:
            mask = env.action_mask()
            action = agent.act(obs, mask, epsilon, rng)
            next_obs, reward, done, info = env.step(action)
            replay.push(obs, action, reward, next_obs, env.action_mask(), done)
            obs = next_obs
            global_step += 1
            if len(replay) >= max(tc.batch_size, tc.warmup_steps):
                agent.update(optimizer, replay.sample(tc.batch_size, rng), tc.gamma, tc.grad_clip)
                if tc.target_tau > 0:
                    agent.soft_update_target(tc.target_tau)
                elif global_step % tc.target_update_interval == 0:
                    agent.sync_target()
        recent_success.append(info["metrics"]["success"])

        if episode % tc.curve_interval == 0 or episode == tc.episodes:
            point: dict[str, Any] = {
                "method": method.name,
                "seed": seed,
                "episode": episode,
                "env_steps": global_step,
                "epsilon": epsilon,
                "train_success_recent": float(np.mean(recent_success[-tc.curve_interval :])),
            }
            for setting in exp.curve_settings:
                metrics = evaluate(
                    greedy_act(agent),
                    exp.method_env(method, setting),
                    setting,
                    seed,
                    tc.curve_episodes,
                    stream=CURVE_STREAM,
                )
                point[f"{setting.name}_success"] = metrics["success"]
                point[f"{setting.name}_fault_detected"] = metrics["fault_detected"]
            curve.append(point)
            score = point[f"{tc.select_setting}_success"]
            if score >= best_score:  # ties go to the later (longer-trained) checkpoint
                best_state, best_score, best_episode = copy.deepcopy(agent.online.state_dict()), score, episode
            if verbose:
                evals = " ".join(f"{s.name}={point[f'{s.name}_success']:.3f}" for s in exp.curve_settings)
                print(f"[train] {method.name} seed={seed} ep={episode} eps={epsilon:.3f} {evals}", flush=True)

    metadata = {
        "method": method.name,
        "seed": seed,
        "episodes": tc.episodes,
        "train_config": asdict(tc),
        "env_steps": global_step,
        "env_config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(env_config).items()},
        "parameter_count": agent.parameter_count(),
    }
    metadata["wall_seconds"] = round(time.time() - start_time, 1)
    final_path = checkpoint_path(out_dir, method, seed, "final")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(final_path, {**metadata, "checkpoint": "final", "checkpoint_episode": tc.episodes})
    agent.online.load_state_dict(best_state)
    agent.save(
        checkpoint_path(out_dir, method, seed, "best"),
        {**metadata, "checkpoint": "best", "checkpoint_episode": best_episode, "selection_success": best_score},
    )
    curve_dir = out_dir / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve).to_csv(curve_dir / f"{method.name}_seed{seed}.csv", index=False)
    return {
        "method": method.name,
        "seed": seed,
        "env_steps": global_step,
        "parameter_count": agent.parameter_count(),
        "best_episode": best_episode,
        "best_selection_success": best_score,
        "wall_seconds": round(time.time() - start_time, 1),
    }
