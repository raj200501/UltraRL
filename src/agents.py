from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.envs import Action, ProcessReasoningEnv, VerifyStatus
from src.utils import bool_mask_to_tensor


class TinyQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass
class ReplayTransition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    next_action_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[ReplayTransition] = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_action_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            ReplayTransition(
                obs=np.asarray(obs, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                next_action_mask=np.asarray(next_action_mask, dtype=bool),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[ReplayTransition]:
        indices = rng.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, device: torch.device) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.device = device
        self.online_net = TinyQNetwork(obs_dim, action_dim, hidden_size).to(device)
        self.target_net = TinyQNetwork(obs_dim, action_dim, hidden_size).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> int:
        valid_actions = np.flatnonzero(action_mask)
        if not deterministic and rng.random() < epsilon:
            return int(rng.choice(valid_actions))
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = bool_mask_to_tensor(action_mask, self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(obs_tensor)
            q_values = q_values.masked_fill(~mask_tensor, -1e9)
        return int(torch.argmax(q_values, dim=-1).item())

    def update(
        self,
        optimizer: torch.optim.Optimizer,
        batch: list[ReplayTransition],
        gamma: float,
    ) -> float:
        obs = torch.as_tensor(np.stack([item.obs for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([item.next_obs for item in batch]), dtype=torch.float32, device=self.device)
        next_masks = bool_mask_to_tensor(np.stack([item.next_action_mask for item in batch]), self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        current_q = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online_q = self.online_net(next_obs).masked_fill(~next_masks, -1e9)
            next_actions = next_online_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * gamma * next_target_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str | Path, metadata: dict[str, Any]) -> None:
        payload = {
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "metadata": metadata,
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        device: torch.device,
    ) -> tuple["DQNAgent", dict[str, Any]]:
        payload = torch.load(path, map_location=device)
        agent = cls(obs_dim=obs_dim, action_dim=action_dim, hidden_size=hidden_size, device=device)
        agent.online_net.load_state_dict(payload["online_state_dict"])
        agent.target_net.load_state_dict(payload["target_state_dict"])
        agent.online_net.eval()
        agent.target_net.eval()
        return agent, payload.get("metadata", {})

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.online_net.parameters())


@dataclass
class PolicyWrapper:
    name: str

    def act(self, obs: np.ndarray, env: ProcessReasoningEnv, action_mask: np.ndarray) -> int:
        raise NotImplementedError


class RandomPolicy(PolicyWrapper):
    def __init__(self, seed: int) -> None:
        super().__init__(name="random")
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, env: ProcessReasoningEnv, action_mask: np.ndarray) -> int:
        valid_actions = np.flatnonzero(action_mask)
        return int(self.rng.choice(valid_actions))


class AlwaysThinkCommitPolicy(PolicyWrapper):
    def __init__(self) -> None:
        super().__init__(name="always_think_commit")

    def act(self, obs: np.ndarray, env: ProcessReasoningEnv, action_mask: np.ndarray) -> int:
        if env.current_step >= env.depth or not action_mask[Action.THINK]:
            return int(Action.COMMIT)
        return int(Action.THINK)


class VerifyEveryStepPolicy(PolicyWrapper):
    def __init__(self) -> None:
        super().__init__(name="verify_every_step")

    def act(self, obs: np.ndarray, env: ProcessReasoningEnv, action_mask: np.ndarray) -> int:
        if env.current_step >= env.depth:
            return int(Action.COMMIT)
        if env.last_action == Action.VERIFY and env.last_verify_status == VerifyStatus.UNSAFE and action_mask[Action.REVISE]:
            return int(Action.REVISE)
        if env.current_step == 0 and action_mask[Action.THINK]:
            return int(Action.THINK)
        if env.last_action == Action.THINK and action_mask[Action.VERIFY]:
            return int(Action.VERIFY)
        if env.last_action == Action.REVISE and action_mask[Action.THINK]:
            return int(Action.THINK)
        if env.last_action == Action.VERIFY:
            if env.last_verify_status == VerifyStatus.SAFE and action_mask[Action.THINK]:
                return int(Action.THINK)
            if env.last_verify_status == VerifyStatus.UNSAFE and action_mask[Action.REVISE]:
                return int(Action.REVISE)
        if action_mask[Action.THINK]:
            return int(Action.THINK)
        return int(Action.COMMIT)


class HintTriggeredPolicy(PolicyWrapper):
    def __init__(self) -> None:
        super().__init__(name="hint_triggered")

    def act(self, obs: np.ndarray, env: ProcessReasoningEnv, action_mask: np.ndarray) -> int:
        current_hint = env.risk_hints[env.current_step] if env.current_step > 0 and env.risk_hints else 0
        if env.current_step >= env.depth:
            return int(Action.COMMIT)
        if env.last_action == Action.VERIFY and env.last_verify_status == VerifyStatus.UNSAFE and action_mask[Action.REVISE]:
            return int(Action.REVISE)
        if env.last_action == Action.REVISE and action_mask[Action.THINK]:
            return int(Action.THINK)
        if env.last_action == Action.THINK and current_hint == 1 and action_mask[Action.VERIFY]:
            return int(Action.VERIFY)
        if action_mask[Action.THINK]:
            return int(Action.THINK)
        return int(Action.COMMIT)


def build_heuristic(name: str, seed: int) -> PolicyWrapper:
    if name == "random":
        return RandomPolicy(seed)
    if name == "always_think_commit":
        return AlwaysThinkCommitPolicy()
    if name == "verify_every_step":
        return VerifyEveryStepPolicy()
    if name == "hint_triggered":
        return HintTriggeredPolicy()
    raise ValueError(f"Unknown heuristic policy: {name}")
