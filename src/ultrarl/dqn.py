"""A small masked Double DQN (van Hasselt et al., 2016) with a uniform replay buffer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

MASKED_Q = -1e9


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ReplayBuffer:
    """Fixed-capacity ring buffer stored as preallocated numpy arrays."""

    def __init__(self, capacity: int, obs_dim: int, num_actions: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, num_actions), dtype=bool)
        self.size = 0
        self._next = 0

    def push(self, obs, action, reward, next_obs, next_mask, done) -> None:
        i = self._next
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.next_masks[i] = next_mask
        self.dones[i] = float(done)
        self._next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "rewards": torch.from_numpy(self.rewards[idx]),
            "next_obs": torch.from_numpy(self.next_obs[idx]),
            "next_masks": torch.from_numpy(self.next_masks[idx]),
            "dones": torch.from_numpy(self.dones[idx]),
        }

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int) -> None:
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.online = QNetwork(obs_dim, num_actions, hidden_size)
        self.target = QNetwork(obs_dim, num_actions, hidden_size)
        self.sync_target()
        self.target.eval()

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self.online(torch.from_numpy(np.asarray(obs, dtype=np.float32)[None])).numpy()[0]

    def act(
        self, obs: np.ndarray, mask: np.ndarray, epsilon: float = 0.0, rng: np.random.Generator | None = None
    ) -> int:
        """Epsilon-greedy over available actions (greedy when ``epsilon == 0``)."""
        if epsilon > 0.0 and rng is not None and rng.random() < epsilon:
            return int(rng.choice(np.flatnonzero(mask)))
        q = np.where(mask, self.q_values(obs), MASKED_Q)
        return int(np.argmax(q))

    def update(
        self, optimizer: torch.optim.Optimizer, batch: dict[str, torch.Tensor], gamma: float, grad_clip: float
    ) -> float:
        q = self.online(batch["obs"]).gather(1, batch["actions"][:, None]).squeeze(1)
        with torch.no_grad():
            next_online = self.online(batch["next_obs"]).masked_fill(~batch["next_masks"], MASKED_Q)
            next_actions = next_online.argmax(dim=1, keepdim=True)
            next_q = self.target(batch["next_obs"]).gather(1, next_actions).squeeze(1)
            target = batch["rewards"] + gamma * (1.0 - batch["dones"]) * next_q
        loss = nn.functional.smooth_l1_loss(q, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), grad_clip)
        optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    @torch.no_grad()
    def soft_update_target(self, tau: float) -> None:
        for target, online in zip(self.target.parameters(), self.online.parameters(), strict=True):
            target.mul_(1.0 - tau).add_(online, alpha=tau)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.online.parameters())

    def save(self, path: str | Path, metadata: dict[str, Any]) -> None:
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "num_actions": self.num_actions,
                "hidden_size": self.hidden_size,
                "state_dict": self.online.state_dict(),
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple[DQNAgent, dict[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(payload["obs_dim"], payload["num_actions"], payload["hidden_size"])
        agent.online.load_state_dict(payload["state_dict"])
        agent.sync_target()
        agent.online.eval()
        return agent, payload["metadata"]
