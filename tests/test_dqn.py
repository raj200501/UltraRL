import numpy as np
import pytest
import torch

from ultrarl.dqn import DQNAgent, ReplayBuffer


def test_greedy_action_respects_mask():
    torch.manual_seed(0)
    agent = DQNAgent(obs_dim=4, num_actions=3, hidden_size=8)
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.normal(size=4).astype(np.float32)
        mask = np.array([False, True, False])
        assert agent.act(obs, mask) == 1
        assert agent.act(obs, mask, epsilon=1.0, rng=rng) == 1


def test_replay_buffer_wraps_around():
    buf = ReplayBuffer(capacity=3, obs_dim=1, num_actions=2)
    for i in range(5):
        buf.push([i], i % 2, float(i), [i + 1], [True, False], i == 4)
    assert len(buf) == 3
    assert sorted(buf.rewards.tolist()) == [2.0, 3.0, 4.0]
    batch = buf.sample(8, np.random.default_rng(0))
    assert batch["obs"].shape == (8, 1) and batch["next_masks"].dtype == torch.bool


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    agent = DQNAgent(obs_dim=5, num_actions=4, hidden_size=16)
    path = tmp_path / "agent.pt"
    agent.save(path, {"note": "x", "n": 3})
    loaded, meta = DQNAgent.load(path)
    obs = np.ones(5, dtype=np.float32)
    assert np.allclose(agent.q_values(obs), loaded.q_values(obs))
    assert meta == {"note": "x", "n": 3}


def test_learns_a_two_armed_bandit():
    # One-step episodes: action 1 pays +1, action 0 pays 0. TD targets have no bootstrap term.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    agent = DQNAgent(obs_dim=2, num_actions=2, hidden_size=16)
    opt = torch.optim.Adam(agent.online.parameters(), lr=1e-2)
    buf = ReplayBuffer(256, 2, 2)
    obs = np.array([1.0, 0.0], dtype=np.float32)
    for _ in range(256):
        a = int(rng.integers(2))
        buf.push(obs, a, float(a), obs, [True, True], True)
    for _ in range(300):
        agent.update(opt, buf.sample(32, rng), gamma=0.99, grad_clip=1.0)
    q = agent.q_values(obs)
    assert agent.act(obs, np.array([True, True])) == 1
    assert abs(q[1] - 1.0) < 0.1 and abs(q[0]) < 0.1


def test_soft_update_moves_target_towards_online():
    torch.manual_seed(0)
    agent = DQNAgent(obs_dim=3, num_actions=2, hidden_size=4)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.add_(1.0)
    before = [p.clone() for p in agent.target.parameters()]
    agent.soft_update_target(0.25)
    for b, t, o in zip(before, agent.target.parameters(), agent.online.parameters(), strict=True):
        assert torch.allclose(t, 0.75 * b + 0.25 * o)


def _set_linear_q(net, q_per_action):
    """Make a QNetwork output constant Q-values regardless of the input."""
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.net[-1].bias.copy_(torch.tensor(q_per_action))


def test_double_dqn_target_masks_terminates_and_decouples_networks():
    agent = DQNAgent(obs_dim=2, num_actions=3, hidden_size=4)
    _set_linear_q(agent.online, [5.0, 1.0, 3.0])  # online prefers action 0, then 2
    _set_linear_q(agent.target, [10.0, 20.0, 30.0])
    batch = {
        "next_obs": torch.zeros(3, 2),
        "rewards": torch.tensor([1.0, 1.0, 1.0]),
        "dones": torch.tensor([0.0, 0.0, 1.0]),
        "next_masks": torch.tensor([[True, True, True], [False, True, True], [True, True, True]]),
    }
    target = agent.td_target(batch, gamma=0.5)
    # row 0: online argmax = 0 -> target net value 10; row 1: action 0 masked -> argmax 2 -> 30;
    # row 2: terminal -> no bootstrap.
    assert torch.allclose(target, torch.tensor([1.0 + 0.5 * 10.0, 1.0 + 0.5 * 30.0, 1.0]))


def test_epsilon_without_rng_is_an_error():
    agent = DQNAgent(obs_dim=2, num_actions=2, hidden_size=4)
    with pytest.raises(ValueError):
        agent.act(np.zeros(2, dtype=np.float32), np.array([True, True]), epsilon=0.5)
