import numpy as np
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
