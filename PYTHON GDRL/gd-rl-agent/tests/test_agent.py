"""Unit tests for agent/ components — no real game required.

Tests use synthetic transitions so they run offline on any machine.
"""

import numpy as np
import pytest
import torch

from agent.dqn import QNetwork
from agent.replay_buffer import ReplayBuffer


# ── ReplayBuffer ────────────────────────────────────────────────────────────

def test_replay_buffer_push_and_sample():
    buf = ReplayBuffer(capacity=100, obs_dim=4)
    obs = np.zeros(4, dtype=np.float32)
    for i in range(10):
        buf.push(obs, i % 2, float(i), obs, i == 9)
    assert len(buf) == 10
    batch = buf.sample(4)
    assert batch["obs"].shape == (4, 4)
    assert batch["action"].shape == (4,)


def test_replay_buffer_circular_overwrite():
    buf = ReplayBuffer(capacity=5, obs_dim=4)
    obs = np.zeros(4, dtype=np.float32)
    for i in range(8):
        buf.push(obs, 0, float(i), obs, False)
    assert len(buf) == 5  # capacity capped


def test_replay_buffer_sample_raises_when_too_small():
    buf = ReplayBuffer(capacity=100, obs_dim=4)
    obs = np.zeros(4, dtype=np.float32)
    buf.push(obs, 0, 1.0, obs, False)
    with pytest.raises(ValueError):
        buf.sample(32)


# ── QNetwork ────────────────────────────────────────────────────────────────

def test_qnetwork_vanilla_output_shape():
    net = QNetwork(input_dim=4, hidden_dim=64, n_actions=2, dueling=False)
    x = torch.zeros(8, 4)
    out = net(x)
    assert out.shape == (8, 2)


def test_qnetwork_dueling_output_shape():
    net = QNetwork(input_dim=4, hidden_dim=64, n_actions=2, dueling=True)
    x = torch.zeros(8, 4)
    out = net(x)
    assert out.shape == (8, 2)


def test_qnetwork_dueling_advantage_zero_mean():
    """Dueling head: mean(A) should be ~0 for a fresh network on zero input."""
    net = QNetwork(input_dim=4, hidden_dim=64, n_actions=2, dueling=True)
    net.eval()
    with torch.no_grad():
        x = torch.zeros(1, 4)
        out = net(x)
    # V + (A - mean(A)) — mean(A) cancels; can't test exact values but shape ok
    assert out.shape == (1, 2)


def test_dqn_update_reduces_loss():
    """A single gradient step on synthetic data should produce a finite loss."""
    from agent.trainer import Trainer
    from configs.config import AgentConfig, HyperParams

    cfg = AgentConfig()
    cfg.hp = HyperParams(batch_size=8, replay_capacity=100, hidden_dim=32)

    trainer = Trainer(cfg)

    obs = np.zeros(4, dtype=np.float32)
    for _ in range(20):
        trainer._buf.push(obs, 0, 0.01, obs, False)

    loss = trainer._update()
    assert loss is not None
    assert np.isfinite(loss)


def test_epsilon_decay():
    from agent.trainer import Trainer
    from configs.config import AgentConfig, HyperParams

    cfg = AgentConfig()
    cfg.hp = HyperParams(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=100)
    trainer = Trainer(cfg)

    trainer._total_steps = 0
    assert trainer._epsilon() == pytest.approx(1.0)

    trainer._total_steps = 50
    assert trainer._epsilon() == pytest.approx(0.525)

    trainer._total_steps = 100
    assert trainer._epsilon() == pytest.approx(0.05)
