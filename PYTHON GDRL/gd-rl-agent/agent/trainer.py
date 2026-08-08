"""DQN trainer: collects transitions from the real GD env and trains the Q-network.

Supports vanilla / Double / Dueling DQN via config flags.
Logs reward, epsilon, and loss per episode to stdout and optionally to a CSV file.
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agent.dqn import QNetwork
from agent.replay_buffer import ReplayBuffer
from configs.config import AgentConfig
from env.gd_env import GDEnv


class Trainer:
    """DQN training loop.

    Args:
        config: Full agent config.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config
        hp = config.hp

        obs_dim = 6 if hp.use_obstacle_lookahead else 4
        self._online = QNetwork(obs_dim, hp.hidden_dim, 2, hp.use_dueling_dqn)
        self._target = QNetwork(obs_dim, hp.hidden_dim, 2, hp.use_dueling_dqn)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        self._opt = torch.optim.Adam(self._online.parameters(), lr=hp.lr)
        self._buf = ReplayBuffer(hp.replay_capacity, obs_dim)

        self._gamma = config.reward.gamma
        self._batch = hp.batch_size
        self._target_freq = hp.target_update_freq
        self._eps_start = hp.epsilon_start
        self._eps_end = hp.epsilon_end
        self._eps_steps = hp.epsilon_decay_steps
        self._double = hp.use_double_dqn

        self._total_steps = 0
        self._log_dir = config.log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def _epsilon(self) -> float:
        frac = min(self._total_steps / max(self._eps_steps, 1), 1.0)
        return self._eps_start + frac * (self._eps_end - self._eps_start)

    def _select_action(self, obs: np.ndarray) -> int:
        if np.random.random() < self._epsilon():
            return np.random.randint(2)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(self._online(t).argmax(dim=-1).item())

    def _update(self) -> float | None:
        if len(self._buf) < self._batch:
            return None

        batch = self._buf.sample(self._batch)
        obs      = torch.tensor(batch["obs"],      dtype=torch.float32)
        actions  = torch.tensor(batch["action"],   dtype=torch.long)
        rewards  = torch.tensor(batch["reward"],   dtype=torch.float32)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32)
        terminals = torch.tensor(batch["terminal"], dtype=torch.float32)

        with torch.no_grad():
            if self._double:
                next_actions = self._online(next_obs).argmax(dim=-1)
                next_q = self._target(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self._target(next_obs).max(dim=-1).values
            target_q = rewards + self._gamma * next_q * (1.0 - terminals)

        current_q = self._online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(current_q, target_q)

        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._online.parameters(), 10.0)
        self._opt.step()

        if self._total_steps % self._target_freq == 0:
            self._target.load_state_dict(self._online.state_dict())

        return loss.item()

    # ------------------------------------------------------------------

    def train(self, n_episodes: int, save_every: int = 50) -> None:
        """Run training for n_episodes episodes.

        Args:
            n_episodes: Total number of episodes to run.
            save_every: Save checkpoint every this many episodes.
        """
        log_path = self._log_dir / "train_log.csv"
        file_exists = log_path.exists()
        with GDEnv(self._cfg) as env, open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["episode", "steps", "reward", "epsilon", "loss", "success"])

            for ep in range(1, n_episodes + 1):
                obs = env.reset()
                ep_reward = 0.0
                ep_loss: list[float] = []
                ep_steps = 0

                while True:
                    action = self._select_action(obs)
                    result = env.step(action)

                    self._buf.push(obs, action, result.reward, result.obs, result.terminal)
                    loss = self._update()
                    if loss is not None:
                        ep_loss.append(loss)

                    ep_reward += result.reward
                    ep_steps += 1
                    self._total_steps += 1
                    obs = result.obs

                    if result.terminal:
                        break

                mean_loss = float(np.mean(ep_loss)) if ep_loss else float("nan")
                eps = self._epsilon()
                print(
                    f"ep={ep:4d}  steps={ep_steps:4d}  "
                    f"reward={ep_reward:7.3f}  eps={eps:.3f}  "
                    f"loss={mean_loss:.4f}  success={result.success}"
                )
                writer.writerow([ep, ep_steps, ep_reward, eps, mean_loss, result.success])
                f.flush()

                if ep % save_every == 0:
                    self._save(ep)

        print(f"Training complete. Log: {log_path}")

    def _save(self, episode: int) -> None:
        path = self._log_dir / f"ckpt_ep{episode:04d}.pt"
        torch.save({
            "episode": episode,
            "online_state_dict": self._online.state_dict(),
            "optimizer_state_dict": self._opt.state_dict(),
            "total_steps": self._total_steps,
        }, path)
        print(f"  Saved checkpoint → {path}")

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self._online.load_state_dict(ckpt["online_state_dict"])
        self._target.load_state_dict(ckpt["online_state_dict"])
        self._opt.load_state_dict(ckpt["optimizer_state_dict"])
        self._total_steps = ckpt["total_steps"]
        print(f"Loaded checkpoint from {path} (step {self._total_steps})")
