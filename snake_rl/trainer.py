from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from snake_rl.config import Config
from snake_rl.environment import SnakeEnv
from snake_rl.model import DQN
from snake_rl.replay_buffer import ReplayBuffer


class Trainer:
    """Standard DQN trainer with epsilon-greedy exploration."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = SnakeEnv(self.cfg)
        self.policy_net = DQN(self.cfg).to(self.device)
        self.target_net = DQN(self.cfg).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg)

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"g{self.cfg.grid_size}_lr{self.cfg.lr}_bs{self.cfg.batch_size}_{timestamp}"
        tb_dir = os.path.join(self.cfg.tb_log_dir, run_name)
        self.tb = SummaryWriter(log_dir=tb_dir)

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    def _epsilon(self, step: int) -> float:
        frac = min(1.0, step / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _select_action(self, state: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.randint(self.cfg.num_actions)
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q = self.policy_net(t)
            return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # TD loss (Huber)
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        return nn.functional.smooth_l1_loss(q_values, target)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        state = self.env.reset()

        # Per-episode accumulators
        ep_reward = 0.0
        ep_count = 0

        # Rolling stats for logging
        recent_rewards: list[float] = []
        recent_lengths: list[int] = []
        recent_steps_alive: list[int] = []
        loss_accum = 0.0
        q_accum = 0.0
        learn_count = 0

        log_path = Path(cfg.log_file)
        log_file = open(log_path, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "step",
                "episodes",
                "epsilon",
                "mean_reward",
                "mean_length",
                "mean_steps_alive",
                "mean_loss",
                "mean_q",
            ]
        )

        pbar = tqdm(range(1, cfg.total_steps + 1), desc="Training", unit="step")
        for step in pbar:
            eps = self._epsilon(step)
            action = self._select_action(state, eps)
            next_state, reward, done, info = self.env.step(action)
            self.buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # Learn
            if len(self.buffer) >= cfg.min_buffer:
                batch = self.buffer.sample(cfg.batch_size, self.device)
                loss = self._compute_loss(*batch)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
                self.optimizer.step()

                loss_accum += loss.item()
                with torch.no_grad():
                    q_accum += self.policy_net(batch[0]).max(dim=1).values.mean().item()
                learn_count += 1

            # Target network sync
            if step % cfg.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Episode end
            if done:
                recent_rewards.append(ep_reward)
                recent_lengths.append(info["length"])
                recent_steps_alive.append(info["steps_alive"])

                self.tb.add_scalar("episode/reward", ep_reward, ep_count)
                self.tb.add_scalar("episode/length", info["length"], ep_count)
                self.tb.add_scalar("episode/steps_alive", info["steps_alive"], ep_count)

                ep_reward = 0.0
                ep_count += 1
                state = self.env.reset()

            # Logging
            if step % cfg.log_every == 0 and recent_rewards:
                mean_r = np.mean(recent_rewards)
                mean_l = np.mean(recent_lengths)
                mean_s = np.mean(recent_steps_alive)
                mean_loss = loss_accum / max(learn_count, 1)
                mean_q = q_accum / max(learn_count, 1)

                writer.writerow(
                    [
                        step,
                        ep_count,
                        f"{eps:.4f}",
                        f"{mean_r:.3f}",
                        f"{mean_l:.1f}",
                        f"{mean_s:.1f}",
                        f"{mean_loss:.5f}",
                        f"{mean_q:.3f}",
                    ]
                )
                log_file.flush()

                self.tb.add_scalar("train/mean_reward", mean_r, step)
                self.tb.add_scalar("train/mean_length", mean_l, step)
                self.tb.add_scalar("train/mean_steps_alive", mean_s, step)
                self.tb.add_scalar("train/loss", mean_loss, step)
                self.tb.add_scalar("train/mean_q", mean_q, step)
                self.tb.add_scalar("train/epsilon", eps, step)

                pbar.set_postfix(
                    eps=f"{eps:.3f}",
                    R=f"{mean_r:.2f}",
                    len=f"{mean_l:.1f}",
                    loss=f"{mean_loss:.4f}",
                    Q=f"{mean_q:.2f}",
                )

                recent_rewards.clear()
                recent_lengths.clear()
                recent_steps_alive.clear()
                loss_accum = 0.0
                q_accum = 0.0
                learn_count = 0

            # Checkpointing
            if step % cfg.checkpoint_every == 0:
                self._save_checkpoint(step)

        # Final save
        self._save_checkpoint(cfg.total_steps)
        log_file.close()
        self.tb.close()
        print(f"\nTraining complete. Logs saved to {log_path}")
        print(f"TensorBoard logs in {cfg.tb_log_dir}/  — run: tensorboard --logdir {cfg.tb_log_dir}")

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int) -> None:
        path = os.path.join(self.cfg.checkpoint_dir, f"step_{step}.pt")
        torch.save(
            {
                "step": step,
                "model_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.cfg,
            },
            path,
        )
        tqdm.write(f"  ✓ Checkpoint saved: {path}")
