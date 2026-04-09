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
from snake_rl.model import DQN
from snake_rl.replay_buffer import ReplayBuffer
from snake_rl.environment import VecSnakeEnv


class Trainer:
    """DQN trainer using a vectorized environment for parallel data collection."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vec_env = VecSnakeEnv(self.cfg)
        self.policy_net = DQN(self.cfg).to(self.device)
        self.target_net = DQN(self.cfg).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg, device=self.device)

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"g{self.cfg.grid_size}_lr{self.cfg.lr}_bs{self.cfg.batch_size}_{timestamp}"
        tb_dir = os.path.join(self.cfg.tb_log_dir, run_name)
        self.tb = SummaryWriter(log_dir=tb_dir)

    def _epsilon(self, step: int) -> float:
        """Linear decay from eps_start to eps_end over eps_decay_steps."""
        frac = min(1.0, step / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def _select_actions(self, states: np.ndarray, eps: float) -> np.ndarray:
        """Batched epsilon-greedy: one forward pass for all N envs."""
        n = states.shape[0]
        with torch.no_grad():
            t = torch.from_numpy(states).to(self.device)
            greedy = self.policy_net(t).argmax(dim=1).cpu().numpy()
        rand_mask = np.random.rand(n) < eps
        random_acts = np.random.randint(0, self.cfg.num_actions, size=n)
        return np.where(rand_mask, random_acts, greedy)

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

    def train(self) -> None:
        cfg = self.cfg
        n = cfg.num_envs
        states = self.vec_env.reset_all()

        ep_rewards = np.zeros(n, dtype=np.float32)
        ep_count = 0

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

        total_transitions = 0
        pbar = tqdm(total=cfg.total_steps, desc="Training", unit="step")

        while total_transitions < cfg.total_steps:
            eps = self._epsilon(total_transitions)
            actions = self._select_actions(states, eps)

            next_states, rewards, dones, infos = self.vec_env.step(actions)

            self.buffer.push_batch(
                states,
                actions.astype(np.int64),
                rewards,
                next_states,
                dones.astype(np.float32),
            )

            ep_rewards += rewards

            done_idx = np.where(dones)[0]
            for i in done_idx:
                recent_rewards.append(float(ep_rewards[i]))
                recent_lengths.append(int(infos["length"][i]))
                recent_steps_alive.append(int(infos["steps_alive"][i]))

                self.tb.add_scalar("episode/reward", ep_rewards[i], ep_count)
                self.tb.add_scalar("episode/length", infos["length"][i], ep_count)
                self.tb.add_scalar("episode/steps_alive", infos["steps_alive"][i], ep_count)
                ep_count += 1

            ep_rewards[done_idx] = 0.0

            states = next_states
            total_transitions += n

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

            # Use < n because total_transitions jumps by n each iteration
            if total_transitions % cfg.target_update_freq < n:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            pbar.update(n)

            if total_transitions % cfg.log_every < n and recent_rewards:
                mean_r = np.mean(recent_rewards)
                mean_l = np.mean(recent_lengths)
                mean_s = np.mean(recent_steps_alive)
                mean_loss = loss_accum / max(learn_count, 1)
                mean_q = q_accum / max(learn_count, 1)

                writer.writerow(
                    [
                        total_transitions,
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

                self.tb.add_scalar("train/mean_reward", mean_r, total_transitions)
                self.tb.add_scalar("train/mean_length", mean_l, total_transitions)
                self.tb.add_scalar("train/mean_steps_alive", mean_s, total_transitions)
                self.tb.add_scalar("train/loss", mean_loss, total_transitions)
                self.tb.add_scalar("train/mean_q", mean_q, total_transitions)
                self.tb.add_scalar("train/epsilon", eps, total_transitions)

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

            if total_transitions % cfg.checkpoint_every < n:
                self._save_checkpoint(total_transitions)

        pbar.close()

        self._save_checkpoint(total_transitions)
        log_file.close()
        self.tb.close()
        print(f"\nTraining complete. Logs saved to {log_path}")
        print(f"TensorBoard logs in {cfg.tb_log_dir}/  — run: tensorboard --logdir {cfg.tb_log_dir}")

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
        tqdm.write(f"  Checkpoint saved: {path}")
