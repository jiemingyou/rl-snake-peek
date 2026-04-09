#!/usr/bin/env python3
"""Entry point for training the Snake DQN agent."""

from __future__ import annotations

import argparse

from snake_rl.config import Config
from snake_rl.trainer import Trainer


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Snake")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps-decay-steps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--min-buffer", type=int, default=10_000)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument("--log-file", type=str, default="training_log.csv")
    parser.add_argument("--tb-log-dir", type=str, default="runs")

    args = parser.parse_args()
    return Config(
        seed=args.seed,
        grid_size=args.grid_size,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        eps_decay_steps=args.eps_decay_steps,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        min_buffer=args.min_buffer,
        target_update_freq=args.target_update_freq,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        log_file=args.log_file,
        tb_log_dir=args.tb_log_dir,
    )


def main() -> None:
    cfg = parse_args()
    print(f"Config: {cfg}")
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
