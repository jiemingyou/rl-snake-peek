#!/usr/bin/env python3
"""Run a trained Snake agent and produce Grad-CAM evaluation videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from snake_rl.config import Config
from snake_rl.environment import SnakeEnv
from snake_rl.gradcam import compute_gradcam
from snake_rl.model import DQN
from snake_rl.visualizer import (
    compose_frame,
    overlay_heatmap,
    render_board,
    render_info_panel,
    save_video,
    CELL_PX,
)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[DQN, Config]:
    """Load a DQN model from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: Config = ckpt["config"]
    model = DQN(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def run_episode(
    model: DQN,
    cfg: Config,
    device: torch.device,
    episode_num: int = 0,
) -> list[np.ndarray]:
    """Play one greedy episode and return composed frames."""
    env = SnakeEnv(cfg)
    state = env.reset()
    frames: list[np.ndarray] = []
    board_px = cfg.grid_size * CELL_PX
    step = 0

    while True:
        with torch.no_grad():
            inp = torch.from_numpy(state).unsqueeze(0).to(device)
            q_values = model(inp).squeeze(0).cpu().numpy()

        action = int(np.argmax(q_values))
        heatmap = compute_gradcam(model, state, action, device=device)

        board_img = render_board(state, cfg.grid_size)
        overlay_img = overlay_heatmap(board_img, heatmap)
        info_panel = render_info_panel(
            action=action,
            q_values=q_values,
            score=env.score,
            step=step,
            episode=episode_num,
            panel_height_px=board_px,
        )

        frame = compose_frame(board_img, overlay_img, info_panel)
        frames.append(frame)

        state, _, done, _ = env.step(action)
        step += 1
        if done:
            break

    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a single checkpoint, or first of several for comparison",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="Multiple checkpoint paths for side-by-side comparison",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="eval_video.mp4")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_paths = args.checkpoints or [args.checkpoint]

    all_frames: list[np.ndarray] = []

    for ckpt_path in checkpoint_paths:
        print(f"\nLoading checkpoint: {ckpt_path}")
        model, cfg = load_model(ckpt_path, device)

        for ep in range(args.episodes):
            print(f"  Episode {ep + 1}/{args.episodes} ...", end=" ")
            frames = run_episode(model, cfg, device, episode_num=ep + 1)
            print(f"{len(frames)} frames")
            all_frames.extend(frames)

    save_video(all_frames, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
