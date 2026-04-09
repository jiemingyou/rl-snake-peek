#!/usr/bin/env python3
"""Sanity-check Grad-CAM by masking salient cells and measuring Q-value shifts.

For each sampled game state the script:
1. Runs the model to get Q-values and the Grad-CAM heatmap.
2. Masks (zeros out) the top-k most salient input cells across all channels.
3. Re-runs the model and checks whether the chosen action or Q-values changed.

A meaningful heatmap should cause noticeable Q-value shifts when masked.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from snake_rl.config import Config
from snake_rl.environment import SnakeEnv
from snake_rl.gradcam import compute_gradcam
from snake_rl.model import build_dqn_for_state_dict
from snake_rl.seeding import set_global_seed


def mask_top_k(state: np.ndarray, heatmap: np.ndarray, k: int) -> np.ndarray:
    """Zero out the top-k cells (across all channels) in *state*."""
    flat_idx = np.argsort(heatmap.ravel())[-k:]
    rows, cols = np.unravel_index(flat_idx, heatmap.shape)
    masked = state.copy()
    masked[:, rows, cols] = 0.0
    return masked


def run_check(
    checkpoint_path: str,
    num_states: int = 200,
    top_k: int = 5,
    seed: int = 42,
) -> None:
    set_global_seed(seed, deterministic_torch=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: Config = ckpt["config"]
    state_dict = ckpt["model_state_dict"]
    model: nn.Module = build_dqn_for_state_dict(cfg, state_dict).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    target_layer_name = "conv6" if hasattr(model, "conv6") else "conv3"

    env = SnakeEnv(cfg)

    action_changed = 0
    q_shifts: list[float] = []

    for i in range(num_states):
        if env.done:
            env.reset()
        state = env._get_state()

        with torch.no_grad():
            inp = torch.from_numpy(state).unsqueeze(0).to(device)
            q_orig = model(inp).squeeze(0).cpu().numpy()
        action_orig = int(np.argmax(q_orig))

        heatmap = compute_gradcam(
            model,
            state,
            action_orig,
            target_layer_name=target_layer_name,
            device=device,
        )

        masked = mask_top_k(state, heatmap, k=top_k)
        with torch.no_grad():
            inp_m = torch.from_numpy(masked).unsqueeze(0).to(device)
            q_masked = model(inp_m).squeeze(0).cpu().numpy()
        action_masked = int(np.argmax(q_masked))

        if action_masked != action_orig:
            action_changed += 1
        q_shifts.append(float(np.abs(q_orig - q_masked).mean()))

        # Advance the game by a random action to get diverse states
        random_action = np.random.randint(cfg.num_actions)
        env.step(random_action)

    print(f"\n=== Grad-CAM Sanity Check ({checkpoint_path}) ===")
    print(f"States sampled : {num_states}")
    print(f"Top-k masked   : {top_k}")
    print(f"Action changed : {action_changed}/{num_states} "
          f"({100 * action_changed / num_states:.1f}%)")
    print(f"Mean |ΔQ|      : {np.mean(q_shifts):.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM sanity check")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-states", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_check(args.checkpoint, args.num_states, args.top_k, args.seed)


if __name__ == "__main__":
    main()
