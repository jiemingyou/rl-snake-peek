# Snake RL with Grad-CAM Visualization

A CNN-based DQN learns to play Snake. Grad-CAM heatmaps reveal what the agent focuses on as it plays.

## Quick Start

```bash
uv sync                        # install deps (requires Python 3.12+, uv)
uv run python train.py         # train (~30-60 min for 500k steps on CPU)
uv run tensorboard --logdir runs  # monitor live in another terminal
```

## Evaluate with Grad-CAM

```bash
# single checkpoint
uv run python evaluate.py --checkpoint checkpoints/step_500000.pt --episodes 3 --output eval.mp4

# compare early / mid / late training
uv run python evaluate.py \
    --checkpoints checkpoints/step_50000.pt checkpoints/step_250000.pt checkpoints/step_500000.pt \
    --episodes 1 --output comparison.mp4

# sanity check: mask salient cells, measure Q-value shift
uv run python sanity_check.py --checkpoint checkpoints/step_500000.pt
```

Each video frame shows the game board, Grad-CAM overlay, and a Q-value panel side-by-side.

## Project Layout

```
train.py              Entry point: training
evaluate.py           Entry point: Grad-CAM video generation
sanity_check.py       Entry point: interpretability validation
snake_rl/
  config.py           Hyperparameters (single dataclass)
  environment.py      Snake game, multi-channel grid state
  model.py            3-conv-layer DQN (no pooling = native Grad-CAM resolution)
  replay_buffer.py    Circular replay buffer
  trainer.py          DQN training loop + TensorBoard + CSV logging
  gradcam.py          Grad-CAM on last conv layer
  visualizer.py       Board rendering, heatmap overlay, video export
```

## State Channels

`(4, 12, 12)` float32 tensor: **head** | **body** (gradient neck-to-tail) | **food** | **direction**

## CLI Overrides

All `Config` fields can be set via `train.py` flags, e.g.:

```bash
uv run python train.py --grid-size 16 --total-steps 1000000 --lr 3e-4
```
