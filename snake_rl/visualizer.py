"""Board rendering, Grad-CAM overlay, info panel, and video export."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import imageio.v3 as iio

ACTION_NAMES = ["Up", "Down", "Left", "Right"]
CELL_PX = 30

_BG = np.array([30, 30, 30], dtype=np.uint8)
_HEAD = np.array([0, 200, 80], dtype=np.uint8)
_BODY = np.array([0, 150, 60], dtype=np.uint8)
_FOOD = np.array([220, 50, 50], dtype=np.uint8)
_GRID_LINE = np.array([55, 55, 55], dtype=np.uint8)


def render_board(state: np.ndarray, grid_size: int) -> np.ndarray:
    """Draw the Snake board from the multi-channel state tensor."""
    size = grid_size * CELL_PX
    img = np.tile(_BG, (size, size, 1)).copy()

    head_ch = state[0]
    body_ch = state[1]
    food_ch = state[2]

    for r in range(grid_size):
        for c in range(grid_size):
            y0, y1 = r * CELL_PX, (r + 1) * CELL_PX
            x0, x1 = c * CELL_PX, (c + 1) * CELL_PX

            if head_ch[r, c] > 0:
                img[y0:y1, x0:x1] = _HEAD
            elif body_ch[r, c] > 0:
                # Fade body colour by the gradient value to show ordering
                alpha = body_ch[r, c]
                img[y0:y1, x0:x1] = (_BODY.astype(np.float32) * alpha).astype(np.uint8)
            elif food_ch[r, c] > 0:
                img[y0:y1, x0:x1] = _FOOD

    for i in range(grid_size + 1):
        px = i * CELL_PX
        img[px : px + 1, :] = _GRID_LINE
        img[:, px : px + 1] = _GRID_LINE

    return img


def overlay_heatmap(
    board_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Alpha-blend a Grad-CAM heatmap (jet colourmap) onto the board image."""
    h, w, _ = board_img.shape
    # Nearest-neighbour upscale so each cell gets a uniform colour
    hm_large = np.kron(heatmap, np.ones((CELL_PX, CELL_PX)))
    hm_large = hm_large[:h, :w]

    colour_map = cm.get_cmap("jet")
    hm_rgb = (colour_map(hm_large)[:, :, :3] * 255).astype(np.uint8)

    blended = (
        (1 - alpha) * board_img.astype(np.float32)
        + alpha * hm_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    return blended


def render_info_panel(
    action: int,
    q_values: np.ndarray,
    score: int,
    step: int,
    episode: int | None = None,
    panel_height_px: int | None = None,
) -> np.ndarray:
    """Render a matplotlib side panel showing Q-values, action, and score."""
    fig_h = (panel_height_px or 360) / 100
    fig, ax = plt.subplots(figsize=(3.0, fig_h), dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    colours = ["#6688cc"] * len(ACTION_NAMES)
    colours[action] = "#44dd88"
    ax.barh(ACTION_NAMES, q_values, color=colours)

    ax.set_xlabel("Q-value", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")

    title_parts = [f"Action: {ACTION_NAMES[action]}   Score: {score}   Step: {step}"]
    if episode is not None:
        title_parts.append(f"Episode: {episode}")
    ax.set_title("  |  ".join(title_parts), color="white", fontsize=8, pad=8)

    fig.tight_layout()

    # Rasterize the figure to a numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    panel = buf[:, :, :3].copy()
    plt.close(fig)

    return panel


def compose_frame(
    board_img: np.ndarray,
    overlay_img: np.ndarray,
    info_panel: np.ndarray,
) -> np.ndarray:
    """Stitch board, overlay, and info panel side-by-side, padding height to match."""
    target_h = max(board_img.shape[0], overlay_img.shape[0], info_panel.shape[0])

    def _pad_v(img: np.ndarray) -> np.ndarray:
        h, w, c = img.shape
        if h >= target_h:
            return img[:target_h]
        pad = np.zeros((target_h - h, w, c), dtype=img.dtype)
        return np.vstack([img, pad])

    return np.hstack([_pad_v(board_img), _pad_v(overlay_img), _pad_v(info_panel)])


def save_video(frames: list[np.ndarray], path: str, fps: int = 8) -> None:
    """Write frames to MP4 or GIF depending on file extension."""
    p = Path(path)
    if p.suffix.lower() == ".gif":
        iio.imwrite(p, frames, duration=int(1000 / fps), loop=0)
    else:
        iio.imwrite(p, frames, fps=fps, codec="libx264")
    print(f"Saved video: {p}  ({len(frames)} frames, {fps} fps)")
