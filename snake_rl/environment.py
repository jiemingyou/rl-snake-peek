from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np

from snake_rl.config import Config

# Actions (absolute directions)
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Direction vectors: (row_delta, col_delta)
DIRECTION_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# Opposite directions – the snake cannot reverse into itself
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class SnakeEnv:
    """Grid-based Snake environment with a multi-channel state tensor."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.grid_size = self.cfg.grid_size
        self.max_steps = self.cfg.max_steps_per_episode
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        mid = self.grid_size // 2
        self.snake = deque([(mid, mid)])  # head is snake[0]
        self.direction = RIGHT
        self.food = self._place_food()
        self.done = False
        self.steps_alive = 0
        self.score = 0
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")

        # Ignore action that would reverse the snake
        if len(self.snake) > 1 and action == OPPOSITE.get(self.direction):
            action = self.direction
        self.direction = action

        # Compute new head position
        dr, dc = DIRECTION_DELTA[self.direction]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        # Check wall collision
        r, c = new_head
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            self.done = True
            return self._get_state(), self.cfg.death_penalty, True, self._info()

        # Check self-collision
        if new_head in self.snake:
            self.done = True
            return self._get_state(), self.cfg.death_penalty, True, self._info()

        # Move
        self.snake.appendleft(new_head)
        reward = self.cfg.step_penalty

        if new_head == self.food:
            reward = self.cfg.food_reward
            self.score += 1
            self.food = self._place_food()
        else:
            self.snake.pop()  # remove tail

        self.steps_alive += 1
        if self.steps_alive >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done, self._info()

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """Return a (NUM_CHANNELS, grid_size, grid_size) float32 array."""
        state = np.zeros(
            (self.cfg.num_channels, self.grid_size, self.grid_size), dtype=np.float32
        )

        # Channel 0: head
        hr, hc = self.snake[0]
        state[0, hr, hc] = 1.0

        # Channel 1: body (gradient from 1.0 at neck to 0.5 at tail)
        body_len = len(self.snake) - 1
        for idx, (br, bc) in enumerate(list(self.snake)[1:]):
            if body_len == 1:
                state[1, br, bc] = 1.0
            else:
                state[1, br, bc] = 1.0 - 0.5 * (idx / (body_len - 1))

        # Channel 2: food
        if self.food is not None:
            fr, fc = self.food
            state[2, fr, fc] = 1.0

        # Channel 3: direction encoded at head position
        dir_val = {UP: 0.25, DOWN: 0.50, LEFT: 0.75, RIGHT: 1.0}
        state[3, hr, hc] = dir_val[self.direction]

        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _place_food(self) -> tuple[int, int] | None:
        occupied = set(self.snake)
        free = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in occupied
        ]
        if not free:
            return None
        return random.choice(free)

    def _info(self) -> dict[str, Any]:
        return {"length": len(self.snake), "steps_alive": self.steps_alive}
