"""Snake environments: single-game (SnakeEnv) and vectorized (VecSnakeEnv)."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np

from snake_rl.config import Config

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

DIRECTION_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# Used to prevent the snake from reversing into itself
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# Lookup tables indexed by action id for vectorized head movement
_DR = np.array([-1, 1, 0, 0], dtype=np.int32)
_DC = np.array([0, 0, -1, 1], dtype=np.int32)
_OPPOSITE = np.array([DOWN, UP, RIGHT, LEFT], dtype=np.int32)
_DIR_VAL = np.array([0.25, 0.50, 0.75, 1.0], dtype=np.float32)


class SnakeEnv:
    """Single-game Snake environment used for evaluation and sanity checks."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.grid_size = self.cfg.grid_size
        self.max_steps = self.cfg.max_steps_per_episode
        self.reset()

    def reset(self) -> np.ndarray:
        mid = self.grid_size // 2
        self.snake = deque([(mid, mid)])
        self.direction = RIGHT
        self.food = self._place_food()
        self.done = False
        self.steps_alive = 0
        self.score = 0
        self._terminated = False  # True only for wall/self-collision, False for timeout
        self._prev_potential = self._potential()
        return self._get_state()

    def _potential(self) -> float:
        """Φ(s) = -manhattan(head, food) / grid_size. Zero when food absent."""
        if self.food is None:
            return 0.0
        hr, hc = self.snake[0]
        fr, fc = self.food
        return -(abs(hr - fr) + abs(hc - fc)) / self.grid_size

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")

        if len(self.snake) > 1 and action == OPPOSITE.get(self.direction):
            action = self.direction
        self.direction = action

        dr, dc = DIRECTION_DELTA[self.direction]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        r, c = new_head
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            self.done = True
            self._terminated = True
            return self._get_state(), self.cfg.death_penalty, True, self._info()

        if new_head in self.snake:
            self.done = True
            self._terminated = True
            return self._get_state(), self.cfg.death_penalty, True, self._info()

        self.snake.appendleft(new_head)
        reward = self.cfg.step_penalty

        if new_head == self.food:
            reward = self.cfg.food_reward
            self.score += 1
            self.food = self._place_food()
        else:
            self.snake.pop()

        # Potential-based shaping: γ·Φ(s') - Φ(s)
        new_potential = self._potential()
        reward += self.cfg.distance_shaping * (
            self.cfg.gamma * new_potential - self._prev_potential
        )
        self._prev_potential = new_potential

        self.steps_alive += 1
        if self.steps_alive >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done, self._info()

    def _get_state(self) -> np.ndarray:
        """Build a (C, H, W) float32 state tensor with 4 semantic channels."""
        state = np.zeros(
            (self.cfg.num_channels, self.grid_size, self.grid_size), dtype=np.float32
        )

        hr, hc = self.snake[0]
        state[0, hr, hc] = 1.0

        # Body channel uses a gradient so the model can distinguish head-end from tail-end
        body_len = len(self.snake) - 1
        for idx, (br, bc) in enumerate(list(self.snake)[1:]):
            if body_len == 1:
                state[1, br, bc] = 1.0
            else:
                state[1, br, bc] = 1.0 - 0.5 * (idx / (body_len - 1))

        if self.food is not None:
            fr, fc = self.food
            state[2, fr, fc] = 1.0

        # Encode direction as a scalar at the head cell so the model knows current heading
        dir_val = {UP: 0.25, DOWN: 0.50, LEFT: 0.75, RIGHT: 1.0}
        state[3, hr, hc] = dir_val[self.direction]

        return state

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
        return {
            "length": len(self.snake),
            "steps_alive": self.steps_alive,
            "terminated": self._terminated,
        }


class VecSnakeEnv:
    """N independent Snake games stepped in lockstep via batched numpy arrays.

    Body tracking uses a countdown-timer grid: each occupied cell stores the
    number of steps until that segment disappears. On every step, all nonzero
    cells are decremented by 1, and the new head cell is set to ``length``.
    This replaces per-snake deques with a single array operation.
    """

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.n = self.cfg.num_envs
        self.gs = self.cfg.grid_size
        self.max_steps = self.cfg.max_steps_per_episode
        self._arange = np.arange(self.n)
        self._all_cells = np.arange(self.gs * self.gs)

        self.head_r = np.zeros(self.n, dtype=np.int32)
        self.head_c = np.zeros(self.n, dtype=np.int32)
        self.direction = np.zeros(self.n, dtype=np.int32)
        self.body_grid = np.zeros((self.n, self.gs, self.gs), dtype=np.int32)
        self.length = np.ones(self.n, dtype=np.int32)
        self.score = np.zeros(self.n, dtype=np.int32)
        self.steps_alive = np.zeros(self.n, dtype=np.int32)
        self.alive = np.ones(self.n, dtype=bool)
        self.food_r = np.full(self.n, -1, dtype=np.int32)
        self.food_c = np.full(self.n, -1, dtype=np.int32)
        self._prev_potential = np.zeros(self.n, dtype=np.float32)

    def _potentials(self) -> np.ndarray:
        """Φ(s) = -manhattan(head, food) / gs for each env. Zero when food absent."""
        valid = self.food_r >= 0
        pot = np.zeros(self.n, dtype=np.float32)
        pot[valid] = -(
            np.abs(self.head_r[valid] - self.food_r[valid])
            + np.abs(self.head_c[valid] - self.food_c[valid])
        ).astype(np.float32) / self.gs
        return pot

    def reset_all(self) -> np.ndarray:
        """Reset every env. Returns states (N, C, H, W)."""
        mid = self.gs // 2
        self.head_r = np.full(self.n, mid, dtype=np.int32)
        self.head_c = np.full(self.n, mid, dtype=np.int32)
        self.direction = np.full(self.n, RIGHT, dtype=np.int32)

        self.body_grid = np.zeros((self.n, self.gs, self.gs), dtype=np.int32)
        self.body_grid[self._arange, self.head_r, self.head_c] = 1

        self.length = np.ones(self.n, dtype=np.int32)
        self.score = np.zeros(self.n, dtype=np.int32)
        self.steps_alive = np.zeros(self.n, dtype=np.int32)
        self.alive = np.ones(self.n, dtype=bool)

        self.food_r = np.full(self.n, -1, dtype=np.int32)
        self.food_c = np.full(self.n, -1, dtype=np.int32)
        self._place_food_batch(self._arange)
        self._prev_potential = self._potentials()

        return self._get_states()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Step all N envs. Dead envs auto-reset after returning terminal info."""

        # Prevent the snake from reversing into its own neck
        reverse_mask = (self.length > 1) & (actions == _OPPOSITE[self.direction])
        actions = np.where(reverse_mask, self.direction, actions)
        self.direction = actions

        new_r = self.head_r + _DR[actions]
        new_c = self.head_c + _DC[actions]

        wall_hit = (new_r < 0) | (new_r >= self.gs) | (new_c < 0) | (new_c >= self.gs)

        # Clamp out-of-bounds coords so we can safely index body_grid (wall_hit already flagged)
        safe_r = np.clip(new_r, 0, self.gs - 1)
        safe_c = np.clip(new_c, 0, self.gs - 1)
        self_hit = self.body_grid[self._arange, safe_r, safe_c] > 0

        # terminated = true collision death (wall or self); excludes timeout truncation
        terminated = wall_hit | self_hit
        died = terminated.copy()

        # Decay body timers: contiguous subtract + clip is faster than masked scatter
        self.body_grid -= 1
        np.maximum(self.body_grid, 0, out=self.body_grid)

        ate_food = (~died) & (self.food_r >= 0) & (new_r == self.food_r) & (new_c == self.food_c)

        live = ~died
        self.head_r = np.where(live, new_r, self.head_r)
        self.head_c = np.where(live, new_c, self.head_c)

        self.length[ate_food] += 1
        self.score[ate_food] += 1

        # New head gets lifetime = current length (will count down to 0 as the tail passes)
        self.body_grid[self._arange[live], self.head_r[live], self.head_c[live]] = (
            self.length[live]
        )

        self.steps_alive[live] += 1

        timed_out = live & (self.steps_alive >= self.max_steps)
        died = died | timed_out

        rewards = np.full(self.n, self.cfg.step_penalty, dtype=np.float32)
        rewards[ate_food] = self.cfg.food_reward
        rewards[terminated] = self.cfg.death_penalty

        dones = died.copy()

        # Snapshot before auto-reset so callers see terminal episode stats
        infos: dict[str, np.ndarray] = {
            "length": self.length.copy(),
            "steps_alive": self.steps_alive.copy(),
            "terminated": terminated.copy(),
        }

        ate_idx = np.where(ate_food)[0]
        if ate_idx.size > 0:
            self._place_food_batch(ate_idx)

        # Potential-based shaping: γ·Φ(s') - Φ(s) for live (non-terminated) envs.
        # Computed after food placement so Φ(s') reflects the new food position.
        if self.cfg.distance_shaping:
            new_potential = self._potentials()
            shaping = self.cfg.distance_shaping * (
                self.cfg.gamma * new_potential - self._prev_potential
            )
            rewards[live & ~timed_out] += shaping[live & ~timed_out]
            self._prev_potential = new_potential

        dead_idx = np.where(died)[0]
        if dead_idx.size > 0:
            self._reset_envs(dead_idx)
            # Store the initial potential for newly reset episodes
            valid = self.food_r[dead_idx] >= 0
            self._prev_potential[dead_idx] = 0.0
            di_valid = dead_idx[valid]
            self._prev_potential[di_valid] = -(
                np.abs(self.head_r[di_valid] - self.food_r[di_valid])
                + np.abs(self.head_c[di_valid] - self.food_c[di_valid])
            ).astype(np.float32) / self.gs

        states = self._get_states()

        return states, rewards, dones, infos

    def _get_states(self) -> np.ndarray:
        """Build (N, 4, gs, gs) float32 state tensor from internal grids."""
        states = np.zeros(
            (self.n, self.cfg.num_channels, self.gs, self.gs), dtype=np.float32
        )

        states[self._arange, 0, self.head_r, self.head_c] = 1.0

        # Body gradient: higher countdown (closer to head) maps to higher values.
        # This lets the model distinguish the neck from the tail.
        body_mask = self.body_grid > 0
        head_mask = np.zeros_like(body_mask)
        head_mask[self._arange, self.head_r, self.head_c] = True
        body_only = body_mask & ~head_mask

        lengths_3d = self.length[:, None, None].astype(np.float32)
        timer_vals = self.body_grid.astype(np.float32)
        gradient = np.where(lengths_3d > 1, 0.5 + 0.5 * timer_vals / lengths_3d, 1.0)
        states[:, 1] = np.where(body_only, gradient, 0.0)

        valid_food = self.food_r >= 0
        if valid_food.any():
            idx = self._arange[valid_food]
            states[idx, 2, self.food_r[valid_food], self.food_c[valid_food]] = 1.0
        states[self._arange, 3, self.head_r, self.head_c] = _DIR_VAL[self.direction]

        return states

    def _reset_envs(self, idx: np.ndarray) -> None:
        """Reset a subset of envs by index (called automatically on death)."""
        mid = self.gs // 2
        self.head_r[idx] = mid
        self.head_c[idx] = mid
        self.direction[idx] = RIGHT
        self.body_grid[idx] = 0
        self.body_grid[idx, mid, mid] = 1
        self.length[idx] = 1
        self.score[idx] = 0
        self.steps_alive[idx] = 0
        self.alive[idx] = True
        self._place_food_batch(idx)

    def _place_food_batch(self, idx: np.ndarray) -> None:
        """Place food on a random empty cell for each env in *idx*.

        Uses rejection sampling: draw a random cell, accept if unoccupied.
        For typical snake lengths (≪ grid area) this converges in 1–2 rounds.
        """
        remaining = idx
        for _ in range(100):
            if remaining.size == 0:
                return
            flat = np.random.randint(0, self.gs * self.gs, size=remaining.size)
            r, c = flat // self.gs, flat % self.gs
            free = self.body_grid[remaining, r, c] == 0

            accepted = remaining[free]
            self.food_r[accepted] = r[free]
            self.food_c[accepted] = c[free]
            remaining = remaining[~free]

        # Remaining envs have near-full grids — no valid food position
        self.food_r[remaining] = -1
        self.food_c[remaining] = -1
