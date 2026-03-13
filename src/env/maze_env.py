"""
src/env/maze_env.py
====================
Member A — Task A.1

Custom Gymnasium maze environment. Generates random grid mazes and
provides (64, 64, 3) RGB image observations at each step.

Interface Contract (GDP Plan §2.3):
    env.reset()     → obs (64, 64, 3) uint8 RGB
    env.step(action)→ obs (64, 64, 3) uint8 RGB, reward float, terminated bool, truncated bool, info dict
    action_space    → Discrete(4)  — 0: Up, 1: Down, 2: Left, 3: Right
    reward          → +1.0 on reaching goal, -0.01 per step otherwise
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional


@dataclass
class MazeConfig:
    grid_size: int = 10
    max_steps: int = 64
    obs_size: int = 64
    wall_prob: float = 0.20
    seed: Optional[int] = None


class MazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[MazeConfig] = None):
        super().__init__()
        self.config = config or MazeConfig()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.config.obs_size, self.config.obs_size, 3),
            dtype=np.uint8,
        )

        self.rng = np.random.default_rng(self.config.seed)
        self.steps = 0
        self._build_grid()

    def _build_grid(self):
        n = self.config.grid_size
        self.grid = np.zeros((n, n), dtype=np.uint8)

        mask = self.rng.random((n, n)) < self.config.wall_prob
        self.grid[mask] = 1

        self.grid[0, 0] = 0
        self.grid[n - 1, n - 1] = 0

        self.agent_pos = [0, 0]
        self.goal_pos = [n - 1, n - 1]

    def _get_obs(self) -> np.ndarray:
        """
        Render the maze as an exact (obs_size, obs_size, 3) uint8 RGB image.
        """
        n = self.config.grid_size
        target = self.config.obs_size

        img = np.ones((n, n, 3), dtype=np.uint8) * 255
        img[self.grid == 1] = [0, 0, 0]
        img[self.agent_pos[0], self.agent_pos[1]] = [255, 0, 0]
        img[self.goal_pos[0], self.goal_pos[1]] = [0, 255, 0]

        # Scale up with nearest-neighbour style repetition
        scale = max(1, target // n)
        up = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        # If scaling undershoots (e.g. 10*6 = 60 < 64), pad with white pixels
        h, w, _ = up.shape
        if h < target or w < target:
            padded = np.ones((target, target, 3), dtype=np.uint8) * 255
            padded[:h, :w, :] = up
            up = padded

        # Final safety crop to exact target size
        return up[:target, :target, :].astype(np.uint8)

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self._build_grid()
        return self._get_obs(), {"pos": tuple(self.agent_pos)}

    def step(self, action: int):
        self.steps += 1
        x, y = self.agent_pos

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        nx, ny = x + dx, y + dy
        n = self.config.grid_size

        nx = int(np.clip(nx, 0, n - 1))
        ny = int(np.clip(ny, 0, n - 1))

        if self.grid[nx, ny] == 0:
            self.agent_pos = [nx, ny]

        terminated = self.agent_pos == self.goal_pos
        truncated = self.steps >= self.config.max_steps
        reward = 1.0 if terminated else -0.01

        return self._get_obs(), reward, terminated, truncated, {"pos": tuple(self.agent_pos)}

    def render(self):
        return self._get_obs()