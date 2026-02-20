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

Usage:
    from src.env.maze_env import MazeEnv, MazeConfig

    cfg = MazeConfig(grid_size=10, max_steps=64, obs_size=64, wall_prob=0.20, seed=0)
    env = MazeEnv(cfg)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional


@dataclass
class MazeConfig:
    grid_size: int          = 10     # Number of cells per side
    max_steps: int          = 64     # Episode truncation limit
    obs_size:  int          = 64     # Output image size (pixels per side)
    wall_prob: float        = 0.20   # Probability of any cell being a wall
    seed:      Optional[int] = None  # RNG seed for reproducible mazes


class MazeEnv(gym.Env):
    """
    A simple randomly-generated grid maze environment.

    Layout:
      - White cells  : free space
      - Black cells  : walls (agent cannot enter)
      - Red cell     : agent position (starts at [0, 0])
      - Green cell   : goal position  (bottom-right corner)

    The maze is regenerated on every reset() call (unless a fixed seed is used).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[MazeConfig] = None):
        super().__init__()
        self.config = config or MazeConfig()

        # Action space: 4 discrete actions — Up / Down / Left / Right
        self.action_space = spaces.Discrete(4)

        # Observation space: (obs_size, obs_size, 3) RGB uint8 image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.config.obs_size, self.config.obs_size, 3),
            dtype=np.uint8,
        )

        self.rng   = np.random.default_rng(self.config.seed)
        self.steps = 0
        self._build_grid()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_grid(self):
        """Generate a new random maze grid."""
        n = self.config.grid_size
        self.grid = np.zeros((n, n), dtype=np.uint8)

        # Randomly place walls
        mask = self.rng.random((n, n)) < self.config.wall_prob
        self.grid[mask] = 1

        # Start [0,0] and goal [n-1, n-1] are always free
        self.grid[0, 0]         = 0
        self.grid[n - 1, n - 1] = 0

        self.agent_pos = [0, 0]
        self.goal_pos  = [n - 1, n - 1]

    def _get_obs(self) -> np.ndarray:
        """Render the current maze state as a (obs_size, obs_size, 3) uint8 RGB image."""
        n   = self.config.grid_size
        img = np.ones((n, n, 3), dtype=np.uint8) * 255   # white background

        img[self.grid == 1]                               = [0,   0,   0  ]  # walls: black
        img[self.agent_pos[0], self.agent_pos[1]]         = [255, 0,   0  ]  # agent: red
        img[self.goal_pos[0],  self.goal_pos[1]]          = [0,   255, 0  ]  # goal:  green

        # Upscale grid to obs_size × obs_size
        scale = max(1, self.config.obs_size // n)
        up    = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        # Crop to exact obs_size (in case scale overshoots slightly)
        return up[: self.config.obs_size, : self.config.obs_size, :].astype(np.uint8)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options=None):
        """
        Reset the environment and return the initial observation.

        :param seed: If provided, re-seeds the RNG and generates a new maze layout.
        :returns: (obs, info) where obs is (obs_size, obs_size, 3) uint8.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self._build_grid()
        return self._get_obs(), {"pos": tuple(self.agent_pos)}

    def step(self, action: int):
        """
        Apply an action and return the next observation.

        Actions: 0=Up, 1=Down, 2=Left, 3=Right

        :returns: (obs, reward, terminated, truncated, info)
                  obs        — (obs_size, obs_size, 3) uint8
                  reward     — +1.0 if goal reached, -0.01 otherwise
                  terminated — True if agent reached the goal
                  truncated  — True if max_steps exceeded
                  info       — dict with 'pos' key
        """
        self.steps += 1
        x, y = self.agent_pos

        # Action → direction vector
        moves        = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        dx, dy       = moves[action]
        nx, ny       = x + dx, y + dy
        n            = self.config.grid_size

        # Clamp to grid bounds
        nx, ny = int(np.clip(nx, 0, n - 1)), int(np.clip(ny, 0, n - 1))

        # Move only if target cell is not a wall
        if self.grid[nx, ny] == 0:
            self.agent_pos = [nx, ny]

        terminated = (self.agent_pos == self.goal_pos)
        truncated  = (self.steps >= self.config.max_steps)
        reward     = 1.0 if terminated else -0.01

        return self._get_obs(), reward, terminated, truncated, {"pos": tuple(self.agent_pos)}

    def render(self):
        """Return the current observation as a numpy array (no display)."""
        return self._get_obs()