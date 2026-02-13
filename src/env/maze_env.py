from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    GYMNASIUM = False


@dataclass
class MazeConfig:
    grid_size: int = 10          # 10x10 grid
    max_steps: int = 64
    obs_size: int = 64           # output image: 64x64
    wall_prob: float = 0.20      # random walls density (kept simple)
    seed: Optional[int] = None


class MazeEnv(gym.Env):
    """
    Minimal Maze environment that returns RGB images of shape (64, 64, 3), dtype uint8.

    Actions (Discrete(4)):
      0: Up, 1: Down, 2: Left, 3: Right

    Observation:
      RGB image (64, 64, 3), uint8, values 0-255
    """

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

        self.grid = None  # 0 free, 1 wall
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

        self._build_grid()

    def _build_grid(self):
        n = self.config.grid_size
        # Start with empty grid
        grid = np.zeros((n, n), dtype=np.uint8)

        # Add random walls
        mask = self.rng.random((n, n)) < self.config.wall_prob
        grid[mask] = 1

        # Ensure borders are walls? (optional) — keep it open for easier learning
        # grid[0, :] = 1; grid[-1, :] = 1; grid[:, 0] = 1; grid[:, -1] = 1

        # Define start and goal, and force them free
        start = (0, 0)
        goal = (n - 1, n - 1)
        grid[start] = 0
        grid[goal] = 0

        self.grid = grid
        self.agent_pos = list(start)
        self.goal_pos = list(goal)

        # If the grid accidentally blocks everything, it’s okay for MVP.
        # (We can later add BFS-reachable generation if needed.)

    def _get_obs(self) -> np.ndarray:
        """
        Render grid -> 64x64 RGB uint8.
        Colors:
          walls: black
          free: white
          agent: red
          goal: green
        """
        n = self.config.grid_size
        img = np.ones((n, n, 3), dtype=np.uint8) * 255  # free = white
        img[self.grid == 1] = np.array([0, 0, 0], dtype=np.uint8)  # wall = black

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        img[ax, ay] = np.array([255, 0, 0], dtype=np.uint8)  # agent = red
        img[gx, gy] = np.array([0, 255, 0], dtype=np.uint8)  # goal = green

        # Upscale to 64x64 using nearest-neighbor (no extra deps)
        obs_size = self.config.obs_size
        scale = obs_size // n
        # ensure exact size
        up = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        # If obs_size not divisible by n, pad/crop (for safety)
        up = up[:obs_size, :obs_size, :]
        if up.shape[0] < obs_size or up.shape[1] < obs_size:
            padded = np.zeros((obs_size, obs_size, 3), dtype=np.uint8)
            padded[:up.shape[0], :up.shape[1], :] = up
            up = padded

        return up.astype(np.uint8)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self._build_grid()

        obs = self._get_obs()
        info = {"pos": tuple(self.agent_pos), "goal": tuple(self.goal_pos)}
        return (obs, info) if GYMNASIUM else obs

    def step(self, action: int):
        self.steps += 1

        # current position
        x, y = self.agent_pos

        # move proposal
        if action == 0:      # up
            nx, ny = x - 1, y
        elif action == 1:    # down
            nx, ny = x + 1, y
        elif action == 2:    # left
            nx, ny = x, y - 1
        elif action == 3:    # right
            nx, ny = x, y + 1
        else:
            nx, ny = x, y

        # bounds check
        n = self.config.grid_size
        nx = int(np.clip(nx, 0, n - 1))
        ny = int(np.clip(ny, 0, n - 1))

        # wall check
        if self.grid[nx, ny] == 0:
            self.agent_pos = [nx, ny]

        # reward shaping
        terminated = (self.agent_pos[0] == self.goal_pos[0] and self.agent_pos[1] == self.goal_pos[1])
        truncated = (self.steps >= self.config.max_steps)

        reward = 1.0 if terminated else -0.01  # small step penalty, goal reward

        obs = self._get_obs()
        info = {
            "pos": tuple(self.agent_pos),
            "goal": tuple(self.goal_pos),
            "goal_reached": bool(terminated),
            "steps": self.steps,
        }

        if GYMNASIUM:
            return obs, reward, terminated, truncated, info
        else:
            done = terminated or truncated
            return obs, reward, done, info
