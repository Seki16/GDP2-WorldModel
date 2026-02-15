from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
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
    grid_size: int = 10
    max_steps: int = 64
    obs_size: int = 64
    wall_prob: float = 0.20
    seed: Optional[int] = None


class MazeEnv(gym.Env):
    """
    Minimal Maze environment.
    Observation: (64, 64, 3) RGB uint8, range 0-255
    Action space: Discrete(4) -> 0 up, 1 down, 2 left, 3 right
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
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

        self._build_grid()

    def _build_grid(self):
        n = self.config.grid_size
        grid = np.zeros((n, n), dtype=np.uint8)

        mask = self.rng.random((n, n)) < self.config.wall_prob
        grid[mask] = 1

        start = (0, 0)
        goal = (n - 1, n - 1)
        grid[start] = 0
        grid[goal] = 0

        self.grid = grid
        self.agent_pos = [start[0], start[1]]
        self.goal_pos = [goal[0], goal[1]]

    def _get_obs(self) -> np.ndarray:
        n = self.config.grid_size
        img = np.ones((n, n, 3), dtype=np.uint8) * 255  # free = white
        img[self.grid == 1] = np.array([0, 0, 0], dtype=np.uint8)  # wall = black

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        img[ax, ay] = np.array([255, 0, 0], dtype=np.uint8)  # agent = red
        img[gx, gy] = np.array([0, 255, 0], dtype=np.uint8)  # goal = green

        obs_size = self.config.obs_size
        scale = max(1, obs_size // n)
        up = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
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
        x, y = self.agent_pos

        if action == 0:
            nx, ny = x - 1, y
        elif action == 1:
            nx, ny = x + 1, y
        elif action == 2:
            nx, ny = x, y - 1
        elif action == 3:
            nx, ny = x, y + 1
        else:
            nx, ny = x, y

        n = self.config.grid_size
        nx = int(np.clip(nx, 0, n - 1))
        ny = int(np.clip(ny, 0, n - 1))

        if self.grid[nx, ny] == 0:
            self.agent_pos = [nx, ny]

        terminated = (self.agent_pos[0] == self.goal_pos[0] and self.agent_pos[1] == self.goal_pos[1])
        truncated = (self.steps >= self.config.max_steps)

        reward = 1.0 if terminated else -0.01
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
