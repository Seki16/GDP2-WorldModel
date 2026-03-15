from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from src.env.maze_env import MazeEnv, MazeConfig
from src.models.dqn import DQNConfig, PixelDQN


def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(obs).float().permute(2, 0, 1) / 255.0


def reset_fixed(env: MazeEnv, fixed_grid: np.ndarray) -> np.ndarray:
    env.grid = fixed_grid.copy()
    env.agent_pos = [0, 0]
    env.steps = 0
    return env._get_obs()


def upscale_frame(frame: np.ndarray, scale: int = 4) -> np.ndarray:
    return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)


def add_border(frame: np.ndarray, border: int = 4) -> np.ndarray:
    h, w, c = frame.shape
    out = np.ones((h + 2 * border, w + 2 * border, c), dtype=np.uint8) * 255
    out[border:border + h, border:border + w] = frame
    return out


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[PixelDQN, int]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    obs_size = ckpt.get("obs_size", 64)
    saved_cfg = ckpt.get("dqn_config", {})

    dqn_config = DQNConfig(
        obs_type="pixel",
        obs_size=obs_size,
        n_actions=saved_cfg.get("n_actions", 4),
        hidden_dim=saved_cfg.get("hidden_dim", 128),
        conv1_channels=saved_cfg.get("conv1_channels", 32),
        conv2_channels=saved_cfg.get("conv2_channels", 64),
        conv3_channels=saved_cfg.get("conv3_channels", 64),
    )

    model = PixelDQN(dqn_config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, obs_size


def main():
    parser = argparse.ArgumentParser(description="Create a demo GIF of the trained pixel DQN solving the maze")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dqn_baseline.pt")
    parser.add_argument("--maze_seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--wall_prob", type=float, default=0.20)
    parser.add_argument("--gif_path", type=str, default="evaluation/dqn_baseline_demo.gif")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    model, obs_size = build_model_from_checkpoint(checkpoint_path, device)

    env = MazeEnv(
        MazeConfig(
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            obs_size=obs_size,
            wall_prob=args.wall_prob,
            seed=args.maze_seed,
        )
    )

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    fixed_grid = env.grid.copy()
    obs = reset_fixed(env, fixed_grid)

    frames = []
    total_reward = 0.0
    success = False
    step_idx = 0

    while True:
        frame = upscale_frame(obs, scale=args.scale)
        frame = add_border(frame, border=4)
        frames.append(frame)

        s = preprocess_obs(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = int(torch.argmax(model(s), dim=1).item())

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _info = step_out
            done = bool(terminated or truncated)
            success = bool(terminated)
        else:
            obs, reward, done, _info = step_out
            success = bool(reward > 0.5)

        total_reward += float(reward)
        step_idx += 1

        if done:
            final_frame = upscale_frame(obs, scale=args.scale)
            final_frame = add_border(final_frame, border=4)
            frames.append(final_frame)
            break

    gif_path = Path(args.gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=args.fps)

    print(f"[DEMO] Saved GIF to: {gif_path}")
    print(f"[DEMO] Steps      : {step_idx}")
    print(f"[DEMO] Success    : {success}")
    print(f"[DEMO] Return     : {total_reward:.4f}")


if __name__ == "__main__":
    main()