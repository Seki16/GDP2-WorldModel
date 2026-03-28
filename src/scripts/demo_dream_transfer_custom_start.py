"""
demo_dream_transfer_custom_start.py — Visualise Dream DQN from any position
=============================================================================
Same as demo_dream_transfer.py but starts from a custom position instead
of always [0,0]. Generates side-by-side GIFs for successful and failing
transfer positions.

Usage:
    # The one that works:
    python -m src.scripts.demo_dream_transfer_custom_start \
        --dream_checkpoint checkpoints/dqn_near_goal_latent_reward/dqn_dream.pt \
        --start_row 8 --start_col 9 \
        --gif_path evaluation/dream_transfer_8_9.gif

    # The one that fails (same distance, wrong action):
    python -m src.scripts.demo_dream_transfer_custom_start \
        --dream_checkpoint checkpoints/dqn_near_goal_latent_reward/dqn_dream.pt \
        --start_row 9 --start_col 8 \
        --gif_path evaluation/dream_transfer_9_8.gif

    # Default [0,0]:
    python -m src.scripts.demo_dream_transfer_custom_start \
        --dream_checkpoint checkpoints/dqn_near_goal_latent_reward/dqn_dream.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from src.env.maze_env import MazeEnv, MazeConfig
from src.models.encoder import DinoV2Encoder
from src.scripts.train_dream_dqn import DreamQNet


def upscale_frame(frame: np.ndarray, scale: int = 4) -> np.ndarray:
    return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)


def add_border(frame: np.ndarray, border: int = 4,
               color: tuple = (255, 0, 0)) -> np.ndarray:
    h, w, c = frame.shape
    out = np.zeros((h + 2 * border, w + 2 * border, c), dtype=np.uint8)
    out[:, :] = color
    out[border:border + h, border:border + w] = frame
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Visualise Dream DQN transfer from custom start position"
    )
    parser.add_argument("--dream_checkpoint", type=str,
                        default="checkpoints/dqn_near_goal_latent_reward/dqn_dream.pt")
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--start_col", type=int, default=0)
    parser.add_argument("--maze_seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--obs_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--wall_prob", type=float, default=0.20)
    parser.add_argument("--gif_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    if args.gif_path is None:
        args.gif_path = (f"evaluation/dream_transfer_"
                         f"{args.start_row}_{args.start_col}.gif")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load Dream DQN ────────────────────────────────────────────────────────
    ckpt_path = Path(args.dream_checkpoint)
    print(f"[INFO] Loading Dream DQN from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    latent_dim = ckpt.get("latent_dim", 384)
    action_dim = ckpt.get("action_dim", 4)
    dream_dqn = DreamQNet(latent_dim=latent_dim, action_dim=action_dim).to(device)
    dream_dqn.load_state_dict(ckpt["model_state"])
    dream_dqn.eval()
    print(f"[INFO] In-dream success: {ckpt.get('final_success_pct', 0):.1f}%")

    # ── Load DINOv2 ───────────────────────────────────────────────────────────
    print("[INFO] Loading DINOv2 encoder...")
    encoder = DinoV2Encoder(device=device)

    # ── Set up maze ───────────────────────────────────────────────────────────
    env = MazeEnv(MazeConfig(
        grid_size=args.grid_size, max_steps=args.max_steps,
        obs_size=args.obs_size, wall_prob=args.wall_prob,
        seed=args.maze_seed,
    ))
    env.reset()
    fixed_grid = env.grid.copy()

    # Set custom start position
    env.grid = fixed_grid.copy()
    env.agent_pos = [args.start_row, args.start_col]
    env.steps = 0
    obs = env._get_obs()

    # ── Run episode ───────────────────────────────────────────────────────────
    frames = []
    total_reward = 0.0
    success = False
    step_idx = 0
    actions_taken = []
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}

    while True:
        # Green border if success so far, red otherwise
        border_color = (0, 200, 0) if success else (255, 0, 0)
        frame = upscale_frame(obs, scale=args.scale)
        frame = add_border(frame, border=4, color=border_color)
        frames.append(frame)

        # Encode and act
        obs_tensor = torch.from_numpy(obs).to(device)
        z = encoder.encode(obs_tensor)
        z_t = z.float().unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = dream_dqn(z_t)
            action = int(torch.argmax(q_values, dim=1).item())

        actions_taken.append(action)

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
            border_color = (0, 200, 0) if success else (255, 0, 0)
            final_frame = upscale_frame(obs, scale=args.scale)
            final_frame = add_border(final_frame, border=4, color=border_color)
            # Hold final frame longer
            for _ in range(3):
                frames.append(final_frame)
            break

    # ── Save GIF ──────────────────────────────────────────────────────────────
    gif_path = Path(args.gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=args.fps)

    # ── Summary ───────────────────────────────────────────────────────────────
    action_counts = {name: 0 for name in action_names.values()}
    for a in actions_taken:
        action_counts[action_names[a]] += 1

    status = "SUCCESS ✓" if success else "FAIL ✗"
    print(f"\n{'='*60}")
    print(f"  DREAM DQN TRANSFER — {status}")
    print(f"{'='*60}")
    print(f"  Start      : ({args.start_row}, {args.start_col})")
    print(f"  Goal       : ({args.grid_size-1}, {args.grid_size-1})")
    print(f"  GIF        : {gif_path}")
    print(f"  Steps      : {step_idx}")
    print(f"  Return     : {total_reward:+.4f}")
    print(f"  Actions    : {action_counts}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()