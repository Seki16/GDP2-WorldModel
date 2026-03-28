"""
demo_dream_transfer.py — Visualise the Dream DQN deployed in the real MazeEnv
===============================================================================
This is the transfer experiment made visible: the Dream DQN was trained
entirely inside the World Model on 384-dim latents. Here we deploy it in
the real MazeEnv by encoding each RGB observation through DINOv2, then
feeding the latent to the Dream Q-network for action selection.

Produces a GIF so you can literally watch the transfer gap in action.

Usage:
    python -m src.scripts.demo_dream_transfer \
        --dream_checkpoint checkpoints/dqn_dream.pt \
        --gif_path evaluation/dream_transfer_demo.gif

    # Side-by-side comparison (run both):
    python -m src.scripts.demo_baseline_rollout  # pixel DQN → solves it
    python -m src.scripts.demo_dream_transfer     # dream DQN → fails
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from src.env.maze_env import MazeEnv, MazeConfig
from src.scripts.train_dream_dqn import DreamQNet
from src.models.encoder import DinoV2Encoder


# ─── Frame helpers (same as demo_baseline_rollout.py) ────────────────────────

def upscale_frame(frame: np.ndarray, scale: int = 4) -> np.ndarray:
    return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)


def add_border(frame: np.ndarray, border: int = 4,
               color: tuple = (255, 0, 0)) -> np.ndarray:
    """Red border to visually distinguish from the green-bordered pixel DQN demo."""
    h, w, c = frame.shape
    out = np.zeros((h + 2 * border, w + 2 * border, c), dtype=np.uint8)
    out[:, :] = color  # red border
    out[border:border + h, border:border + w] = frame
    return out


def reset_fixed(env: MazeEnv, fixed_grid: np.ndarray) -> np.ndarray:
    env.grid = fixed_grid.copy()
    env.agent_pos = [0, 0]
    env.steps = 0
    return env._get_obs()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise Dream DQN transfer in real MazeEnv"
    )
    parser.add_argument("--dream_checkpoint", type=str,
                        default="checkpoints/dqn_dream.pt")
    parser.add_argument("--maze_seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--obs_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--wall_prob", type=float, default=0.20)
    parser.add_argument("--gif_path", type=str,
                        default="evaluation/dream_transfer_demo.gif")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load Dream DQN ────────────────────────────────────────────────────────
    # Dream DQN takes 384-dim DINOv2 latents → Q-values for 4 actions.
    # It was trained entirely inside WorldModelEnv, never saw a real pixel.
    ckpt_path = Path(args.dream_checkpoint)
    print(f"[INFO] Loading Dream DQN from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    latent_dim = ckpt.get("latent_dim", 384)
    action_dim = ckpt.get("action_dim", 4)
    dream_dqn = DreamQNet(latent_dim=latent_dim, action_dim=action_dim).to(device)
    dream_dqn.load_state_dict(ckpt["model_state"])
    dream_dqn.eval()

    steps_trained = ckpt.get("steps_trained", "?")
    dream_success = ckpt.get("final_success_pct", 0.0)
    print(f"[INFO] Dream DQN: {steps_trained} training steps, "
          f"{dream_success:.1f}% in-dream success")

    # ── Load DINOv2 encoder ───────────────────────────────────────────────────
    # This is the transfer bridge: real pixels → DINOv2 → 384-dim latent →
    # Dream DQN selects action. If latents from real obs match what the WM
    # produced during training, transfer works. If not, it fails.
    print("[INFO] Loading DINOv2 encoder...")
    encoder = DinoV2Encoder(device=device)

    # ── Set up maze (same seed as all evaluations) ────────────────────────────
    env = MazeEnv(
        MazeConfig(
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            obs_size=args.obs_size,
            wall_prob=args.wall_prob,
            seed=args.maze_seed,
        )
    )

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    fixed_grid = env.grid.copy()
    obs = reset_fixed(env, fixed_grid)

    # ── Run episode ───────────────────────────────────────────────────────────
    frames = []
    total_reward = 0.0
    success = False
    step_idx = 0
    actions_taken = []

    while True:
        # Capture frame
        frame = upscale_frame(obs, scale=args.scale)
        frame = add_border(frame, border=4, color=(255, 0, 0))  # red = dream
        frames.append(frame)

        # Encode real observation through DINOv2 → 384-dim latent
        obs_tensor = torch.from_numpy(obs).to(device)
        z = encoder.encode(obs_tensor)          # (384,)
        z_t = z.float().unsqueeze(0).to(device)  # (1, 384)

        # Dream DQN selects action from latent (greedy, no epsilon)
        with torch.no_grad():
            q_values = dream_dqn(z_t)
            action = int(torch.argmax(q_values, dim=1).item())

        actions_taken.append(action)

        # Step in REAL environment
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
            final_frame = add_border(final_frame, border=4, color=(255, 0, 0))
            frames.append(final_frame)
            break

    # ── Save GIF ──────────────────────────────────────────────────────────────
    gif_path = Path(args.gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=args.fps)

    # ── Summary ───────────────────────────────────────────────────────────────
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
    action_counts = {name: 0 for name in action_names.values()}
    for a in actions_taken:
        action_counts[action_names[a]] += 1

    print(f"\n{'='*60}")
    print(f"  DREAM DQN TRANSFER DEMO")
    print(f"{'='*60}")
    print(f"  GIF saved to : {gif_path}")
    print(f"  Steps        : {step_idx}")
    print(f"  Success      : {success}")
    print(f"  Return       : {total_reward:.4f}")
    print(f"  Actions      : {action_counts}")
    print(f"{'='*60}")

    # ── Diagnosis: check if agent is stuck / looping ──────────────────────────
    if step_idx >= args.max_steps - 1 and not success:
        print("\n  [DIAGNOSIS] Agent hit max_steps without reaching the goal.")
        # Check for action repetition (sign of a collapsed policy)
        most_common = max(action_counts, key=action_counts.get)
        pct = action_counts[most_common] / step_idx * 100
        if pct > 60:
            print(f"  [DIAGNOSIS] Dominant action: '{most_common}' "
                  f"({pct:.0f}% of steps) — policy likely collapsed to "
                  f"a single action. This is consistent with training on "
                  f"corrupted WM reward signal.")
        else:
            print(f"  [DIAGNOSIS] Actions are somewhat distributed — agent "
                  f"is moving but not navigating toward the goal.")


if __name__ == "__main__":
    main()