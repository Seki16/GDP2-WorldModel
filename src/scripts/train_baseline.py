from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.env.maze_env import MazeEnv, MazeConfig
from src.models.dqn import DQNConfig, PixelDQN


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    """
    Convert raw RGB observation to channel-first float tensor.

    Args:
        obs: (H, W, 3) uint8 RGB

    Returns:
        torch.Tensor: (3, H, W) float32 in [0, 1]
    """
    x = torch.from_numpy(obs).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).contiguous()
    return x


def reset_fixed(env: MazeEnv, fixed_grid: np.ndarray) -> np.ndarray:
    """
    Reset the agent to the start position without regenerating the maze.
    This ensures every episode uses the exact same layout.
    """
    env.grid = fixed_grid.copy()
    env.agent_pos = [0, 0]
    env.steps = 0
    return env._get_obs()


@dataclass
class Transition:
    s: torch.Tensor
    a: int
    r: float
    s2: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = torch.stack([b.s for b in batch], dim=0)
        a = torch.tensor([b.a for b in batch], dtype=torch.int64)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32)
        s2 = torch.stack([b.s2 for b in batch], dim=0)
        d = torch.tensor([b.done for b in batch], dtype=torch.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


# -----------------------------
# Training
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train pixel DQN baseline on MazeEnv")

    # Global
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)

    # Environment
    parser.add_argument("--maze_seed", type=int, default=0,
                        help="Fixed maze layout seed for reproducible baseline training")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--wall_prob", type=float, default=0.20)

    # DQN optimization
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--learn_starts", type=int, default=1_000)
    parser.add_argument("--train_every", type=int, default=4)
    parser.add_argument("--target_update", type=int, default=1_000)

    # Exploration
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_steps", type=int, default=150_000)

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--conv1_channels", type=int, default=32)
    parser.add_argument("--conv2_channels", type=int, default=64)
    parser.add_argument("--conv3_channels", type=int, default=64)

    # Logging / output
    parser.add_argument("--log_every", type=int, default=10_000)
    parser.add_argument("--out", type=str, default="evaluation/dqn_baseline_curve.csv")
    parser.add_argument("--save_path", type=str, default="checkpoints/dqn_baseline.pt")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Device    : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU       : {torch.cuda.get_device_name(0)}")

    cfg = MazeConfig(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        obs_size=64,
        wall_prob=args.wall_prob,
        seed=args.maze_seed,
    )
    env = MazeEnv(cfg)

    # Build env once, then lock layout for all episodes
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    fixed_grid = env.grid.copy()

    obs_size = obs.shape[0]
    n_actions = int(env.action_space.n)

    dqn_config = DQNConfig(
        obs_type="pixel",
        obs_size=obs_size,
        n_actions=n_actions,
        hidden_dim=args.hidden_dim,
        conv1_channels=args.conv1_channels,
        conv2_channels=args.conv2_channels,
        conv3_channels=args.conv3_channels,
    )

    q = PixelDQN(dqn_config).to(device)
    q_tgt = PixelDQN(dqn_config).to(device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    rb = ReplayBuffer(args.buffer_size)

    # Metrics output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("step,episode,episode_return,episode_len,epsilon,success\n", encoding="utf-8")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    episode_return = 0.0
    episode_len = 0
    episode_idx = 0
    success_count = 0
    recent_returns = []

    # Start from fixed maze
    obs = reset_fixed(env, fixed_grid)
    s = preprocess_obs(obs)

    def epsilon_by_step(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / max(1, args.eps_decay_steps)
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    print("\n" + "─" * 60)
    print("  Training Pixel DQN Baseline on fixed maze")
    print("─" * 60)

    for t in range(1, args.steps + 1):
        eps = float(epsilon_by_step(t))

        if random.random() < eps:
            a = int(env.action_space.sample())
        else:
            with torch.no_grad():
                qs = q(s.unsqueeze(0).to(device))
                a = int(torch.argmax(qs, dim=1).item())

        step_out = env.step(a)
        if len(step_out) == 5:
            obs2, r, terminated, truncated, _info = step_out
            done = bool(terminated or truncated)
            success = bool(terminated)
        else:
            obs2, r, done, _info = step_out
            success = bool(r > 0.5)

        s2 = preprocess_obs(obs2)
        rb.push(Transition(s=s.cpu(), a=a, r=float(r), s2=s2.cpu(), done=done))

        episode_return += float(r)
        episode_len += 1
        s = s2

        if done:
            recent_returns.append(episode_return)
            if len(recent_returns) > 100:
                recent_returns.pop(0)
            if success:
                success_count += 1

            with out_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"{t},{episode_idx},{episode_return:.4f},{episode_len},{eps:.4f},{int(success)}\n"
                )

            episode_idx += 1
            episode_return = 0.0
            episode_len = 0

            obs = reset_fixed(env, fixed_grid)
            s = preprocess_obs(obs)

        if t >= args.learn_starts and (t % args.train_every == 0) and len(rb) >= args.batch_size:
            s_b, a_b, r_b, s2_b, d_b = rb.sample(args.batch_size)
            s_b = s_b.to(device)
            a_b = a_b.to(device)
            r_b = r_b.to(device)
            s2_b = s2_b.to(device)
            d_b = d_b.to(device)

            q_sa = q(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                q_next = q_tgt(s2_b).max(dim=1).values
                target = r_b + (1.0 - d_b) * args.gamma * q_next

            loss = nn.functional.smooth_l1_loss(q_sa, target)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

        if t % args.target_update == 0:
            q_tgt.load_state_dict(q.state_dict())

        if t % args.log_every == 0:
            avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            success_rate = 100.0 * success_count / max(1, episode_idx)
            print(
                f"[DQN] step={t:7d}  eps={eps:.3f}  "
                f"episodes={episode_idx:5d}  buffer={len(rb):6d}  "
                f"avg_ret={avg_return:+.3f}  success={success_rate:.1f}%"
            )

    torch.save(
        {
            "model_state": q.state_dict(),
            "dqn_config": vars(dqn_config),
            "obs_size": obs_size,
            "maze_seed": args.maze_seed,
            "args": vars(args),
        },
        save_path,
    )

    final_success_rate = 100.0 * success_count / max(1, episode_idx)
    print("\n[DQN] DONE")
    print(f"[DQN] Checkpoint saved to: {save_path}")
    print(f"[DQN] Curve saved to     : {out_path}")
    print(f"[DQN] Episodes          : {episode_idx}")
    print(f"[DQN] Success rate      : {final_success_rate:.1f}%")
    if recent_returns:
        print(f"[DQN] Avg return (last 100): {np.mean(recent_returns):+.4f}")


if __name__ == "__main__":
    main()