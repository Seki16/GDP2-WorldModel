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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 -> float32 tensor (3,H,W) in [0,1]"""
    return torch.from_numpy(obs).float().permute(2, 0, 1) / 255.0


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
        s  = torch.stack([b.s  for b in batch])
        a  = torch.tensor([b.a  for b in batch], dtype=torch.int64)
        r  = torch.tensor([b.r  for b in batch], dtype=torch.float32)
        s2 = torch.stack([b.s2 for b in batch])
        d  = torch.tensor([b.done for b in batch], dtype=torch.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


class TinyCNN(nn.Module):
    def __init__(self, n_actions: int, obs_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.net(torch.zeros(1, 3, obs_size, obs_size)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def reset_fixed(env: MazeEnv, fixed_grid: np.ndarray) -> np.ndarray:
    """
    Reset agent to [0,0] without rebuilding the maze.
    Ensures every episode uses the exact same layout.
    """
    env.grid = fixed_grid.copy()
    env.agent_pos = [0, 0]
    env.steps = 0
    return env._get_obs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",           type=int,   default=200_000)
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--maze_seed",       type=int,   default=0,
                        help="Seed for fixed maze layout (same maze every episode)")
    parser.add_argument("--grid_size",       type=int,   default=10)
    parser.add_argument("--max_steps",       type=int,   default=64)
    parser.add_argument("--wall_prob",       type=float, default=0.20)
    parser.add_argument("--gamma",           type=float, default=0.99)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--buffer_size",     type=int,   default=50_000)
    parser.add_argument("--learn_starts",    type=int,   default=1_000)
    parser.add_argument("--train_every",     type=int,   default=4)
    parser.add_argument("--target_update",   type=int,   default=1_000)
    parser.add_argument("--eps_start",       type=float, default=1.0)
    parser.add_argument("--eps_end",         type=float, default=0.05)
    parser.add_argument("--eps_decay_steps", type=int,   default=100_000)
    parser.add_argument("--log_every",       type=int,   default=2_000)
    parser.add_argument("--out",             type=str,   default="evaluation/baseline_metrics.csv")
    parser.add_argument("--save_dir",        type=str,   default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device    : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU       : {torch.cuda.get_device_name(0)}")

    # ── Build env and lock maze layout ───────────────────────────────────────
    cfg = MazeConfig(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        obs_size=64,
        wall_prob=args.wall_prob,
        seed=args.maze_seed,
    )
    env = MazeEnv(cfg)
    obs, _ = env.reset()
    fixed_grid = env.grid.copy()   # lock this layout for all episodes

    obs_size  = obs.shape[0]
    n_actions = int(env.action_space.n)

    print(f"[INFO] Maze seed : {args.maze_seed} — same layout every episode")
    print(f"[INFO] Obs size  : {obs_size}x{obs_size}")
    print(f"[INFO] Steps     : {args.steps:,}")

    q     = TinyCNN(n_actions, obs_size).to(device)
    q_tgt = TinyCNN(n_actions, obs_size).to(device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    rb  = ReplayBuffer(args.buffer_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("step,episode,episode_return,episode_len,epsilon,success\n")

    episode_return = 0.0
    episode_len    = 0
    episode_idx    = 0
    success_count  = 0
    recent_returns = []

    s = preprocess_obs(obs)

    def epsilon_by_step(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        return args.eps_start + (t / args.eps_decay_steps) * (args.eps_end - args.eps_start)

    print(f"\n{'─'*60}")
    print(f"  Training DQN on fixed maze")
    print(f"{'─'*60}")

    for t in range(1, args.steps + 1):
        eps = epsilon_by_step(t)

        if random.random() < eps:
            a = int(env.action_space.sample())
        else:
            with torch.no_grad():
                a = int(torch.argmax(q(s.unsqueeze(0).to(device)), dim=1).item())

        step_out = env.step(a)
        if len(step_out) == 5:
            obs2, r, terminated, truncated, _ = step_out
            done    = bool(terminated or truncated)
            success = bool(terminated)
        else:
            obs2, r, done, _ = step_out
            success = r > 0.5

        s2 = preprocess_obs(obs2)
        rb.push(Transition(s=s.cpu(), a=a, r=float(r), s2=s2.cpu(), done=done))

        episode_return += float(r)
        episode_len    += 1
        s = s2

        if done:
            recent_returns.append(episode_return)
            if len(recent_returns) > 100:
                recent_returns.pop(0)
            if success:
                success_count += 1

            out_path.open("a").write(
                f"{t},{episode_idx},{episode_return:.4f},"
                f"{episode_len},{eps:.4f},{int(success)}\n"
            )
            episode_idx   += 1
            episode_return = 0.0
            episode_len    = 0

            # Fixed-maze reset — same layout, agent back to start
            obs = reset_fixed(env, fixed_grid)
            s   = preprocess_obs(obs)

        # Learning
        if t >= args.learn_starts and t % args.train_every == 0 and len(rb) >= args.batch_size:
            s_b, a_b, r_b, s2_b, d_b = rb.sample(args.batch_size)
            s_b, a_b, r_b, s2_b, d_b = (
                s_b.to(device), a_b.to(device), r_b.to(device),
                s2_b.to(device), d_b.to(device)
            )
            q_sa = q(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                target = r_b + (1.0 - d_b) * args.gamma * q_tgt(s2_b).max(1).values
            loss = nn.functional.smooth_l1_loss(q_sa, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

        if t % args.target_update == 0:
            q_tgt.load_state_dict(q.state_dict())

        if t % args.log_every == 0:
            avg = np.mean(recent_returns) if recent_returns else 0.0
            sr  = 100 * success_count / max(1, episode_idx)
            print(f"[DQN] step={t:7d}  eps={eps:.3f}  "
                  f"ep={episode_idx:5d}  avg_ret={avg:+.3f}  "
                  f"success={sr:.1f}%")

    # Save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "dqn_baseline.pt"
    torch.save({
        "model_state": q.state_dict(),
        "obs_size":    obs_size,
        "maze_seed":   args.maze_seed,
        "args":        vars(args),
    }, path)

    sr = 100 * success_count / max(1, episode_idx)
    print(f"\n[DQN] DONE")
    print(f"[DQN] Weights → {path}")
    print(f"[DQN] Metrics → {out_path}")
    print(f"[DQN] Episodes: {episode_idx}  |  Success rate: {sr:.1f}%")
    if recent_returns:
        print(f"[DQN] Avg return (last 100): {np.mean(recent_returns):+.4f}")


if __name__ == "__main__":
    main()