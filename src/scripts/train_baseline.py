from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.env.maze_env import MazeEnv, MazeConfig


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    """
    obs: (64,64,3) uint8 RGB -> float32 tensor (3,64,64) in [0,1]
    """
    x = torch.from_numpy(obs).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).contiguous()  # (3,64,64)
    return x


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
# Model
# -----------------------------
class TinyCNN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),  # 64->30
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), # 30->13
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2), # 13->6
            nn.ReLU(),
            nn.Flatten(),
        )
        # compute flatten size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            n_flat = self.net(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return self.head(z)


# -----------------------------
# Training
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)

    # env
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--wall_prob", type=float, default=0.20)

    # dqn
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--learn_starts", type=int, default=1_000)
    parser.add_argument("--train_every", type=int, default=4)
    parser.add_argument("--target_update", type=int, default=1_000)

    # exploration
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_steps", type=int, default=30_000)

    # logging
    parser.add_argument("--log_every", type=int, default=1_000)
    parser.add_argument("--out", type=str, default="evaluation/baseline_metrics.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    cfg = MazeConfig(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        obs_size=64,
        wall_prob=args.wall_prob,
        seed=args.seed,
    )
    env = MazeEnv(cfg)

    n_actions = int(env.action_space.n)
    q = TinyCNN(n_actions).to(device)
    q_tgt = TinyCNN(n_actions).to(device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    rb = ReplayBuffer(args.buffer_size)

    # metrics
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("step,episode,episode_return,episode_len,epsilon\n")

    episode_return = 0.0
    episode_len = 0
    episode_idx = 0

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    s = preprocess_obs(obs)

    def epsilon_by_step(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / max(1, args.eps_decay_steps)
        return args.eps_start + frac * (args.eps_end - args.eps_start)

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
            obs2, r, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs2, r, done, info = step_out

        s2 = preprocess_obs(obs2)
        rb.push(Transition(s=s, a=a, r=float(r), s2=s2, done=done))

        episode_return += float(r)
        episode_len += 1

        # advance state
        s = s2

        # if episode ended
        if done:
            # log per-episode
            out_path.open("a", encoding="utf-8").write(
                f"{t},{episode_idx},{episode_return:.4f},{episode_len},{eps:.4f}\n"
            )
            episode_idx += 1
            episode_return = 0.0
            episode_len = 0

            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            s = preprocess_obs(obs)

        # learning
        if t >= args.learn_starts and (t % args.train_every == 0) and len(rb) >= args.batch_size:
            s_b, a_b, r_b, s2_b, d_b = rb.sample(args.batch_size)
            s_b = s_b.to(device)
            a_b = a_b.to(device)
            r_b = r_b.to(device)
            s2_b = s2_b.to(device)
            d_b = d_b.to(device)

            # Q(s,a)
            q_sa = q(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                # max_a' Q_tgt(s',a')
                q_next = q_tgt(s2_b).max(dim=1).values
                target = r_b + (1.0 - d_b) * args.gamma * q_next

            loss = nn.functional.smooth_l1_loss(q_sa, target)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

        # target update
        if t % args.target_update == 0:
            q_tgt.load_state_dict(q.state_dict())

        # periodic console log
        if t % args.log_every == 0:
            print(f"[DQN] step={t} eps={eps:.3f} buffer={len(rb)} episodes={episode_idx}")

    print(f"[DQN] DONE. Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
