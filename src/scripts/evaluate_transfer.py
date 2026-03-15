"""
evaluate_transfer.py  —  Head-to-Head Transfer Evaluation
===========================================================
Compares two agents on a FIXED maze layout:

  1. Pixel DQN    — trained in the real MazeEnv on raw pixel observations
                    (TinyCNN, checkpoint: dqn_baseline.pt)

  2. Dream DQN    — trained entirely inside the latent world model
                    (DreamQNet MLP, checkpoint: dqn_dream.pt)
                    At evaluation time this agent receives DINOv2-encoded
                    latents from the REAL MazeEnv — never the world model.

Both agents face the exact same maze every episode (fixed_grid, seed=0).
This is the primary CDR experiment: the reward gap between the two agents
directly answers the research question "can a latent world model replace
a simulator for RL training?"

Outputs
-------
  evaluation/transfer_results.csv   — per-episode log for both agents
  evaluation/transfer_summary.csv   — headline metrics for CDR slides

Usage
-----
python -m src.scripts.evaluate_transfer \\
    --dqn_weights  checkpoints/dqn_baseline.pt \\
    --dream_weights checkpoints/dqn_dream.pt \\
    --episodes     50

# Quick smoke-test (5 episodes per agent)
python -m src.scripts.evaluate_transfer --smoke_test
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.env.maze_env import MazeEnv, MazeConfig
from src.models.encoder import DinoV2Encoder


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

class TinyCNN(nn.Module):
    """Pixel DQN — must exactly match train_baseline.py architecture."""
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


class DreamQNet(nn.Module):
    """Dream DQN — must exactly match train_dream_dqn.py architecture."""
    def __init__(self, latent_dim: int = 384, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_pixel_obs(obs: np.ndarray) -> torch.Tensor:
    """HxWxC uint8 → 1xCxHxW float32 in [0,1]."""
    return torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def make_env(args) -> tuple[MazeEnv, np.ndarray]:
    """Build env, lock maze layout, return (env, fixed_grid)."""
    cfg = MazeConfig(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        obs_size=64,
        wall_prob=args.wall_prob,
        seed=args.maze_seed,
    )
    env = MazeEnv(cfg)
    env.reset()
    fixed_grid = env.grid.copy()
    return env, fixed_grid


def reset_fixed(env: MazeEnv, fixed_grid: np.ndarray) -> np.ndarray:
    """Reset agent to start position without regenerating the maze."""
    env.grid = fixed_grid.copy()
    env.agent_pos = [0, 0]
    env.steps = 0
    return env._get_obs()


# ─────────────────────────────────────────────────────────────────────────────
# Episode runners
# ─────────────────────────────────────────────────────────────────────────────

def run_episode_pixel_dqn(
    env: MazeEnv,
    fixed_grid: np.ndarray,
    model: TinyCNN,
    device: torch.device,
) -> dict:
    obs = reset_fixed(env, fixed_grid)
    s = preprocess_pixel_obs(obs).to(device)
    total_reward = 0.0
    steps = 0
    success = False

    for _ in range(env.config.max_steps):
        with torch.no_grad():
            action = int(torch.argmax(model(s), dim=1).item())

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
            success = bool(terminated)
        else:
            obs, reward, done, _ = step_out
            success = reward > 0.5

        total_reward += float(reward)
        steps += 1

        if done:
            break

        s = preprocess_pixel_obs(obs).to(device)

    return {"return": total_reward, "steps": steps, "success": int(success)}


def run_episode_dream_dqn(
    env: MazeEnv,
    fixed_grid: np.ndarray,
    model: DreamQNet,
    encoder: DinoV2Encoder,
    device: torch.device,
) -> dict:
    obs = reset_fixed(env, fixed_grid)
    total_reward = 0.0
    steps = 0
    success = False

    for _ in range(env.config.max_steps):
        obs_tensor = torch.from_numpy(obs).to(device)
        z = encoder.encode(obs_tensor)
        z_t = z.float().unsqueeze(0).to(device)

        with torch.no_grad():
            action = int(torch.argmax(model(z_t), dim=1).item())

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
            success = bool(terminated)
        else:
            obs, reward, done, _ = step_out
            success = reward > 0.5

        total_reward += float(reward)
        steps += 1

        if done:
            break

    return {"return": total_reward, "steps": steps, "success": int(success)}


def run_episode_random(
    env: MazeEnv,
    fixed_grid: np.ndarray,
) -> dict:
    obs = reset_fixed(env, fixed_grid)
    total_reward = 0.0
    steps = 0
    success = False

    for _ in range(env.config.max_steps):
        action = random.randint(0, 3)
        step_out = env.step(action)

        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
            success = bool(terminated)
        else:
            obs, reward, done, _ = step_out
            success = reward > 0.5

        total_reward += float(reward)
        steps += 1

        if done:
            break

    return {"return": total_reward, "steps": steps, "success": int(success)}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(label: str, episode_fn, n_episodes: int) -> list[dict]:
    print(f"\n{'─'*60}")
    print(f"  Evaluating: {label}")
    print(f"{'─'*60}")

    results = []
    t0 = time.time()

    for ep in range(n_episodes):
        r = episode_fn()
        r["agent"] = label
        r["episode"] = ep
        results.append(r)

        status = "✔ GOAL" if r["success"] else "✘ fail"
        print(
            f"  ep {ep+1:>3}/{n_episodes}  ret={r['return']:+.4f}  "
            f"steps={r['steps']:>3}  {status}"
        )

    elapsed = time.time() - t0
    sr = 100.0 * sum(r["success"] for r in results) / n_episodes
    avg_ret = np.mean([r["return"] for r in results])
    avg_len = np.mean([r["steps"] for r in results])

    print(f"\n  → Success rate : {sr:.1f}%")
    print(f"  → Mean return  : {avg_ret:+.4f}")
    print(f"  → Mean ep len  : {avg_len:.1f} steps")
    print(f"  → Wall time    : {elapsed:.1f}s")

    return results


def summarise(results: list[dict]) -> dict:
    agent = results[0]["agent"]
    returns = [r["return"] for r in results]
    lengths = [r["steps"] for r in results]
    success = [r["success"] for r in results]
    return {
        "agent": agent,
        "episodes": len(results),
        "success_rate_pct": 100.0 * float(np.mean(success)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_ep_len": float(np.mean(lengths)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
    }


def plot_transfer_summary(
    summaries: list[dict],
    out_path: str,
    dream_train_success: float | None = None,
):
    labels = [s["agent"] for s in summaries]
    success_rates = [s["success_rate_pct"] for s in summaries]
    mean_returns = [s["mean_return"] for s in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].bar(labels, success_rates)
    axes[0].set_title("Goal Reach Rate")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_ylim(0, 110)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")

    axes[1].bar(labels, mean_returns)
    axes[1].set_title("Mean Episodic Return")
    axes[1].set_ylabel("Mean Return")
    for i, v in enumerate(mean_returns):
        offset = 0.02 if v >= 0 else -0.02
        axes[1].text(i, v + offset, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Transfer Experiment: Dream DQN vs Pixel DQN vs Random\n"
        "(Real MazeEnv, seed=0, 50 episodes)",
        fontsize=18,
        fontweight="bold",
    )

    if dream_train_success is not None:
        fig.text(
            0.5,
            0.03,
            f"Dream DQN in world model: {dream_train_success:.1f}% success | "
            f"zero-shot transfer evaluated separately in real environment",
            ha="center",
            fontsize=12,
            style="italic",
        )

    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Head-to-head: Pixel DQN vs Dream DQN vs Random in real MazeEnv"
    )
    p.add_argument("--dqn_weights", type=str, default="checkpoints/dqn_baseline.pt")
    p.add_argument("--dream_weights", type=str, default="checkpoints/dqn_dream.pt")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--maze_seed", type=int, default=0)
    p.add_argument("--grid_size", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=64)
    p.add_argument("--wall_prob", type=float, default=0.20)
    p.add_argument("--out_csv", type=str, default="evaluation/transfer_results.csv")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument(
        "--dream_train_success",
        type=float,
        default=94.1,
        help="Final dream-training success rate used only for plot annotation.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.smoke_test:
        print("=" * 60)
        print("  SMOKE TEST — 5 episodes per agent")
        print("=" * 60)
        args.episodes = 5

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device     : {device}")

    env, fixed_grid = make_env(args)
    print(f"[INFO] Maze seed  : {args.maze_seed} — same layout every episode")
    print(f"[INFO] Episodes   : {args.episodes} per agent")

    print("\n[INFO] Loading DINOv2 encoder for latent encoding...")
    encoder = DinoV2Encoder(device=device)

    all_results = []
    summaries = []

    dqn_path = Path(args.dqn_weights)
    if dqn_path.exists():
        print(f"\n[INFO] Loading Pixel DQN from {dqn_path}")
        ckpt = torch.load(dqn_path, map_location=device)
        obs_size = ckpt.get("obs_size", 64)
        pixel_dqn = TinyCNN(n_actions=4, obs_size=obs_size).to(device)
        pixel_dqn.load_state_dict(ckpt.get("model_state", ckpt))
        pixel_dqn.eval()

        pixel_fn = lambda: run_episode_pixel_dqn(env, fixed_grid, pixel_dqn, device)
        results_p = evaluate_agent("Pixel DQN (real env trained)", pixel_fn, args.episodes)
        all_results.extend(results_p)
        summaries.append(summarise(results_p))
    else:
        print(f"\n[WARN] Pixel DQN weights not found at {dqn_path} — skipping.")

    dream_path = Path(args.dream_weights)
    if dream_path.exists():
        print(f"\n[INFO] Loading Dream DQN from {dream_path}")
        ckpt = torch.load(dream_path, map_location=device)
        latent_dim = ckpt.get("latent_dim", 384)
        action_dim = ckpt.get("action_dim", 4)
        dream_dqn = DreamQNet(latent_dim=latent_dim, action_dim=action_dim).to(device)
        dream_dqn.load_state_dict(ckpt["model_state"])
        dream_dqn.eval()

        dream_fn = lambda: run_episode_dream_dqn(env, fixed_grid, dream_dqn, encoder, device)
        results_d = evaluate_agent("Dream DQN (world-model trained)", dream_fn, args.episodes)
        all_results.extend(results_d)
        summaries.append(summarise(results_d))
    else:
        print(f"\n[WARN] Dream DQN weights not found at {dream_path} — skipping.")

    print("\n[INFO] Evaluating Random baseline")
    random_fn = lambda: run_episode_random(env, fixed_grid)
    results_r = evaluate_agent("Random baseline", random_fn, args.episodes)
    all_results.extend(results_r)
    summaries.append(summarise(results_r))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["agent", "episode", "return", "steps", "success"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "agent": r["agent"],
                "episode": r["episode"],
                "return": f"{r['return']:.4f}",
                "steps": r["steps"],
                "success": r["success"],
            })
    print(f"\n[INFO] Per-episode results → {out_path}")

    summary_path = Path("evaluation/transfer_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[INFO] Summary           → {summary_path}")

    plot_transfer_summary(
        summaries,
        out_path="evaluation/transfer_curves.png",
        dream_train_success=args.dream_train_success,
    )
    print("[INFO] Plot summary       → evaluation/transfer_curves.png")


if __name__ == "__main__":
    main()
