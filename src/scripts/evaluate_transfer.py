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
    env.grid      = fixed_grid.copy()
    env.agent_pos = [0, 0]
    env.steps     = 0
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
    """
    Run one greedy episode with the pixel DQN in the real MazeEnv.
    Observations: raw RGB pixels (HxWx3 uint8).
    """
    obs          = reset_fixed(env, fixed_grid)
    s            = preprocess_pixel_obs(obs).to(device)
    total_reward = 0.0
    steps        = 0
    success      = False

    for _ in range(env.config.max_steps):
        with torch.no_grad():
            action = int(torch.argmax(model(s), dim=1).item())

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done    = bool(terminated or truncated)
            success = bool(terminated)          # goal reached before step limit
        else:                                   # legacy 4-tuple API
            obs, reward, done, _ = step_out
            success = reward > 0.5

        total_reward += float(reward)
        steps        += 1

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
    """
    Run one greedy episode with the dream DQN in the REAL MazeEnv.

    The key transfer step: the agent was trained on latent observations
    inside the world model. Here we encode real pixel observations with
    DINOv2 and feed the resulting latent vectors to the same Q-network.
    This is the zero-shot transfer from dream → reality.

    Observations: DINOv2 latent vectors (384-dim float32).
    """
    obs          = reset_fixed(env, fixed_grid)
    total_reward = 0.0
    steps        = 0
    success      = False

    for _ in range(env.config.max_steps):
        # Encode real pixel obs → latent vector
        # encoder.encode expects (H, W, C) uint8, returns (384,) float32
        obs_tensor = torch.from_numpy(obs).to(device)
        z = encoder.encode(obs_tensor)
        z_t = z.float().unsqueeze(0).to(device)  # (1, 384)

        with torch.no_grad():
            action = int(torch.argmax(model(z_t), dim=1).item())

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done    = bool(terminated or truncated)
            success = bool(terminated)
        else:
            obs, reward, done, _ = step_out
            success = reward > 0.5

        total_reward += float(reward)
        steps        += 1

        if done:
            break

    return {"return": total_reward, "steps": steps, "success": int(success)}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(label: str, episode_fn, n_episodes: int) -> list[dict]:
    """Run n_episodes and return per-episode result dicts."""
    print(f"\n{'─'*60}")
    print(f"  Evaluating: {label}")
    print(f"{'─'*60}")

    results = []
    t0      = time.time()

    for ep in range(n_episodes):
        r = episode_fn()
        r["agent"]   = label
        r["episode"] = ep
        results.append(r)

        status = "✔ GOAL" if r["success"] else "✘ fail"
        print(f"  ep {ep+1:>3}/{n_episodes}  ret={r['return']:+.4f}  "
              f"steps={r['steps']:>3}  {status}")

    elapsed = time.time() - t0
    sr      = 100.0 * sum(r["success"] for r in results) / n_episodes
    avg_ret = np.mean([r["return"] for r in results])
    avg_len = np.mean([r["steps"]  for r in results])

    print(f"\n  → Success rate : {sr:.1f}%")
    print(f"  → Mean return  : {avg_ret:+.4f}")
    print(f"  → Mean ep len  : {avg_len:.1f} steps")
    print(f"  → Wall time    : {elapsed:.1f}s")

    return results


def summarise(results: list[dict]) -> dict:
    agent   = results[0]["agent"]
    returns = [r["return"]  for r in results]
    lengths = [r["steps"]   for r in results]
    success = [r["success"] for r in results]
    return {
        "agent":            agent,
        "episodes":         len(results),
        "success_rate_pct": 100.0 * float(np.mean(success)),
        "mean_return":      float(np.mean(returns)),
        "std_return":       float(np.std(returns)),
        "mean_ep_len":      float(np.mean(lengths)),
        "min_return":       float(np.min(returns)),
        "max_return":       float(np.max(returns)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Head-to-head: Pixel DQN vs Dream DQN in real MazeEnv"
    )
    p.add_argument("--dqn_weights",   type=str,   default="checkpoints/dqn_baseline.pt",
                   help="Pixel DQN checkpoint (train_baseline.py output)")
    p.add_argument("--dream_weights", type=str,   default="checkpoints/dqn_dream.pt",
                   help="Dream DQN checkpoint (train_dream_dqn.py output)")
    p.add_argument("--episodes",      type=int,   default=50)
    p.add_argument("--maze_seed",     type=int,   default=0)
    p.add_argument("--grid_size",     type=int,   default=10)
    p.add_argument("--max_steps",     type=int,   default=64)
    p.add_argument("--wall_prob",     type=float, default=0.20)
    p.add_argument("--out_csv",       type=str,   default="evaluation/transfer_results.csv")
    p.add_argument("--smoke_test",    action="store_true",
                   help="Run 5 episodes per agent for a quick sanity check")
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

    # ── Build env with fixed maze ─────────────────────────────────────────────
    env, fixed_grid = make_env(args)
    print(f"[INFO] Maze seed  : {args.maze_seed} — same layout every episode")
    print(f"[INFO] Episodes   : {args.episodes} per agent")

    # ── Load DINOv2 encoder (needed for Dream DQN transfer) ───────────────────
    print("\n[INFO] Loading DINOv2 encoder for latent encoding...")
    encoder = DinoV2Encoder(device=device)

    all_results = []
    summaries   = []

    # ── Agent 1: Pixel DQN ────────────────────────────────────────────────────
    dqn_path = Path(args.dqn_weights)
    if dqn_path.exists():
        print(f"\n[INFO] Loading Pixel DQN from {dqn_path}")
        ckpt      = torch.load(dqn_path, map_location=device)
        obs_size  = ckpt.get("obs_size", 64)
        pixel_dqn = TinyCNN(n_actions=4, obs_size=obs_size).to(device)
        pixel_dqn.load_state_dict(ckpt.get("model_state", ckpt))
        pixel_dqn.eval()

        pixel_fn  = lambda: run_episode_pixel_dqn(env, fixed_grid, pixel_dqn, device)
        results_p = evaluate_agent("Pixel DQN (real env trained)", pixel_fn, args.episodes)
        all_results.extend(results_p)
        summaries.append(summarise(results_p))
    else:
        print(f"\n[WARN] Pixel DQN weights not found at {dqn_path} — skipping.")
        print(f"       Run: python -m src.scripts.train_baseline")

    # ── Agent 2: Dream DQN ────────────────────────────────────────────────────
    dream_path = Path(args.dream_weights)
    if dream_path.exists():
        print(f"\n[INFO] Loading Dream DQN from {dream_path}")
        ckpt       = torch.load(dream_path, map_location=device)
        latent_dim = ckpt.get("latent_dim", 384)
        action_dim = ckpt.get("action_dim", 4)
        dream_dqn  = DreamQNet(latent_dim=latent_dim, action_dim=action_dim).to(device)
        dream_dqn.load_state_dict(ckpt["model_state"])
        dream_dqn.eval()
        print(f"[INFO] Dream DQN trained for {ckpt.get('steps_trained', '?')} steps "
              f"| final dream success: {ckpt.get('final_success_pct', 0.0):.1f}%")

        dream_fn  = lambda: run_episode_dream_dqn(
            env, fixed_grid, dream_dqn, encoder, device
        )
        results_d = evaluate_agent("Dream DQN (world-model trained)", dream_fn, args.episodes)
        all_results.extend(results_d)
        summaries.append(summarise(results_d))
    else:
        print(f"\n[WARN] Dream DQN weights not found at {dream_path} — skipping.")
        print(f"       Run: python -m src.scripts.train_dream_dqn")

    # ── Save per-episode CSV ──────────────────────────────────────────────────
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["agent", "episode", "return", "steps", "success"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "agent":   r["agent"],
                "episode": r["episode"],
                "return":  f"{r['return']:.4f}",
                "steps":   r["steps"],
                "success": r["success"],
            })
    print(f"\n[INFO] Per-episode results → {out_path}")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summary_path = Path("evaluation/transfer_summary.csv")
    if summaries:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"[INFO] Summary           → {summary_path}")

    # ── Print CDR headline table ──────────────────────────────────────────────
    if len(summaries) == 2:
        p_sum, d_sum = summaries[0], summaries[1]
        gap = p_sum["success_rate_pct"] - d_sum["success_rate_pct"]

        print(f"\n{'='*60}")
        print(f"  CDR TRANSFER EXPERIMENT RESULTS")
        print(f"  {'Metric':<28} {'Pixel DQN':>12} {'Dream DQN':>12}")
        print(f"  {'-'*54}")
        print(f"  {'Success rate (%)':<28} "
              f"{p_sum['success_rate_pct']:>11.1f}% "
              f"{d_sum['success_rate_pct']:>11.1f}%")
        print(f"  {'Mean return':<28} "
              f"{p_sum['mean_return']:>+12.4f} "
              f"{d_sum['mean_return']:>+12.4f}")
        print(f"  {'Std return':<28} "
              f"{p_sum['std_return']:>12.4f} "
              f"{d_sum['std_return']:>12.4f}")
        print(f"  {'Mean episode length':<28} "
              f"{p_sum['mean_ep_len']:>12.1f} "
              f"{d_sum['mean_ep_len']:>12.1f}")
        print(f"  {'-'*54}")
        print(f"  Transfer reward gap (Pixel − Dream): {gap:+.1f}%")
        print(f"{'='*60}\n")

        if abs(gap) < 10:
            print("  ✔ Small gap — world model is a good simulator substitute.")
        elif d_sum["success_rate_pct"] > p_sum["success_rate_pct"]:
            print("  ✔ Dream DQN outperforms pixel DQN — latent training advantage.")
        else:
            print(f"  ✘ Gap of {gap:.1f}% — world model introduces bias; discuss in CDR.")


if __name__ == "__main__":
    main()