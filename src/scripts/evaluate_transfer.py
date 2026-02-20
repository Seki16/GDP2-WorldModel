"""
evaluate_transfer.py  —  Head-to-Head Transfer Evaluation
===========================================================
Compares two agents on a FIXED maze layout:

  1. DQN Baseline  — trained inside the conventional simulator
  2. CEM Agent     — plans in latent space using the trained world model

Both agents face the exact same maze every episode (fixed_grid).
Results saved to evaluation/transfer_results.csv and transfer_summary.csv.

Usage
-----
python -m src.scripts.evaluate_transfer \
    --dqn_weights checkpoints/dqn_baseline.pt \
    --wm_weights  checkpoints/world_model_best.pt \
    --episodes    50

# Quick smoke-test
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

try:
    from src.models.transformer import DinoWorldModel, CEMPlanner, MPCController
    from src.models.transformer_configuration import TransformerWMConfiguration as WMConfig
    _REAL_MODEL = True
except ImportError:
    _REAL_MODEL = False
    print("[WARN] Could not import transformer — CEM agent will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# DQN model — must match train_baseline.py exactly
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(obs).float().permute(2, 0, 1) / 255.0


def make_env(args) -> tuple[MazeEnv, np.ndarray]:
    """
    Build env, do one reset to generate the maze, then lock the layout.
    Returns env and fixed_grid so every episode uses the same maze.
    """
    cfg = MazeConfig(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        obs_size=60,
        wall_prob=args.wall_prob,
        seed=args.maze_seed,
    )
    env = MazeEnv(cfg)
    env.reset()
    fixed_grid = env.grid.copy()
    return env, fixed_grid


def reset_fixed(env: MazeEnv, fixed_grid: np.ndarray) -> np.ndarray:
    """Reset agent to [0,0] without rebuilding the maze."""
    env.grid = fixed_grid.copy()
    env.agent_pos = [0, 0]
    env.steps = 0
    return env._get_obs()


# ─────────────────────────────────────────────────────────────────────────────
# Episode runners
# ─────────────────────────────────────────────────────────────────────────────

def run_episode_dqn(env: MazeEnv, fixed_grid: np.ndarray,
                    model: TinyCNN, device: torch.device) -> dict:
    obs = reset_fixed(env, fixed_grid)
    s   = preprocess_obs(obs).unsqueeze(0).to(device)

    total_reward = 0.0
    steps        = 0
    success      = False

    while True:
        with torch.no_grad():
            action = int(torch.argmax(model(s), dim=1).item())

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done    = bool(terminated or truncated)
            success = bool(terminated)
        else:
            obs, reward, done, _ = step_out
            success = reward > 0.5

        total_reward += reward
        steps        += 1

        if done:
            break

        s = preprocess_obs(obs).unsqueeze(0).to(device)

    return {"return": total_reward, "steps": steps, "success": int(success)}


def run_episode_cem(env: MazeEnv, fixed_grid: np.ndarray,
                    encoder: DinoV2Encoder, controller: MPCController,
                    device: torch.device) -> dict:
    """
    CEM planning loop:
      encode real obs → plan in latent space → act in real env → repeat.
    The agent never trains — it plans at inference time using the world model.
    """
    obs = reset_fixed(env, fixed_grid)

    total_reward = 0.0
    steps        = 0
    success      = False

    while True:
        # Encode current observation to latent state
        obs_tensor = torch.from_numpy(obs).to(device)
        z  = encoder.encode(obs_tensor)       # (384,)
        z0 = z.unsqueeze(0).unsqueeze(0)      # (1, 1, 384)

        # Plan in latent space, return best first action
        action = int(controller.act(z0).item())

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done    = bool(terminated or truncated)
            success = bool(terminated)
        else:
            obs, reward, done, _ = step_out
            success = reward > 0.5

        total_reward += reward
        steps        += 1

        if done:
            break

    return {"return": total_reward, "steps": steps, "success": int(success)}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-episode loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent_name: str, run_fn, n_episodes: int) -> list[dict]:
    results = []
    print(f"\n{'─'*60}")
    print(f"  Evaluating: {agent_name}  ({n_episodes} episodes)")
    print(f"{'─'*60}")

    for ep in range(n_episodes):
        metrics          = run_fn()
        metrics["ep"]    = ep
        metrics["agent"] = agent_name
        results.append(metrics)

        status = "✓ GOAL" if metrics["success"] else "✗ timeout"
        print(f"  ep {ep+1:4d}/{n_episodes}  "
              f"return={metrics['return']:+.3f}  "
              f"steps={metrics['steps']:3d}  {status}")

    return results


def summarise(results: list[dict]) -> dict:
    returns  = [r["return"]  for r in results]
    steps    = [r["steps"]   for r in results]
    successes= [r["success"] for r in results]
    return {
        "agent":        results[0]["agent"],
        "episodes":     len(results),
        "success_rate": f"{100*sum(successes)/len(successes):.1f}%",
        "mean_return":  f"{np.mean(returns):.4f}",
        "std_return":   f"{np.std(returns):.4f}",
        "mean_steps":   f"{np.mean(steps):.1f}",
        "std_steps":    f"{np.std(steps):.1f}",
    }


def print_table(summaries: list[dict]):
    keys    = ["agent","episodes","success_rate","mean_return","std_return","mean_steps","std_steps"]
    headers = ["Agent","Episodes","Success Rate","Mean Return","Std Return","Mean Steps","Std Steps"]
    col_w   = [max(len(h), max(len(str(s[k])) for s in summaries))
               for h, k in zip(headers, keys)]
    sep  = "  ".join("─"*w for w in col_w)
    head = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(f"\n{'═'*len(sep)}")
    print("  HEAD-TO-HEAD TRANSFER EVALUATION RESULTS")
    print(f"{'═'*len(sep)}")
    print(head)
    print(sep)
    for s in summaries:
        print("  ".join(str(s[k]).ljust(w) for k, w in zip(keys, col_w)))
    print(f"{'═'*len(sep)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dqn_weights",    type=str,   default="checkpoints/dqn_baseline.pt")
    p.add_argument("--wm_weights",     type=str,   default="checkpoints/world_model_best.pt")
    p.add_argument("--episodes",       type=int,   default=50)
    p.add_argument("--smoke_test",     action="store_true")
    p.add_argument("--maze_seed",      type=int,   default=0,
                   help="Must match the maze_seed used in train_baseline.py")
    p.add_argument("--grid_size",      type=int,   default=10)
    p.add_argument("--max_steps",      type=int,   default=64)
    p.add_argument("--wall_prob",      type=float, default=0.20)
    p.add_argument("--cem_candidates", type=int,   default=8)
    p.add_argument("--cem_elites",     type=int,   default=8)
    p.add_argument("--cem_iters",      type=int,   default=4)
    p.add_argument("--cem_horizon",    type=int,   default=16)
    p.add_argument("--out_csv",        type=str,   default="evaluation/transfer_results.csv")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device    : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU       : {torch.cuda.get_device_name(0)}")

    # Build env with fixed maze
    env, fixed_grid = make_env(args)
    print(f"[INFO] Maze seed : {args.maze_seed} — same layout every episode")

    all_results = []
    summaries   = []

    # ── Agent 1: DQN ─────────────────────────────────────────────────────────
    dqn_path = Path(args.dqn_weights)
    if dqn_path.exists():
        print(f"\n[INFO] Loading DQN from {dqn_path}")
        ckpt     = torch.load(dqn_path, map_location=device)
        obs_size = ckpt.get("obs_size", 60)
        dqn      = TinyCNN(n_actions=4, obs_size=obs_size).to(device)
        dqn.load_state_dict(ckpt.get("model_state", ckpt))
        dqn.eval()

        dqn_fn      = lambda: run_episode_dqn(env, fixed_grid, dqn, device)
        dqn_results = evaluate_agent("DQN (real simulator)", dqn_fn, args.episodes)
        all_results.extend(dqn_results)
        summaries.append(summarise(dqn_results))
    else:
        print(f"\n[WARN] DQN weights not found at {dqn_path} — skipping.")
        print(f"       Run train_baseline.py first.")

    # ── Agent 2: CEM ─────────────────────────────────────────────────────────
    wm_path = Path(args.wm_weights)
    if wm_path.exists() and _REAL_MODEL:
        print(f"\n[INFO] Loading world model from {wm_path}")
        wm_ckpt     = torch.load(wm_path, map_location=device)
        world_model = DinoWorldModel(WMConfig()).to(device)
        world_model.load_state_dict(wm_ckpt.get("model_state", wm_ckpt))
        world_model.eval()

        planner    = CEMPlanner(
            model          = world_model,
            action_dim     = 4,
            horizon        = args.cem_horizon,
            num_candidates = args.cem_candidates,
            num_elites     = args.cem_elites,
            num_iters      = args.cem_iters,
            gamma          = 0.99,
            device         = device,
        )
        controller = MPCController(planner)

        print("[INFO] Loading DINOv2 encoder …")
        encoder = DinoV2Encoder(device=str(device))

        cem_fn      = lambda: run_episode_cem(env, fixed_grid, encoder, controller, device)
        cem_results = evaluate_agent("CEM (latent planner)", cem_fn, args.episodes)
        all_results.extend(cem_results)
        summaries.append(summarise(cem_results))
    elif not wm_path.exists():
        print(f"\n[WARN] World model not found at {wm_path} — skipping.")
    else:
        print("\n[WARN] Transformer import failed — skipping CEM.")

    if not summaries:
        print("\n[ERROR] No agents evaluated. Check weight paths.")
        return

    print_table(summaries)

    # Save CSVs
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent","ep","return","steps","success"])
        writer.writeheader()
        writer.writerows(all_results)

    summary_path = out_path.parent / "transfer_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print(f"[INFO] Per-episode CSV → {out_path}")
    print(f"[INFO] Summary CSV     → {summary_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[INFO] Total time: {(time.time()-t0)/60:.1f} min")