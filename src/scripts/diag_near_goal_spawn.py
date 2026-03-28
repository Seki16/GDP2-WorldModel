"""
diag_near_goal_spawn.py — Near-goal spawn diagnostic for Dream DQN
====================================================================
Tests the Dream DQN from multiple starting positions at varying
Manhattan distances from the goal [9,9].

Research question (from supervisors):
    If we spawn the agent close to the goal, does transfer improve?
    - YES → long-horizon value propagation problem (credit assignment)
    - NO  → Q-function is fundamentally disconnected from real latents

We test distances 1, 2, 3, 5, and the default [0,0] (~18 steps optimal).
For each distance, we find all valid (non-wall) cells at that Manhattan
distance from the goal, run the Dream DQN from each, and report results.

Usage:
    python -m src.scripts.diag_near_goal_spawn \
        --dream_checkpoint checkpoints/dqn_ss_tbptt/dqn_dream.pt

    # Also test with CDR checkpoint for comparison:
    python -m src.scripts.diag_near_goal_spawn \
        --dream_checkpoint checkpoints/dqn_dream.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.env.maze_env import MazeEnv, MazeConfig
from src.models.encoder import DinoV2Encoder
from src.scripts.train_dream_dqn import DreamQNet


def find_cells_at_distance(
    grid: np.ndarray, goal: tuple[int, int], target_dist: int
) -> list[tuple[int, int]]:
    """
    Find all non-wall cells at exactly `target_dist` Manhattan distance
    from the goal. Returns list of (row, col) positions.
    """
    n = grid.shape[0]
    cells = []
    for r in range(n):
        for c in range(n):
            if grid[r, c] == 0:  # not a wall
                dist = abs(r - goal[0]) + abs(c - goal[1])
                if dist == target_dist:
                    cells.append((r, c))
    return cells


def run_episode_from_pos(
    env: MazeEnv,
    fixed_grid: np.ndarray,
    start_pos: tuple[int, int],
    dream_dqn: DreamQNet,
    encoder: DinoV2Encoder,
    device: torch.device,
) -> dict:
    """
    Run one greedy episode with Dream DQN starting from a custom position.
    Uses DINOv2 encoding of real observations (transfer setting).
    """
    # Reset env to custom start position
    env.grid = fixed_grid.copy()
    env.agent_pos = list(start_pos)
    env.steps = 0
    obs = env._get_obs()

    total_reward = 0.0
    steps = 0
    success = False
    actions = []

    for _ in range(env.config.max_steps):
        # Encode real pixel obs → DINOv2 latent
        obs_tensor = torch.from_numpy(obs).to(device)
        z = encoder.encode(obs_tensor)
        z_t = z.float().unsqueeze(0).to(device)

        # Dream DQN selects action
        with torch.no_grad():
            q_vals = dream_dqn(z_t)
            action = int(torch.argmax(q_vals, dim=1).item())

        actions.append(action)

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
            success = bool(terminated)
        else:
            obs, reward, done, info = step_out
            success = bool(reward > 0.5)

        total_reward += float(reward)
        steps += 1

        if done:
            break

    # Action distribution
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
    action_counts = {name: 0 for name in action_names.values()}
    for a in actions:
        action_counts[action_names[a]] += 1

    return {
        "start_pos": start_pos,
        "success": success,
        "steps": steps,
        "return": total_reward,
        "actions": action_counts,
        "dominant_action": max(action_counts, key=action_counts.get),
        "dominant_pct": max(action_counts.values()) / max(len(actions), 1) * 100,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Near-goal spawn diagnostic for Dream DQN"
    )
    parser.add_argument("--dream_checkpoint", type=str,
                        default="checkpoints/dqn_ss_tbptt/dqn_dream.pt")
    parser.add_argument("--maze_seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--obs_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--wall_prob", type=float, default=0.20)
    parser.add_argument("--distances", type=str, default="1,2,3,5,18",
                        help="Comma-separated Manhattan distances to test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distances = [int(d) for d in args.distances.split(",")]

    # ── Load Dream DQN ────────────────────────────────────────────────────────
    ckpt_path = Path(args.dream_checkpoint)
    print(f"[INFO] Loading Dream DQN from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    latent_dim = ckpt.get("latent_dim", 384)
    action_dim = ckpt.get("action_dim", 4)
    dream_dqn = DreamQNet(latent_dim=latent_dim, action_dim=action_dim).to(device)
    dream_dqn.load_state_dict(ckpt["model_state"])
    dream_dqn.eval()

    dream_success = ckpt.get("final_success_pct", 0.0)
    print(f"[INFO] Dream DQN in-dream success: {dream_success:.1f}%")

    # ── Load DINOv2 ───────────────────────────────────────────────────────────
    print("[INFO] Loading DINOv2 encoder...")
    encoder = DinoV2Encoder(device=device)

    # ── Set up maze ───────────────────────────────────────────────────────────
    env = MazeEnv(
        MazeConfig(
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            obs_size=args.obs_size,
            wall_prob=args.wall_prob,
            seed=args.maze_seed,
        )
    )
    env.reset()
    fixed_grid = env.grid.copy()
    goal = (args.grid_size - 1, args.grid_size - 1)

    # ── Run diagnostic per distance ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  NEAR-GOAL SPAWN DIAGNOSTIC")
    print(f"  Goal position: {goal}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*70}")

    summary_rows = []

    for dist in distances:
        cells = find_cells_at_distance(fixed_grid, goal, dist)

        if not cells:
            print(f"\n  Distance {dist}: no valid (non-wall) cells found — skipping")
            continue

        print(f"\n  Distance {dist}: {len(cells)} valid cell(s)")
        print(f"  {'Pos':>8}  {'Success':>8}  {'Steps':>6}  {'Return':>8}  "
              f"{'Dominant':>10}  {'Dom%':>5}")
        print(f"  {'-'*55}")

        dist_successes = 0
        dist_results = []

        for cell in cells:
            result = run_episode_from_pos(
                env, fixed_grid, cell, dream_dqn, encoder, device
            )
            dist_results.append(result)

            s = "YES" if result["success"] else "no"
            if result["success"]:
                dist_successes += 1

            print(f"  {str(cell):>8}  {s:>8}  {result['steps']:>6}  "
                  f"{result['return']:>+8.2f}  "
                  f"{result['dominant_action']:>10}  "
                  f"{result['dominant_pct']:>5.0f}%")

        success_rate = dist_successes / len(cells) * 100
        avg_return = np.mean([r["return"] for r in dist_results])

        summary_rows.append({
            "distance": dist,
            "n_cells": len(cells),
            "success_rate": success_rate,
            "avg_return": avg_return,
        })

        print(f"  ── Distance {dist} summary: "
              f"{success_rate:.0f}% success ({dist_successes}/{len(cells)}), "
              f"avg return {avg_return:+.2f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Dist':>5}  {'Cells':>6}  {'Success%':>9}  {'AvgReturn':>10}")
    print(f"  {'-'*35}")
    for row in summary_rows:
        print(f"  {row['distance']:>5}  {row['n_cells']:>6}  "
              f"{row['success_rate']:>8.0f}%  {row['avg_return']:>+10.2f}")

    print(f"\n  INTERPRETATION:")
    any_success = any(r["success_rate"] > 0 for r in summary_rows)
    near_success = any(r["success_rate"] > 0 for r in summary_rows
                       if r["distance"] <= 3)

    if near_success:
        print(f"  → Agent succeeds near the goal but fails far away.")
        print(f"  → Diagnosis: HORIZON PROBLEM — Q-values don't propagate")
        print(f"    across long drifted rollouts, but local dynamics are OK.")
        print(f"  → Implication: shorter rollouts (Dyna) could help if drift")
        print(f"    is controlled within the rollout window.")
    elif any_success:
        print(f"  → Agent succeeds at intermediate distances but not near/far.")
        print(f"  → Diagnosis: PARTIAL ALIGNMENT — some latent regions transfer.")
        print(f"  → Needs further investigation.")
    else:
        print(f"  → Agent fails at ALL distances, including 1 step from goal.")
        print(f"  → Diagnosis: DISTRIBUTION MISMATCH — the Dream DQN's")
        print(f"    Q-function was trained on drifted latents that occupy a")
        print(f"    different region of R^384 than real DINOv2 latents.")
        print(f"  → The Q-function has never seen inputs from the real latent")
        print(f"    distribution, so its outputs are meaningless at transfer.")
        print(f"  → This is consistent with the single-action collapse observed")
        print(f"    in demo_dream_transfer.py.")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()