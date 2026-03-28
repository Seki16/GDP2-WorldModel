"""
diag_reward_head_near_goal.py — Does the WM reward head fire near the goal?
=============================================================================
This diagnostic answers: if we give the WM the CORRECT actions from
near-goal positions, does the reward head predict +1.0 when the agent
should reach the goal?

If NO → the reward head is the bottleneck. No DQN training will work
         because there's no positive reward signal in dream episodes.
If YES → the reward head works, and the 0% in-dream success is caused
          by the DQN not finding the correct actions (exploration issue).

Action mapping: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
Goal: [9,9]

Usage:
    python -m src.scripts.diag_reward_head_near_goal \
        --wm_checkpoint checkpoints/wm_ss_tbptt/world_model_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.env.maze_env import MazeEnv, MazeConfig
from src.models.encoder import DinoV2Encoder

try:
    from src.models.transformer import DinoWorldModel
    from src.models.transformer_configuration import TransformerWMConfiguration as Config
except ImportError:
    raise ImportError("Need src.models.transformer for this diagnostic")


def compute_optimal_actions(start: tuple[int, int], goal: tuple[int, int],
                            grid: np.ndarray) -> list[int]:
    """
    Compute a simple greedy action sequence from start to goal.
    Moves: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)

    Strategy: go down first (increase row), then right (increase col).
    Skip walls by trying the other axis.
    """
    r, c = start
    gr, gc = goal
    actions = []
    n = grid.shape[0]

    for _ in range(20):  # safety limit
        if (r, c) == (gr, gc):
            break

        # Prefer moving toward goal: down if r < gr, right if c < gc
        candidates = []
        if r < gr:
            candidates.append((1, r + 1, c))   # down
        if r > gr:
            candidates.append((0, r - 1, c))   # up
        if c < gc:
            candidates.append((3, r, c + 1))   # right
        if c > gc:
            candidates.append((2, r, c - 1))   # left

        moved = False
        for action, nr, nc in candidates:
            nr = int(np.clip(nr, 0, n - 1))
            nc = int(np.clip(nc, 0, n - 1))
            if grid[nr, nc] == 0:  # not a wall
                actions.append(action)
                r, c = nr, nc
                moved = True
                break

        if not moved:
            # Stuck — try any non-wall neighbor
            for action, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                nr, nc = r + dr, c + dc
                nr = int(np.clip(nr, 0, n - 1))
                nc = int(np.clip(nc, 0, n - 1))
                if grid[nr, nc] == 0 and (nr, nc) != (r, c):
                    actions.append(action)
                    r, c = nr, nc
                    moved = True
                    break
            if not moved:
                break  # completely stuck

    return actions


def main():
    parser = argparse.ArgumentParser(
        description="Reward head diagnostic near goal"
    )
    parser.add_argument("--wm_checkpoint", type=str,
                        default="checkpoints/wm_ss_tbptt/world_model_best.pt")
    parser.add_argument("--maze_seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--obs_size", type=int, default=64)
    parser.add_argument("--max_distance", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    goal = (args.grid_size - 1, args.grid_size - 1)

    # ── Load WM ───────────────────────────────────────────────────────────────
    print(f"[INFO] Loading WM from {args.wm_checkpoint}")
    config = Config()
    model = DinoWorldModel(config).to(device)
    ckpt = torch.load(args.wm_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # ── Load DINOv2 ───────────────────────────────────────────────────────────
    print("[INFO] Loading DINOv2 encoder...")
    encoder = DinoV2Encoder(device=device)

    # ── Build maze ────────────────────────────────────────────────────────────
    env = MazeEnv(MazeConfig(
        grid_size=args.grid_size, max_steps=64, obs_size=args.obs_size,
        wall_prob=0.20, seed=args.maze_seed,
    ))
    env.reset()
    grid = env.grid.copy()

    # ── Find near-goal cells ──────────────────────────────────────────────────
    anchors = []
    for r in range(args.grid_size):
        for c in range(args.grid_size):
            dist = abs(r - goal[0]) + abs(c - goal[1])
            if 0 < dist <= args.max_distance and grid[r, c] == 0:
                actions = compute_optimal_actions((r, c), goal, grid)
                anchors.append({
                    "pos": (r, c),
                    "distance": dist,
                    "optimal_actions": actions,
                })

    print(f"\n{'='*70}")
    print(f"  REWARD HEAD NEAR-GOAL DIAGNOSTIC")
    print(f"  WM: {args.wm_checkpoint}")
    print(f"  Goal: {goal}")
    print(f"{'='*70}")

    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}

    for anchor in sorted(anchors, key=lambda a: a["distance"]):
        pos = anchor["pos"]
        dist = anchor["distance"]
        actions = anchor["optimal_actions"]

        # Encode real observation from this position
        env.grid = grid.copy()
        env.agent_pos = list(pos)
        env.steps = 0
        obs = env._get_obs()

        obs_tensor = torch.from_numpy(obs).to(device)
        z = encoder.encode(obs_tensor)  # (384,)
        z_current = z.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 384)

        print(f"\n  Anchor {pos}, distance={dist}, "
              f"optimal path: {[action_names[a] for a in actions]}")

        # Also get real latent at goal for MSE comparison
        env.grid = grid.copy()
        env.agent_pos = list(goal)
        env.steps = 0
        goal_obs = env._get_obs()
        goal_tensor = torch.from_numpy(goal_obs).to(device)
        z_goal = encoder.encode(goal_tensor)  # (384,)

        # Step 0: what does reward head say on the REAL starting latent?
        with torch.no_grad():
            # We need to call the WM with a dummy action to get reward prediction
            # for the current state. Use action 0 (doesn't matter for step 0 report).
            dummy_a = torch.tensor([[0]], dtype=torch.long, device=device)
            _, pred_rew_0, pred_done_0 = model(z_current, dummy_a)
            rew_0 = float(pred_rew_0[0, 0, 0].item())
            done_0 = float(pred_done_0[0, 0, 0].item())

        print(f"    Step 0 (real latent at {pos}): "
              f"reward={rew_0:+.4f}, done_logit={done_0:+.4f}")

        # Roll out with optimal actions
        z_rollout = z_current.clone()
        for step_idx, action in enumerate(actions):
            a_tensor = torch.tensor([[action]], dtype=torch.long, device=device)

            with torch.no_grad():
                pred_next, pred_rew, pred_done = model(z_rollout, a_tensor)

            reward_val = float(pred_rew[0, -1, 0].item())
            done_logit = float(pred_done[0, -1, 0].item())
            done_prob = 1.0 / (1.0 + np.exp(-done_logit))  # sigmoid

            # MSE between current dreamed latent and real goal latent
            z_next = pred_next[:, -1:, :]
            mse_to_goal = float(
                torch.mean((z_next.squeeze() - z_goal) ** 2).item()
            )

            is_last = (step_idx == len(actions) - 1)
            marker = " ← SHOULD BE +1.0" if is_last else ""

            print(f"    Step {step_idx + 1} (action={action_names[action]}, "
                  f"WM dreamed): reward={reward_val:+.4f}, "
                  f"done_logit={done_logit:+.4f} (prob={done_prob:.3f}), "
                  f"MSE_to_goal={mse_to_goal:.2f}{marker}")

            # Feed prediction back for next step (autoregressive)
            z_rollout = z_next

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print(f"  reward ≈ +1.0 at final step  → Reward head works near goal")
    print(f"  reward ≈ -0.01 at final step → Reward head FAILS to detect goal")
    print(f"  MSE_to_goal < 1.0            → WM latent is close to real goal")
    print(f"  MSE_to_goal > 10.0           → WM latent has drifted far from goal")
    print(f"  done_prob > 0.5 at final     → Done head fires (may be correct)")
    print(f"  done_prob > 0.5 before final → Done head misfiring (premature)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()