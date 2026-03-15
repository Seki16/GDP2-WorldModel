"""
collect_data_dqn.py  —  Phase 1 data collection with converged DQN policy
==========================================================================
Collects episodes using a mix of DQN policy (goal-reaching transitions)
and random actions (broad state coverage) across multiple maze seeds.

Usage:
    python -m src.scripts.collect_data_dqn \
        --checkpoint checkpoints/dqn_baseline.pt \
        --episodes 1000 \
        --dqn_fraction 0.5 \
        --seeds 0 1 2 3 \
        --out data/raw
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.env.maze_env import MazeEnv, MazeConfig
from src.models.dqn import DQNConfig, PixelDQN


def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    """HxWxC uint8 → CxHxW float32 in [0,1]."""
    return torch.from_numpy(obs).float().permute(2, 0, 1) / 255.0


def load_dqn(checkpoint_path: Path, device: torch.device):
    ckpt     = torch.load(checkpoint_path, map_location=device)
    obs_size = ckpt.get("obs_size", 64)
    cfg      = ckpt.get("dqn_config", {})
    dqn_cfg  = DQNConfig(
        obs_type        = "pixel",
        obs_size        = obs_size,
        n_actions       = cfg.get("n_actions", 4),
        hidden_dim      = cfg.get("hidden_dim", 128),
        conv1_channels  = cfg.get("conv1_channels", 32),
        conv2_channels  = cfg.get("conv2_channels", 64),
        conv3_channels  = cfg.get("conv3_channels", 64),
    )
    model = PixelDQN(dqn_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFO] Loaded DQN checkpoint (obs_size={obs_size})")
    return model


def run_episode(env: MazeEnv, max_steps: int,
                model=None, device=None,
                epsilon: float = 0.05,
                fixed_grid=None) -> tuple:
    """
    Run one episode. If fixed_grid is provided, resets to fixed layout.
    If model is provided, uses epsilon-greedy DQN policy.
    """
    if fixed_grid is not None:
        env.grid      = fixed_grid.copy()
        env.agent_pos = [0, 0]
        env.steps     = 0
        obs           = env._get_obs()
    else:
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out

    obs_list, actions, rewards, dones = [], [], [], []

    for _ in range(max_steps):
        if model is not None and random.random() > epsilon:
            s = preprocess_obs(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = int(torch.argmax(model(s), dim=1).item())
        else:
            action = int(env.action_space.sample())

        step_out = env.step(action)
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, _ = step_out

        obs_list.append(obs)
        actions.append(action)
        rewards.append(float(reward))
        dones.append(bool(done))

        obs = next_obs
        if done:
            break

    return (
        np.stack(obs_list, axis=0).astype(np.uint8),
        np.array(actions,  dtype=np.int64),
        np.array(rewards,  dtype=np.float32),
        np.array(dones,    dtype=bool),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str,   default="checkpoints/dqn_baseline.pt")
    parser.add_argument("--episodes",      type=int,   default=1000)
    parser.add_argument("--dqn_fraction",  type=float, default=0.5,
                        help="Fraction of episodes using DQN policy (rest are random)")
    parser.add_argument("--epsilon",       type=float, default=0.05,
                        help="Epsilon for DQN epsilon-greedy during collection")
    parser.add_argument("--seeds",         type=int,   nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--out",           type=str,   default="data/raw")
    parser.add_argument("--grid_size",     type=int,   default=10)
    parser.add_argument("--max_steps",     type=int,   default=64)
    parser.add_argument("--wall_prob",     type=float, default=0.20)
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_dqn(Path(args.checkpoint), device)

    episodes_per_seed = args.episodes // len(args.seeds)
    dqn_eps_per_seed  = int(episodes_per_seed * args.dqn_fraction)
    rnd_eps_per_seed  = episodes_per_seed - dqn_eps_per_seed

    print(f"\n[INFO] Collecting {args.episodes} episodes across seeds {args.seeds}")
    print(f"[INFO] Per seed: {dqn_eps_per_seed} DQN + {rnd_eps_per_seed} random")

    ep_global   = 0
    goal_count  = 0

    for seed in args.seeds:
        cfg = MazeConfig(
            grid_size = args.grid_size,
            max_steps = args.max_steps,
            obs_size  = 64,
            wall_prob = args.wall_prob,
            seed      = seed,
        )
        env = MazeEnv(cfg)

        # Lock the maze layout once per seed
        reset_out  = env.reset()
        fixed_grid = env.grid.copy()

        schedule = (
            [(model, device, fixed_grid)] * dqn_eps_per_seed +
            [(None,  None,   None)]       * rnd_eps_per_seed
        )

        for use_model, use_device, use_grid in schedule:
            obs, acts, rews, dns = run_episode(
                env, args.max_steps, use_model, use_device,
                args.epsilon, use_grid
            )
            np.savez_compressed(
                out_dir / f"ep_{ep_global:06d}.npz",
                obs=obs, actions=acts, rewards=rews, dones=dns,
            )
            if rews.max() > 0.5:
                goal_count += 1

            ep_global += 1
            if ep_global % 100 == 0:
                print(f"[collect_data_dqn] {ep_global}/{args.episodes} episodes  "
                      f"| goal-reaching so far: {goal_count} "
                      f"({100*goal_count/ep_global:.1f}%)")

    print(f"\n[collect_data_dqn] DONE")
    print(f"  Total episodes     : {ep_global}")
    print(f"  Goal-reaching eps  : {goal_count} ({100*goal_count/ep_global:.1f}%)")
    print(f"  Seeds used         : {args.seeds}")
    print(f"\n  ✅ Proceed to encode_dataset if goal-reaching % > 10%")
    print(f"  ❌ Recheck DQN checkpoint if goal-reaching % < 5%")


if __name__ == "__main__":
    main()