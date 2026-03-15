from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src.env.maze_env import MazeEnv, MazeConfig


def run_episode(env: MazeEnv, max_steps: int):
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out

    obs_list = []
    actions = []
    rewards = []
    dones = []

    for _ in range(max_steps):
        action = int(env.action_space.sample())
        step_out = env.step(action)

        # gymnasium: (obs, reward, terminated, truncated, info)
        # gym:       (obs, reward, done, info)
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, info = step_out

        obs_list.append(obs)
        actions.append(action)
        rewards.append(float(reward))
        dones.append(bool(done))

        obs = next_obs
        if done:
            break

    return (
        np.stack(obs_list, axis=0).astype(np.uint8),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=bool),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--out", type=str, default="data/raw")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--wall_prob", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple maze seeds. Overrides --seed. "
                             "Episodes are split evenly across seeds.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds if args.seeds is not None else [args.seed]
    episodes_per_seed = args.episodes // len(seeds)

    ep_global = 0
    for seed in seeds:
        cfg = MazeConfig(
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            obs_size=64,
            wall_prob=args.wall_prob,
            seed=seed,
        )
        env = MazeEnv(cfg)

        for ep in range(episodes_per_seed):
            obs, actions, rewards, dones = run_episode(env, args.max_steps)
            np.savez_compressed(
                out_dir / f"ep_{ep_global:06d}.npz",
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
            )
            ep_global += 1
            if ep_global % 50 == 0:
                print(f"[collect_data] saved {ep_global}/{args.episodes} episodes "
                      f"(seed={seed}) -> {out_dir}")

    print(f"[collect_data] DONE — {ep_global} episodes across seeds {seeds}")


if __name__ == "__main__":
    main()