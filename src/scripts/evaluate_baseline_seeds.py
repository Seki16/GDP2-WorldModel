from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from src.env.maze_env import MazeEnv, MazeConfig
from src.models.dqn import DQNConfig, PixelDQN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(obs).float().permute(2, 0, 1) / 255.0
    return x


def reset_fixed(env: MazeEnv, fixed_grid: np.ndarray) -> np.ndarray:
    env.grid = fixed_grid.copy()
    env.agent_pos = [0, 0]
    env.steps = 0
    return env._get_obs()


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[PixelDQN, int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    obs_size = ckpt.get("obs_size", 64)
    saved_cfg = ckpt.get("dqn_config", {})

    dqn_config = DQNConfig(
        obs_type="pixel",
        obs_size=obs_size,
        n_actions=saved_cfg.get("n_actions", 4),
        hidden_dim=saved_cfg.get("hidden_dim", 128),
        conv1_channels=saved_cfg.get("conv1_channels", 32),
        conv2_channels=saved_cfg.get("conv2_channels", 64),
        conv3_channels=saved_cfg.get("conv3_channels", 64),
    )

    model = PixelDQN(dqn_config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, obs_size


def evaluate_on_seed(
    model: PixelDQN,
    device: torch.device,
    maze_seed: int,
    episodes: int,
    grid_size: int,
    max_steps: int,
    wall_prob: float,
    obs_size: int,
) -> dict:
    cfg = MazeConfig(
        grid_size=grid_size,
        max_steps=max_steps,
        obs_size=obs_size,
        wall_prob=wall_prob,
        seed=maze_seed,
    )
    env = MazeEnv(cfg)

    reset_out = env.reset()
    _ = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    fixed_grid = env.grid.copy()

    returns = []
    steps_list = []
    successes = []

    for _ep in range(episodes):
        obs = reset_fixed(env, fixed_grid)
        s = preprocess_obs(obs).unsqueeze(0).to(device)

        total_reward = 0.0
        steps = 0
        success = False

        while True:
            with torch.no_grad():
                action = int(torch.argmax(model(s), dim=1).item())

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
                success = bool(terminated)
            else:
                obs, reward, done, _info = step_out
                success = bool(reward > 0.5)

            total_reward += float(reward)
            steps += 1

            if done:
                break

            s = preprocess_obs(obs).unsqueeze(0).to(device)

        returns.append(total_reward)
        steps_list.append(steps)
        successes.append(int(success))

    return {
        "agent": "pixel_dqn",
        "seed": maze_seed,
        "episodes": episodes,
        "success_rate": float(100.0 * np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pixel DQN on unseen maze seeds")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dqn_baseline.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--eval_seed", type=int, default=0,
                        help="RNG seed for reproducibility of evaluation code")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--wall_prob", type=float, default=0.20)
    parser.add_argument("--out", type=str, default="evaluation/robustness_seeds.csv")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    model, obs_size = build_model_from_checkpoint(checkpoint_path, device)

    results = []
    print("=" * 60)
    print("  Pixel DQN Robustness Evaluation (A.2 partial)")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device    : {device}")
    print(f"Seeds     : {args.seeds}")
    print(f"Episodes  : {args.episodes}")

    for maze_seed in args.seeds:
        summary = evaluate_on_seed(
            model=model,
            device=device,
            maze_seed=maze_seed,
            episodes=args.episodes,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            wall_prob=args.wall_prob,
            obs_size=obs_size,
        )
        results.append(summary)

        print(
            f"[seed={maze_seed}] "
            f"success={summary['success_rate']:.1f}%  "
            f"mean_return={summary['mean_return']:.4f}  "
            f"mean_steps={summary['mean_steps']:.2f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent",
                "seed",
                "episodes",
                "success_rate",
                "mean_return",
                "std_return",
                "mean_steps",
                "std_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()