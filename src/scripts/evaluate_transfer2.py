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
    --dqn_weights   checkpoints/dqn_baseline.pt \\
    --dream_weights checkpoints/dqn_dream.pt \\
    --episodes      50

# With fine-tuned encoder (after joint training):
python -m src.scripts.evaluate_transfer \\
    --dqn_weights        checkpoints/dqn_baseline.pt \\
    --dream_weights      checkpoints/dqn_dream.pt \\
    --encoder_checkpoint checkpoints/encoder_finetuned.pt \\
    --episodes           50

# Quick smoke-test (5 episodes per agent)
python -m src.scripts.evaluate_transfer --smoke_test

─────────────────────────────────────────────────────────────
CRITICAL FIX — Member B (Fine-tuned Encoder, Final Sprint)
─────────────────────────────────────────────────────────────
The original script always loaded the frozen DINOv2 encoder for
Dream DQN transfer. After joint training, the Dream DQN was
trained on latents from the FINE-TUNED encoder — feeding it
frozen DINOv2 latents at transfer time puts it in a completely
different latent space, causing 0% transfer and -6.4 return
(wall-hitting every step).

Fix: added --encoder_checkpoint flag. When provided, the
fine-tuned encoder is loaded for Dream DQN evaluation.
The Pixel DQN is unaffected (it uses raw pixels, not latents).

New flag:
  --encoder_checkpoint : path to encoder_finetuned.pt
                         If not provided, uses frozen DINOv2
                         (original behaviour, for non-joint runs).
─────────────────────────────────────────────────────────────
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
import torch.nn.functional as F

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
# Encoder loading
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet normalisation constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_encoder(
    device:               torch.device,
    encoder_checkpoint:   str | None = None,
) -> tuple[nn.Module, bool]:
    """
    Load DINOv2 backbone for Dream DQN transfer encoding.

    If encoder_checkpoint is provided, loads the fine-tuned backbone
    from train_world_model_joint.py. Otherwise loads frozen DINOv2.

    CRITICAL: The encoder used at transfer time MUST match the encoder
    used during Dream DQN training. Using frozen DINOv2 when the Dream
    DQN was trained on fine-tuned latents causes complete transfer failure
    (0% success, -6.4 return) because the latent spaces are different.

    Args:
        device             : torch device
        encoder_checkpoint : path to encoder_finetuned.pt, or None

    Returns:
        backbone   : DINOv2 backbone in eval mode
        is_finetuned : True if fine-tuned encoder was loaded
    """
    if encoder_checkpoint is None:
        print("[INFO] Encoder: frozen DINOv2 (standard pipeline)")
        encoder  = DinoV2Encoder(device=device)
        return encoder.backbone, False

    ckpt_path = Path(encoder_checkpoint)
    if not ckpt_path.exists():
        print(f"[WARN] encoder_checkpoint not found at {ckpt_path}")
        print(f"[WARN] Falling back to frozen DINOv2 — transfer may fail if")
        print(f"[WARN] Dream DQN was trained with joint training!")
        encoder = DinoV2Encoder(device=device)
        return encoder.backbone, False

    print(f"[INFO] Encoder: fine-tuned ({ckpt_path})")
    encoder  = DinoV2Encoder(device=device)
    backbone = encoder.backbone

    ckpt = torch.load(ckpt_path, map_location=device)
    encoder_state = ckpt["encoder_state"]

    # Extract backbone weights from FineTunableDinoV2 state dict
    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in encoder_state.items()
        if k.startswith("backbone.")
    }

    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if unexpected:
        print(f"[WARN] Unexpected keys in backbone state: {unexpected}")

    epoch = ckpt.get("epoch", "?")
    print(f"[INFO] Fine-tuned encoder loaded (epoch={epoch})")
    backbone.eval().to(device)
    return backbone, True


def encode_obs(
    obs:      np.ndarray,
    backbone: nn.Module,
    device:   torch.device,
    mean:     torch.Tensor,
    std:      torch.Tensor,
) -> torch.Tensor:
    """
    Encode a single (H, W, 3) uint8 observation into a (384,) latent.

    Uses the same preprocessing as encode_dataset.py and the joint
    training script to ensure consistency.

    Args:
        obs      : (H, W, 3) uint8 numpy array
        backbone : DINOv2 backbone (frozen or fine-tuned)
        device   : torch device
        mean/std : ImageNet normalisation constants on device

    Returns:
        z : (384,) float32 tensor on device
    """
    x = torch.from_numpy(obs).to(device)            # (H, W, 3) uint8
    x = x.permute(2, 0, 1).float().unsqueeze(0)     # (1, 3, H, W)
    x = x / 255.0
    x = (x - mean) / std
    x = F.interpolate(x, size=(224, 224),
                      mode="bilinear", align_corners=False)
    with torch.no_grad():
        z = backbone(x).squeeze(0)                  # (384,)
    return z


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
        grid_size = args.grid_size,
        max_steps = args.max_steps,
        obs_size  = 64,
        wall_prob = args.wall_prob,
        seed      = args.maze_seed,
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
    env:        MazeEnv,
    fixed_grid: np.ndarray,
    model:      TinyCNN,
    device:     torch.device,
) -> dict:
    """Run one greedy episode with the pixel DQN in the real MazeEnv."""
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
            success = bool(terminated)
        else:
            obs, reward, done, _ = step_out
            success = reward > 0.5

        total_reward += float(reward)
        steps        += 1

        if done:
            break

        s = preprocess_pixel_obs(obs).to(device)

    return {"return": total_reward, "steps": steps, "success": int(success)}


def run_episode_dream_dqn(
    env:          MazeEnv,
    fixed_grid:   np.ndarray,
    model:        DreamQNet,
    backbone:     nn.Module,
    device:       torch.device,
    mean:         torch.Tensor,
    std:          torch.Tensor,
) -> dict:
    """
    Run one greedy episode with the dream DQN in the REAL MazeEnv.

    The key transfer step: encodes real pixel observations with the
    SAME encoder used during Dream DQN training (frozen or fine-tuned),
    then feeds the latent to the Dream DQN Q-network.

    IMPORTANT: backbone must match the encoder used during Dream DQN
    training. Using a different encoder produces latents in a different
    space, causing complete transfer failure.
    """
    obs          = reset_fixed(env, fixed_grid)
    total_reward = 0.0
    steps        = 0
    success      = False

    for _ in range(env.config.max_steps):
        # Encode real obs with the correct encoder
        z   = encode_obs(obs, backbone, device, mean, std)  # (384,)
        z_t = z.unsqueeze(0)                                # (1, 384)

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
    p.add_argument("--dqn_weights",   type=str, default="checkpoints/dqn_baseline.pt")
    p.add_argument("--dream_weights", type=str, default="checkpoints/dqn_dream.pt")
    p.add_argument("--episodes",      type=int, default=50)
    p.add_argument("--maze_seed",     type=int, default=0)
    p.add_argument("--grid_size",     type=int, default=10)
    p.add_argument("--max_steps",     type=int, default=64)
    p.add_argument("--wall_prob",     type=float, default=0.20)
    p.add_argument("--out_csv",       type=str,
                   default="evaluation/transfer_results.csv")
    p.add_argument(
        "--encoder_checkpoint", type=str, default=None,
        help=(
            "Path to fine-tuned encoder checkpoint (checkpoints/encoder_finetuned.pt). "
            "REQUIRED if Dream DQN was trained with joint DINOv2+WM training. "
            "Using wrong encoder causes 0%% transfer — latent spaces won't match."
        ),
    )
    p.add_argument("--smoke_test", action="store_true",
                   help="5 episodes per agent for quick sanity check")
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
    print(f"[INFO] Maze seed  : {args.maze_seed}")
    print(f"[INFO] Episodes   : {args.episodes} per agent")

    # ── Build env with fixed maze ─────────────────────────────────────────────
    env, fixed_grid = make_env(args)

    # ── Load encoder for Dream DQN transfer ───────────────────────────────────
    backbone, is_finetuned = load_encoder(device, args.encoder_checkpoint)
    mean = IMAGENET_MEAN.to(device)
    std  = IMAGENET_STD.to(device)

    if not is_finetuned and args.encoder_checkpoint is None:
        print("[WARN] No --encoder_checkpoint provided.")
        print("[WARN] If Dream DQN was trained with joint training,")
        print("[WARN] transfer will fail. Pass --encoder_checkpoint")
        print("[WARN] checkpoints/encoder_finetuned.pt to fix this.")

    all_results = []
    summaries   = []

    # ── Agent 1: Pixel DQN ────────────────────────────────────────────────────
    dqn_path = Path(args.dqn_weights)
    if dqn_path.exists():
        print(f"\n[INFO] Loading Pixel DQN from {dqn_path}")
        ckpt      = torch.load(dqn_path, map_location=device)
        obs_size  = ckpt.get("obs_size", 64)
        from src.models.dqn import DQNConfig, PixelDQN
        cfg = ckpt.get("dqn_config", {})
        dqn_config = DQNConfig(
            obs_type       = "pixel",
            obs_size       = obs_size,
            n_actions      = cfg.get("n_actions", 4),
            hidden_dim     = cfg.get("hidden_dim", 128),
            conv1_channels = cfg.get("conv1_channels", 32),
            conv2_channels = cfg.get("conv2_channels", 64),
            conv3_channels = cfg.get("conv3_channels", 64),
        )
        pixel_dqn = PixelDQN(dqn_config).to(device)
        pixel_dqn.load_state_dict(ckpt.get("model_state", ckpt))
        pixel_dqn.eval()

        pixel_fn  = lambda: run_episode_pixel_dqn(
            env, fixed_grid, pixel_dqn, device
        )
        results_p = evaluate_agent(
            "Pixel DQN (real env trained)", pixel_fn, args.episodes
        )
        all_results.extend(results_p)
        summaries.append(summarise(results_p))
    else:
        print(f"\n[WARN] Pixel DQN not found at {dqn_path} — skipping.")

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
        print(f"[INFO] Dream DQN: {ckpt.get('steps_trained', '?')} steps trained  "
              f"| dream success: {ckpt.get('final_success_pct', 0.0):.1f}%")
        print(f"[INFO] Encoder  : {'fine-tuned' if is_finetuned else 'frozen DINOv2'}")

        dream_fn = lambda: run_episode_dream_dqn(
            env, fixed_grid, dream_dqn, backbone, device, mean, std
        )
        results_d = evaluate_agent(
            "Dream DQN (world-model trained)", dream_fn, args.episodes
        )
        all_results.extend(results_d)
        summaries.append(summarise(results_d))
    else:
        print(f"\n[WARN] Dream DQN not found at {dream_path} — skipping.")

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
        print(f"  TRANSFER EXPERIMENT RESULTS")
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
        print(f"  Transfer gap (Pixel − Dream): {gap:+.1f}%")
        print(f"{'='*60}\n")

        if abs(gap) < 10:
            print("  ✔ Small gap — world model is a good simulator substitute.")
        elif d_sum["success_rate_pct"] > p_sum["success_rate_pct"]:
            print("  ✔ Dream DQN outperforms pixel DQN.")
        else:
            print(f"  ✘ Gap of {gap:.1f}% — transfer gap persists.")


if __name__ == "__main__":
    main()