"""
src/scripts/latent_drift_diagnostic.py
========================================
Latent Drift Diagnostic — autoregressive rollout quality check.

Runs a 50-step autoregressive rollout from 10 different real starting
latents, computes MSE between predicted and ground-truth latent at each
step, and reports the horizon at which prediction quality degrades.

Usage
-----
    python -m src.scripts.latent_drift_diagnostic \\
        --checkpoint checkpoints/dqn_dream.pt \\
        --data_dir   data/processed \\
        --out_dir    evaluation/

Outputs
-------
    evaluation/latent_drift.png   — MSE vs step horizon plot
    stdout                        — step threshold where MSE > 2× step-1 MSE

Interface Contract (GDP Plan §2.3)
------------------------------------
  Latent dim  : 384
  Rollout len : 50 steps
  Seeds       : 10 different real starting latents from buffer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Imports ───────────────────────────────────────────────────────────────────

try:
    from src.models.transformer import DinoWorldModel
    from src.models.transformer_configuration import TransformerWMConfiguration as Config
    _REAL_MODEL = True
except ImportError:
    _REAL_MODEL = False
    print("[WARN] src.models.transformer not found — cannot run diagnostic")

try:
    from src.data.buffer import LatentReplayBuffer
    _REAL_BUFFER = True
except ImportError:
    _REAL_BUFFER = False
    print("[WARN] src.data.buffer not found — cannot run diagnostic")

# ── Constants ─────────────────────────────────────────────────────────────────

LATENT_DIM   = 384
ACTION_DIM   = 4
ROLLOUT_STEPS = 50
N_SEEDS      = 10


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> "DinoWorldModel":
    """Load DinoWorldModel weights from a dqn_dream.pt checkpoint."""
    if not _REAL_MODEL:
        raise RuntimeError("src.models.transformer is required for this diagnostic.")

    config = Config()
    model  = DinoWorldModel(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Support checkpoints saved by train_world_model.py (key: "model_state")
    # and any DQN wrapper that may nest it differently
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "world_model_state" in ckpt:
        state_dict = ckpt["world_model_state"]
    else:
        state_dict = ckpt   # bare state dict

    model.load_state_dict(state_dict)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("metrics", {}).get("avg_loss", float("nan"))
    print(f"[INFO] Loaded checkpoint  epoch={epoch}  avg_loss={loss:.6f}")
    return model


# ── Buffer loading ────────────────────────────────────────────────────────────

def load_buffer(data_dir: Path) -> "LatentReplayBuffer":
    """Populate buffer from .npz files in data_dir."""
    if not _REAL_BUFFER:
        raise RuntimeError("src.data.buffer is required for this diagnostic.")

    buf   = LatentReplayBuffer(capacity_steps=200_000)
    files = sorted(data_dir.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    loaded = 0
    for fpath in files:
        data    = np.load(fpath)
        missing = [k for k in ("latents", "actions", "rewards", "dones")
                   if k not in data]
        if missing:
            print(f"[SKIP] {fpath.name} — missing keys: {missing}")
            continue
        buf.add_episode(
            latents = data["latents"],
            actions = data["actions"],
            rewards = data["rewards"],
            dones   = data["dones"],
        )
        loaded += 1

    print(f"[INFO] Loaded {loaded}/{len(files)} episodes  "
          f"total_steps={buf.total_steps}")
    return buf


# ── Starting latent extraction ────────────────────────────────────────────────

def get_seed_episodes(buffer: "LatentReplayBuffer", n_seeds: int, rollout_steps: int):
    """
    Return n_seeds Episode namedtuples that are long enough for a full rollout.
    Each Episode has fields: .latents (T, 384), .actions (T,), .rewards (T,), .dones (T,).
    Access latents via ep.latents, actions via ep.actions, etc.

    Episodes are spaced evenly across the buffer so seeds are diverse.
    Requires at least n_seeds episodes with T >= rollout_steps + 1 steps.

    Returns
    -------
    list of Episode namedtuples, length n_seeds
    """
    candidates = [ep for ep in buffer.episodes
                  if ep.latents.shape[0] >= rollout_steps + 1]

    if len(candidates) < n_seeds:
        raise ValueError(
            f"Need {n_seeds} episodes of length >= {rollout_steps + 1}, "
            f"but only {len(candidates)} qualify."
        )

    # Space evenly across buffer for diversity
    indices = np.linspace(0, len(candidates) - 1, n_seeds, dtype=int)
    return [candidates[i] for i in indices]


# ── Core diagnostic ───────────────────────────────────────────────────────────

def run_drift_diagnostic(
    model:         "DinoWorldModel",
    seed_episodes: list,
    device:        torch.device,
    rollout_steps: int = ROLLOUT_STEPS,
) -> np.ndarray:
    """
    For each seed episode, run an autoregressive rollout for rollout_steps,
    replaying the episode's recorded actions for a fair MSE comparison.

    Returns
    -------
    mse_matrix : np.ndarray (n_seeds, rollout_steps)
        Per-seed, per-step MSE values.
    """
    n_seeds    = len(seed_episodes)
    mse_matrix = np.zeros((n_seeds, rollout_steps), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for s, ep in enumerate(seed_episodes):
            # Ground-truth latents: (rollout_steps+1, 384)
            gt = torch.tensor(
                ep.latents[: rollout_steps + 1],
                dtype=torch.float32,
                device=device,
            )

            # Recorded actions: (rollout_steps,) — replay these exactly
            recorded_actions = torch.tensor(
                ep.actions[: rollout_steps],
                dtype=torch.long,
                device=device,
            )

            # Starting latent z0: shape (1, 1, 384)
            z_current = gt[0].unsqueeze(0).unsqueeze(0)

            for h in range(rollout_steps):
                # Replay recorded action — (1, 1)
                a = recorded_actions[h].unsqueeze(0).unsqueeze(0)

                # One-step forward pass
                pred_next, _, _ = model(z_current, a)   # (1, 1, 384)

                # MSE against ground-truth next latent
                gt_next = gt[h + 1].unsqueeze(0).unsqueeze(0)   # (1, 1, 384)
                mse     = torch.mean((pred_next - gt_next) ** 2).item()
                mse_matrix[s, h] = mse

                # Autoregressive: feed prediction back as next input
                z_current = pred_next

    return mse_matrix


# ── Threshold detection ───────────────────────────────────────────────────────

def find_drift_threshold(mean_mse: np.ndarray) -> int | None:
    """
    Return the first step h where mean_mse[h] > 2 × mean_mse[0].
    Returns None if MSE never exceeds the threshold.
    """
    baseline = mean_mse[0]
    for h, mse in enumerate(mean_mse):
        if mse > 2.0 * baseline:
            return h
    return None


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_drift(
    mse_matrix:  np.ndarray,
    threshold_step: int | None,
    out_dir:     Path,
):
    """
    Plot MSE vs step horizon with per-seed lines, mean, and threshold marker.
    Saves to out_dir/latent_drift.png.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds, rollout_steps = mse_matrix.shape
    steps     = np.arange(1, rollout_steps + 1)
    mean_mse  = mse_matrix.mean(axis=0)
    std_mse   = mse_matrix.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Per-seed lines (faint)
    for s in range(n_seeds):
        ax.plot(steps, mse_matrix[s], color="steelblue", alpha=0.25,
                linewidth=0.8, label="_nolegend_")

    # Mean ± 1 std
    ax.plot(steps, mean_mse, color="steelblue", linewidth=2.5,
            label="Mean MSE (10 seeds)")
    ax.fill_between(steps, mean_mse - std_mse, mean_mse + std_mse,
                    color="steelblue", alpha=0.15, label="±1 std")

    # 2× baseline threshold line
    baseline = mean_mse[0]
    ax.axhline(2.0 * baseline, color="tomato", linestyle="--", linewidth=1.5,
               label=f"2× step-1 MSE ({2.0 * baseline:.4f})")

    # Threshold marker
    if threshold_step is not None:
        ax.axvline(threshold_step + 1, color="tomato", linestyle=":",
                   linewidth=2.0,
                   label=f"Drift threshold: step {threshold_step + 1}")
        ax.scatter([threshold_step + 1], [mean_mse[threshold_step]],
                   color="tomato", zorder=5, s=80)

    ax.set_xlabel("Rollout Step (horizon)", fontsize=12)
    ax.set_ylabel("Latent MSE", fontsize=12)
    ax.set_title("Latent Drift Diagnostic — Autoregressive Rollout MSE vs Horizon",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(1, rollout_steps)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "latent_drift.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Latent Drift Diagnostic — MSE vs rollout horizon"
    )
    p.add_argument("--checkpoint",     type=str, default="checkpoints/dqn_dream.pt")
    p.add_argument("--data_dir",       type=str, default="data/processed")
    p.add_argument("--out_dir",        type=str, default="evaluation")
    p.add_argument("--rollout_steps",  type=int, default=ROLLOUT_STEPS)
    p.add_argument("--n_seeds",        type=int, default=N_SEEDS)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 58)
    print("  Latent Drift Diagnostic")
    print("=" * 58)
    print(f"[INFO] Checkpoint    : {args.checkpoint}")
    print(f"[INFO] Data dir      : {args.data_dir}")
    print(f"[INFO] Rollout steps : {args.rollout_steps}")
    print(f"[INFO] Seeds         : {args.n_seeds}")
    print(f"[INFO] Device        : {device}")

    # 1. Load model and buffer
    model  = load_model(Path(args.checkpoint), device)
    buffer = load_buffer(Path(args.data_dir))

    # 2. Extract diverse seed episodes
    seed_episodes = get_seed_episodes(buffer, args.n_seeds, args.rollout_steps)
    print(f"[INFO] Using {len(seed_episodes)} seed episodes")

    # 3. Run autoregressive rollout and collect MSE
    print(f"\n[INFO] Running {args.rollout_steps}-step autoregressive rollout "
          f"from {args.n_seeds} seeds...")
    mse_matrix = run_drift_diagnostic(
        model, seed_episodes, device,
        rollout_steps=args.rollout_steps,
    )

    # 4. Compute mean MSE and find drift threshold
    mean_mse        = mse_matrix.mean(axis=0)
    threshold_step  = find_drift_threshold(mean_mse)

    # 5. Report
    print("\n─── Drift Report " + "─" * 41)
    if len(mean_mse) > 0:
        print(f"  Step-1  MSE (baseline) : {mean_mse[0]:.6f}")
        
    # Conditionally report additional steps if they are within the rollout.
    for step in (10, 25, 50):
        idx = step - 1  # zero-based index
        if idx < len(mean_mse):
            print(f"  Step-{step:<2} MSE            : {mean_mse[idx]:.6f}")

    if threshold_step is not None:
        print(f"\n  ⚠️  MSE exceeds 2× baseline at step {threshold_step + 1}  "
              f"(MSE={mean_mse[threshold_step]:.6f}  "
              f"threshold={2.0 * mean_mse[0]:.6f})")
    else:
        print(f"\n  ✅  MSE never exceeded 2× baseline across "
              f"{args.rollout_steps} steps — model is stable.")

    # 6. Plot
    plot_drift(mse_matrix, threshold_step, Path(args.out_dir))

    print("\n" + "=" * 58)
    print("  Latent Drift Diagnostic complete.")
    print("=" * 58)


if __name__ == "__main__":
    main()