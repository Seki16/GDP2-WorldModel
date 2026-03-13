"""
visualizer.py  —  Member D: Tasks D.3 / D.4
=============================================
All plotting functions needed for the Preliminary Design Review.

Outputs are saved as .png files in an evaluation/ directory.

Functions:
    plot_training_loss()        — from training_log.json  (E.4 output)
    plot_predicted_vs_actual()  — the "money shot" (D.3)
    plot_cosine_over_horizon()  — how fast predictions degrade
    plot_per_step_mse()         — MSE per rollout step
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import numpy as np
import matplotlib.pyplot as plt
import torch

# ── Styling ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size":       11,
})
SAVE_DIR = Path("evaluation")


def _save(fig, name: str, out_dir: Path = SAVE_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)
    return path


# ── D.3 / Plot 1: Training Loss Curve ────────────────────────────────────────

def plot_training_loss(log_file: str = "training_log.json",
                       out_dir: Path = SAVE_DIR) -> Path:
    """
    Reads training_log.json produced by train_world_model.py and
    plots avg loss per epoch.
    """
    with open(log_file) as f:
        log = json.load(f)

    epochs = [entry["epoch"]    for entry in log]
    losses = [entry["avg_loss"] for entry in log]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, linewidth=2, color="#2563EB", label="Avg Loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("World Model Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    return _save(fig, "training_loss.png", out_dir)


# ── D.3 / Plot 2: Predicted vs Actual Latent Paths ───────────────────────────

def plot_predicted_vs_actual(
    pred_latents:   torch.Tensor,
    actual_latents: torch.Tensor,
    sample_idx:     int  = 0,
    dims:           tuple = (0, 1),
    out_dir:        Path  = SAVE_DIR,
) -> Path:
    """
    The "money shot" — D.3.
    Projects one trajectory's predicted vs actual latents onto 2 chosen
    latent dimensions and plots them as time-series lines.

    Args:
        pred_latents   : (B, T, 384) — model rollout predictions
        actual_latents : (B, T, 384) — ground truth latent sequences
        sample_idx     : which sample in the batch to visualise
        dims           : which two latent dims to plot (default 0 and 1)
    """
    pred   = pred_latents  [sample_idx].detach().cpu().numpy()   # (T, 384)
    actual = actual_latents[sample_idx].detach().cpu().numpy()   # (T, 384)
    T      = pred.shape[0]
    steps  = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, dim in zip(axes, dims):
        ax.plot(steps, actual[:, dim], "o-",  color="#16A34A", linewidth=2,
                label="Actual",    markersize=5)
        ax.plot(steps, pred[:, dim],   "s--", color="#DC2626", linewidth=2,
                label="Predicted", markersize=5, alpha=0.85)
        ax.set_title(f"Latent Dimension {dim}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Latent Value")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Predicted vs Actual Latent Trajectory", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "predicted_vs_actual.png", out_dir)


# ── D.3 / Plot 3: Cosine Similarity over Rollout Horizon ─────────────────────

def plot_cosine_over_horizon(
    pred_latents:   torch.Tensor,
    actual_latents: torch.Tensor,
    out_dir:        Path = SAVE_DIR,
) -> Path:
    """
    Shows how cosine similarity between predicted and actual latents
    degrades as the rollout horizon increases.
    A healthy model stays above ~0.8 for several steps.
    """
    from src.utils.metrics import per_step_cosine

    cos_per_step = per_step_cosine(pred_latents, actual_latents).numpy()
    steps = np.arange(len(cos_per_step))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, cos_per_step, "D-", color="#7C3AED", linewidth=2, markersize=6)
    ax.axhline(0.8, linestyle="--", color="gray", alpha=0.6, label="0.8 threshold")
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Prediction Quality over Rollout Horizon")
    ax.legend()
    ax.grid(alpha=0.3)

    return _save(fig, "cosine_over_horizon.png", out_dir)


# ── D.3 / Plot 4: Per-step MSE ────────────────────────────────────────────────

def plot_per_step_mse(
    pred_latents:   torch.Tensor,
    actual_latents: torch.Tensor,
    out_dir:        Path = SAVE_DIR,
) -> Path:
    """
    MSE at each rollout step — should be low early and may grow
    as the model compounds errors over longer horizons.
    """
    from src.utils.metrics import per_step_mse

    mse_per_step = per_step_mse(pred_latents, actual_latents).numpy()
    steps = np.arange(len(mse_per_step))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(steps, mse_per_step, color="#0EA5E9", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("MSE")
    ax.set_title("Per-Step Prediction MSE")
    ax.grid(axis="y", alpha=0.3)

    return _save(fig, "per_step_mse.png", out_dir)