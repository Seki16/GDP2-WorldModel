"""
metrics.py  —  Member D: Task D.2
===================================
Functions for evaluating World Model prediction quality.

All functions accept plain torch.Tensor inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean Squared Error between predicted and actual next latents.

    Args:
        pred   : (B, T, 384) or (T, 384)
        target : same shape as pred

    Returns:
        scalar float
    """
    return F.mse_loss(pred, target).item()


def cosine_similarity_mean(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean cosine similarity between predicted and actual latent vectors.
    Higher = better (max 1.0).

    Args:
        pred   : (B, T, 384) or (T, 384)
        target : same shape as pred

    Returns:
        scalar float in [-1, 1]
    """
    # Flatten to (N, 384) for batch computation
    p = pred.reshape(-1, pred.shape[-1])
    t = target.reshape(-1, target.shape[-1])
    cos = F.cosine_similarity(p, t, dim=-1)   # (N,)
    return cos.mean().item()


def per_step_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE at each time step (for plotting prediction degradation over horizon).

    Args:
        pred   : (B, T, 384)
        target : (B, T, 384)

    Returns:
        (T,) tensor — one MSE value per timestep
    """
    # MSE per step: mean over batch and latent dim
    return ((pred - target) ** 2).mean(dim=(0, 2))   # (T,)


def per_step_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean cosine similarity at each time step.

    Returns:
        (T,) tensor
    """
    B, T, D = pred.shape
    p = pred.reshape(B * T, D)
    t = target.reshape(B * T, D)
    cos = F.cosine_similarity(p, t, dim=-1).reshape(B, T)
    return cos.mean(dim=0)   # (T,)


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Convenience wrapper — returns all metrics as a dict.

    Usage:
        metrics = compute_all_metrics(pred_latents, target_latents)
        print(metrics)
    """
    return {
        "mse":                  latent_mse(pred, target),
        "cosine_similarity":    cosine_similarity_mean(pred, target),
        "per_step_mse":         per_step_mse(pred, target).tolist(),
        "per_step_cosine":      per_step_cosine(pred, target).tolist(),
    }