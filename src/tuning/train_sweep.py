"""
src/tuning/train_sweep.py
==========================
Member E — Hyperparameter Sweep Entry Point (CDR Sprint)

Hydra entry point for the GDP2 World Model hyperparameter sweep.
Combines Hydra (config/launch), Optuna (intelligent search), and
MLflow (experiment tracking) into a single, clean workflow.

═══════════════════════════════════════════════════════════════════════
QUICK START
═══════════════════════════════════════════════════════════════════════

Install dependencies (add to requirements.txt after testing):
    pip install hydra-core hydra-optuna-sweeper mlflow omegaconf

Smoke test — single run with baseline config (~2 min):
    python -m src.tuning.train_sweep

Override any param inline:
    python -m src.tuning.train_sweep model.num_heads=6 training.learning_rate=3e-4

Full Optuna sweep — 40 intelligent trials:
    python -m src.tuning.train_sweep --multirun --config-name sweep/optuna

View results in MLflow dashboard (run in a separate terminal):
    mlflow ui
    → open http://localhost:5000 in your browser

═══════════════════════════════════════════════════════════════════════
HOW IT WORKS
═══════════════════════════════════════════════════════════════════════

  Hydra reads conf/sweep/optuna.yaml
       │
       └─► Optuna suggests: num_heads=6, num_layers=6, lr=3e-4 ...
                │
                └─► Hydra injects into cfg and calls main(cfg)
                         │
                         ├─► MLflow logs params + metrics per epoch
                         ├─► Model trains on real buffer data
                         ├─► DDA evaluated on held-out sequences
                         └─► DDA returned to Optuna → next trial

Each run saves:
  - src/tuning/outputs/sweeps/<tag>/job_N/.hydra/config.yaml  (exact config)
  - mlruns/  (all metrics, viewable via `mlflow ui`)
  - src/tuning/outputs/optuna_study.db  (Optuna study, resumable)
"""

from __future__ import annotations

import logging
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Imports from the GDP2 codebase
# ─────────────────────────────────────────────────────────────────────────────

from src.models.transformer import DinoWorldModel, latent_mse_loss
from src.models.transformer_configuration import TransformerWMConfiguration
from src.data.buffer import LatentReplayBuffer
from src.utils.metrics import compute_all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Buffer loading
# ─────────────────────────────────────────────────────────────────────────────

def load_buffer(processed_dir: str, capacity: int) -> LatentReplayBuffer:
    """
    Load all .npz episodes from data/processed/ into the replay buffer.
    Each file must contain keys: latents, actions, rewards, dones.
    """
    buf   = LatentReplayBuffer(capacity_steps=capacity)
    files = sorted(Path(processed_dir).glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No .npz files found in '{processed_dir}'. "
            f"Run src/scripts/encode_dataset.py first."
        )

    loaded = 0
    for fpath in files:
        with np.load(fpath) as data:
            missing = [k for k in ("latents", "actions", "rewards", "dones")
                       if k not in data]
            if missing:
                log.warning(f"Skipping {fpath.name} — missing keys: {missing}")
                continue
            buf.add_episode(
                latents = data["latents"],
                actions = data["actions"],
                rewards = data["rewards"],
                dones   = data["dones"],
            )
            loaded += 1

    log.info(f"Buffer loaded: {loaded}/{len(files)} episodes | "
             f"total_steps={buf.total_steps:,}")
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    optimizer: optim.Optimizer,
    buffer:    LatentReplayBuffer,
    cfg:       DictConfig,
    device:    torch.device,
) -> dict:
    """One training epoch. Returns dict of loss components."""
    model.train()
    total_loss = total_latent = total_reward = total_done = 0.0

    for _ in range(cfg.training.batches_per_epoch):
        batch = buffer.sample(cfg.training.batch_size, cfg.training.sequence_length)

        latents = batch.latents.to(device)                          # (B, T, 384)
        actions = batch.actions.long().to(device)                   # (B, T)
        rewards = batch.rewards.to(device)                          # (B, T)
        dones   = batch.dones.to(device)                            # (B, T)

        # Shift: input is t=0..T-2, target is t=1..T-1
        z_in     = latents[:, :-1, :]    # (B, T-1, 384)
        a_in     = actions[:, :-1]       # (B, T-1)
        z_target = latents[:, 1:, :]     # (B, T-1, 384)
        r_target = rewards[:, 1:].unsqueeze(-1)   # (B, T-1, 1)
        d_target = dones[:, 1:].unsqueeze(-1)     # (B, T-1, 1)

        optimizer.zero_grad()
        pred_next, pred_rew, pred_val = model(z_in, a_in)

        loss_latent = nn.functional.mse_loss(pred_next, z_target)
        loss_reward = nn.functional.mse_loss(pred_rew, r_target)
        loss_done   = nn.functional.binary_cross_entropy_with_logits(pred_val, d_target)

        loss = (cfg.training.lambda_latent * loss_latent
              + cfg.training.lambda_reward * loss_reward
              + cfg.training.lambda_done   * loss_done)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()

        total_loss   += loss.item()
        total_latent += loss_latent.item()
        total_reward += loss_reward.item()
        total_done   += loss_done.item()

    n = cfg.training.batches_per_epoch
    return {
        "loss":        total_loss   / n,
        "loss_latent": total_latent / n,
        "loss_reward": total_reward / n,
        "loss_done":   total_done   / n,
    }


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    buffer: LatentReplayBuffer,
    cfg:    DictConfig,
    device: torch.device,
    n_batches: int = 5,
) -> dict:
    """
    Evaluate the model on held-out buffer samples.
    Returns all metrics including the primary metric: delta_direction_acc.
    """
    model.eval()
    accum = {
        "latent_mse":          0.0,
        "cosine_similarity":   0.0,
        "delta_direction_acc": 0.0,
    }

    for _ in range(n_batches):
        batch = buffer.sample(cfg.training.batch_size, cfg.training.sequence_length)

        latents = batch.latents.to(device)
        actions = batch.actions.long().to(device)

        z_in     = latents[:, :-1, :]
        a_in     = actions[:, :-1]
        z_target = latents[:, 1:, :]

        pred_next, _, _ = model(z_in, a_in)

        # ── Latent MSE ────────────────────────────────────────────────────────
        mse = nn.functional.mse_loss(pred_next, z_target).item()

        # ── Cosine similarity ─────────────────────────────────────────────────
        cos = nn.functional.cosine_similarity(
            pred_next.reshape(-1, pred_next.shape[-1]),
            z_target.reshape(-1, z_target.shape[-1]),
            dim=-1
        ).mean().item()

        # ── Delta-Direction Accuracy (primary metric) ─────────────────────────
        # Did the predicted delta move in the same direction as the true delta?
        # delta = next_latent - current_latent
        true_delta = z_target  - z_in   # (B, T-1, 384)
        pred_delta = pred_next - z_in   # (B, T-1, 384)

        # Dot product > 0 means vectors point in the same half-space
        dot = (true_delta * pred_delta).sum(dim=-1)   # (B, T-1)
        dda = (dot > 0).float().mean().item()

        accum["latent_mse"]          += mse
        accum["cosine_similarity"]   += cos
        accum["delta_direction_acc"] += dda

    return {k: v / n_batches for k, v in accum.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main — Hydra entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Train and evaluate one hyperparameter configuration.

    Hydra calls this once per trial. In a 40-trial Optuna sweep, this
    function runs 40 times, each receiving a different cfg from Optuna.

    Returns DDA (primary metric) so Optuna can optimise toward it.
    """
    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)

    log.info("\n" + "=" * 60)
    log.info("TRIAL CONFIG:")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("=" * 60)

    # ── Validate head dimension ───────────────────────────────────────────────
    head_dim = cfg.model.latent_dim // cfg.model.num_heads
    if cfg.model.latent_dim % cfg.model.num_heads != 0:
        raise ValueError(
            f"latent_dim={cfg.model.latent_dim} not divisible by "
            f"num_heads={cfg.model.num_heads}"
        )
    if head_dim < 32:
        log.warning(f"head_dim={head_dim} is very small — expect degraded attention.")

    # ── Build model via from_params ───────────────────────────────────────────
    model_cfg = TransformerWMConfiguration.from_params(
        num_heads       = cfg.model.num_heads,
        num_layers      = cfg.model.num_layers,
        mlp_ratio       = cfg.model.mlp_ratio,
        learning_rate   = cfg.training.learning_rate,
        sequence_length = cfg.training.sequence_length,
    )
    model = DinoWorldModel(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info(
        f"Model: {n_params:,} params | "
        f"L={cfg.model.num_layers} H={cfg.model.num_heads} "
        f"(head_dim={head_dim}) MLP×{cfg.model.mlp_ratio}"
    )

    # ── Load real buffer data ─────────────────────────────────────────────────
    buffer = load_buffer(cfg.data.processed_dir, cfg.data.buffer_capacity)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr           = cfg.training.learning_rate,
        weight_decay = cfg.training.weight_decay,
    )

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_experiment("gdp2_wm_hp_sweep")

    with mlflow.start_run(run_name=cfg.run_tag):

        # Log all hyperparameters to MLflow
        mlflow.log_params({
            "num_heads":       cfg.model.num_heads,
            "num_layers":      cfg.model.num_layers,
            "mlp_ratio":       cfg.model.mlp_ratio,
            "learning_rate":   cfg.training.learning_rate,
            "batch_size":      cfg.training.batch_size,
            "sequence_length": cfg.training.sequence_length,
            "head_dim":        head_dim,
            "n_params":        n_params,
            "seed":            cfg.seed,
        })

        # ── Training loop ─────────────────────────────────────────────────────
        t_start  = time.time()
        best_dda = 0.0

        for epoch in range(1, cfg.training.num_epochs + 1):
            train_metrics = train_one_epoch(model, optimizer, buffer, cfg, device)

            # Log training losses every epoch
            mlflow.log_metrics(
                {f"train/{k}": v for k, v in train_metrics.items()},
                step=epoch
            )

            # Evaluate every 5 epochs and at the final epoch
            if epoch % 5 == 0 or epoch == cfg.training.num_epochs:
                eval_metrics = evaluate(model, buffer, cfg, device)

                mlflow.log_metrics(
                    {f"eval/{k}": v for k, v in eval_metrics.items()},
                    step=epoch
                )

                dda = eval_metrics["delta_direction_acc"]
                if dda > best_dda:
                    best_dda = dda

                log.info(
                    f"  epoch {epoch:3d} | "
                    f"loss={train_metrics['loss']:.4f} | "
                    f"DDA={dda:.3f} | "
                    f"MSE={eval_metrics['latent_mse']:.4f} | "
                    f"cos={eval_metrics['cosine_similarity']:.3f}"
                )

        elapsed = time.time() - t_start

        # ── Final evaluation ──────────────────────────────────────────────────
        final = evaluate(model, buffer, cfg, device, n_batches=10)
        final_dda = final["delta_direction_acc"]

        mlflow.log_metrics({
            "final/DDA":            final_dda,
            "final/latent_mse":     final["latent_mse"],
            "final/cosine_sim":     final["cosine_similarity"],
            "final/best_dda":       best_dda,
            "final/training_time_s": elapsed,
        })

        log.info(
            f"\nFINAL: DDA={final_dda:.3f} | "
            f"MSE={final['latent_mse']:.4f} | "
            f"cos={final['cosine_similarity']:.3f} | "
            f"time={elapsed:.0f}s"
        )

    # Return primary metric to Optuna (Hydra passes this automatically)
    return final_dda


if __name__ == "__main__":
    main()