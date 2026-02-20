"""
evaluate_world_model.py  —  Member D: D.3 support script
==========================================================
Loads a trained world model + latent buffer, runs a rollout,
and generates all evaluation plots for the presentation.

Usage:
    python -m src.scripts.evaluate_world_model \
        --checkpoint checkpoints/world_model_best.pt \
        --data_dir   data/processed \
        --log_file   training_log.json \
        --out_dir    evaluation/

Outputs (in evaluation/):
    training_loss.png
    predicted_vs_actual.png
    cosine_over_horizon.png
    per_step_mse.png
    metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.models.transformer import DinoWorldModel
from src.models.transformer_configuration import TransformerWMConfiguration as Config
from src.data.buffer import LatentReplayBuffer
from src.utils.metrics import compute_all_metrics
from src.utils.visualizer import (
    plot_training_loss,
    plot_predicted_vs_actual,
    plot_cosine_over_horizon,
    plot_per_step_mse,
)

SEQ_LEN    = 32
LATENT_DIM = 384
ACTION_DIM = 4


def load_model(checkpoint_path: str, device: torch.device) -> DinoWorldModel:
    config = Config()
    model  = DinoWorldModel(config).to(device)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFO] Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(loss={ckpt['metrics']['avg_loss']:.6f})")
    return model


def load_buffer(data_dir: str) -> LatentReplayBuffer:
    buf   = LatentReplayBuffer(capacity_steps=200_000)
    files = sorted(Path(data_dir).glob("*.npz"))
    for f in files:
        d = np.load(f)
        if "latents" in d:
            buf.add_episode(d["latents"])
    print(f"[INFO] Buffer: {len(buf.episodes)} episodes, {buf.total_steps} steps")
    return buf


def run_rollout(model, buffer, device, batch_size=32):
    """
    Samples one batch of real latents, feeds T-1 steps into the model,
    and collects predicted vs actual for evaluation.

    Returns:
        pred_latents   : (B, T-1, 384)
        actual_latents : (B, T-1, 384)
    """
    latents = buffer.sample(batch_size, seq_len=SEQ_LEN).to(device)  # (B, T, 384)
    actions = torch.randint(0, ACTION_DIM, (batch_size, SEQ_LEN), device=device)

    z_in     = latents[:, :-1]    # (B, T-1, 384)
    a_in     = actions[:, :-1]    # (B, T-1)
    z_target = latents[:, 1:]     # (B, T-1, 384)  — ground truth

    with torch.no_grad():
        pred_next, _, _ = model(z_in, a_in)   # (B, T-1, 384)

    return pred_next.cpu(), z_target.cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir",   type=str, default="data/processed")
    p.add_argument("--log_file",   type=str, default="training_log.json")
    p.add_argument("--out_dir",    type=str, default="evaluation")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)

    print("=" * 56)
    print("  World Model Evaluation (D.3)")
    print("=" * 56)

    # 1. Load model & buffer
    model  = load_model(args.checkpoint, device)
    buffer = load_buffer(args.data_dir)

    # 2. Run rollout
    pred_latents, actual_latents = run_rollout(model, buffer, device, args.batch_size)

    # 3. Compute metrics
    metrics = compute_all_metrics(pred_latents, actual_latents)
    print(f"\n[Metrics]")
    print(f"  MSE               : {metrics['mse']:.6f}")
    print(f"  Cosine Similarity : {metrics['cosine_similarity']:.4f}")

    # 4. Save metrics JSON (for D.4 slide text)
    metrics_path = out_dir / "metrics_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {metrics_path}")

    # 5. Generate all plots
    print("\n[Plots]")
    if Path(args.log_file).exists():
        plot_training_loss(args.log_file, out_dir)
    plot_predicted_vs_actual(pred_latents, actual_latents, out_dir=out_dir)
    plot_cosine_over_horizon(pred_latents, actual_latents, out_dir=out_dir)
    plot_per_step_mse(pred_latents,        actual_latents, out_dir=out_dir)

    print("\n✅ Evaluation complete. All plots saved to:", out_dir)


if __name__ == "__main__":
    main()


## The complete execution order, from zero to presentation

# Step 1 — Collect raw episodes
#   python -m src.scripts.collect_data --episodes 1000

# Step 2 — Encode latents (C.3)
#   python -m src.scripts.encode_dataset

# Step 3 — E.3 smoke test
#   python -m src.scripts.train_world_model --smoke_test

# Step 4 — E.4 full training
#   bash run_full_training.sh

# Step 5 — Generate all evaluation plots (D.3)
#   python -m src.scripts.evaluate_world_model \
#       --checkpoint checkpoints/.../world_model_best.pt

# Step 6 — evaluation/ now contains:
#   training_loss.png          ← loss curve for slides
#   predicted_vs_actual.png    ← D.3 "money shot"
#   cosine_over_horizon.png    ← model quality metric
#   per_step_mse.png           ← step-by-step error
#   metrics_summary.json       ← numbers for slide text