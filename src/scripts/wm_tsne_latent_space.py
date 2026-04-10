"""
tsne_latent_space.py  —  T-SNE Visualization of World Model Latents
====================================================================

Visualizes real DINOv2 latents vs world-model predicted latents in 
2D T-SNE space to diagnose latent drift and distribution mismatch.

Usage:
    python -m src.scripts.tsne_latent_space \\
        --checkpoint checkpoints/world_model_best.pt \\
        --data_dir   data/processed \\
        --out_file   evaluation/tsne_real_vs_pred.png \\
        --n_samples  5000 \\
        --perplexity 30

Output:
    evaluation/tsne_real_vs_pred.png  — scatter plot with real (blue) vs predicted (red)
    evaluation/tsne_stats.json        — summary statistics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.models.transformer import DinoWorldModel
from src.models.transformer_configuration import TransformerWMConfiguration as Config
from src.data.buffer import LatentReplayBuffer


LATENT_DIM = 384
SEQ_LEN    = 24


def load_model(checkpoint_path: str, device: torch.device) -> DinoWorldModel:
    """Load trained world model from checkpoint."""
    config = Config.from_params(num_layers=8, mlp_ratio=4, num_heads=8, learning_rate=3e-4, sequence_length=24)
    model  = DinoWorldModel(config).to(device)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("metrics", {}).get("avg_loss", "?")
    print(f"[INFO] World model loaded: epoch {epoch}, loss {loss}")
    return model


def load_buffer(data_dir: str) -> LatentReplayBuffer:
    """Load latent buffer from processed .npz files."""
    buf   = LatentReplayBuffer(capacity_steps=500_000)
    files = sorted(Path(data_dir).glob("*.npz"))
    
    for f in files:
        try:
            d = np.load(f)
            missing = [k for k in ("latents", "actions", "rewards", "dones") if k not in d]
            if missing:
                continue
            buf.add_episode(
                latents = d["latents"],
                actions = d["actions"],
                rewards = d["rewards"],
                dones   = d["dones"],
            )
            d.close()
        except Exception as e:
            print(f"[WARN] Failed to load {f.name}: {e}")
            continue
    
    print(f"[INFO] Buffer: {len(buf.episodes)} episodes, {buf.total_steps} steps")
    return buf


def collect_latents(
    model: DinoWorldModel,
    buffer: LatentReplayBuffer,
    device: torch.device,
    n_samples: int = 5000,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Collect real and predicted latents from the buffer and world model.
    
    Returns:
        real_latents  : (N, 384) — real DINOv2 latents from buffer
        pred_latents  : (N, 384) — world-model predictions
        stats         : dict with collection metadata
    """
    real_list = []
    pred_list = []
    
    model.eval()
    batch_size = 32
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"\n[INFO] Collecting {n_samples} latent pairs ({n_batches} batches)...")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Collecting latents"):
            # Sample batch from buffer
            batch = buffer.sample(batch_size, seq_len=seq_len)
            
            z_real  = batch.latents        # (B, T, 384)
            actions = batch.actions.long() # (B, T)
            
            # Split into input and target
            z_in     = z_real[:, :-1]    # (B, T-1, 384)
            a_in     = actions[:, :-1]   # (B, T-1)
            z_target = z_real[:, 1:]     # (B, T-1, 384)
            
            # Predict
            z_in = z_in.to(device)
            a_in = a_in.to(device)
            z_target = z_target.to(device)
            
            pred_next, _, _ = model(z_in, a_in)  # (B, T-1, 384)
            
            # Flatten and collect
            real_flat = z_target.cpu().numpy().reshape(-1, LATENT_DIM)
            pred_flat = pred_next.cpu().numpy().reshape(-1, LATENT_DIM)
            
            real_list.append(real_flat)
            pred_list.append(pred_flat)
    
    real_latents = np.vstack(real_list)[:n_samples]
    pred_latents = np.vstack(pred_list)[:n_samples]
    
    stats = {
        "n_samples":           n_samples,
        "real_mean":           float(np.mean(real_latents)),
        "real_std":            float(np.std(real_latents)),
        "pred_mean":           float(np.mean(pred_latents)),
        "pred_std":            float(np.std(pred_latents)),
        "euclidean_distance":  float(np.mean(np.linalg.norm(
            real_latents - pred_latents, axis=1
        ))),
    }
    
    print(f"[INFO] Real latents  μ={stats['real_mean']:.4f} σ={stats['real_std']:.4f}")
    print(f"[INFO] Pred latents  μ={stats['pred_mean']:.4f} σ={stats['pred_std']:.4f}")
    print(f"[INFO] Mean L2 distance: {stats['euclidean_distance']:.4f}")
    
    return real_latents, pred_latents, stats


def plot_tsne(
    real_latents: np.ndarray,
    pred_latents: np.ndarray,
    out_file: str,
    perplexity: int = 30,
) -> None:
    """
    Compute T-SNE embedding and plot real vs predicted latents with multiple views.
    
    Creates a 2x2 subplot grid showing:
    - Top-left: Combined overlay (transparent, different markers & edge colors)
    - Top-right: Real latents only (blue)
    - Bottom-left: Predicted latents only (red)
    - Bottom-right: Hexbin density heatmap of both
    
    Args:
        real_latents  : (N, 384)
        pred_latents  : (N, 384)
        out_file      : output path
        perplexity    : T-SNE perplexity parameter
    """
    print(f"\n[INFO] Computing T-SNE embedding (perplexity={perplexity})...")
    
    # Stack and apply T-SNE
    combined = np.vstack([real_latents, pred_latents])  # (2N, 384)
    n_real = len(real_latents)
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        verbose=1,
    )
    embedded = tsne.fit_transform(combined)
    
    real_2d = embedded[:n_real]
    pred_2d = embedded[n_real:]
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ── Top-left: Combined overlay with edge colors ───────────────────────────
    ax = axes[0, 0]
    
    # Plot real first (semi-transparent background)
    ax.scatter(
        real_2d[:, 0], real_2d[:, 1],
        label=f"Real DINOv2 (n={len(real_2d)})",
        alpha=0.3, s=40, c="blue", edgecolors="darkblue", linewidth=0.5,
        marker="o"
    )
    # Plot predicted on top (semi-transparent foreground with edge colors)
    ax.scatter(
        pred_2d[:, 0], pred_2d[:, 1],
        label=f"World Model Pred (n={len(pred_2d)})",
        alpha=0.3, s=40, c="red", edgecolors="darkred", linewidth=0.5,
        marker="^"
    )
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title("Combined Overlay\n(Blue circles=Real, Red triangles=Predicted)", 
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # ── Top-right: Real latents only ──────────────────────────────────────────
    ax = axes[0, 1]
    ax.scatter(
        real_2d[:, 0], real_2d[:, 1],
        label=f"Real DINOv2 (n={len(real_2d)})",
        alpha=0.7, s=30, c="blue", edgecolors="darkblue", linewidth=0.3
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title("Real Latents Only", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ── Bottom-left: Predicted latents only ──────────────────────────────────
    ax = axes[1, 0]
    ax.scatter(
        pred_2d[:, 0], pred_2d[:, 1],
        label=f"World Model Pred (n={len(pred_2d)})",
        alpha=0.7, s=30, c="red", edgecolors="darkred", linewidth=0.3
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title("Predicted Latents Only", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ── Bottom-right: Hexbin density plot ────────────────────────────────────
    ax = axes[1, 1]
    
    # Combined heatmap showing density
    hb_all = ax.hexbin(
        np.vstack([real_2d, pred_2d])[:, 0],
        np.vstack([real_2d, pred_2d])[:, 1],
        gridsize=25, cmap="YlOrRd", mincnt=1, edgecolors="face", linewidths=0.2
    )
    
    # Overlay real as semi-transparent blue
    ax.scatter(
        real_2d[:, 0], real_2d[:, 1],
        alpha=0.2, s=15, c="blue", label="Real"
    )
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title("Density Heatmap (Combined)", fontsize=12, fontweight="bold")
    plt.colorbar(hb_all, ax=ax, label="Count")
    ax.legend(fontsize=10)
    
    plt.suptitle(
        f"Real vs Predicted Latent Distributions (T-SNE, perplexity={perplexity})",
        fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved to {out_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="T-SNE visualization of real vs predicted world-model latents"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/world_model_best.pt",
        help="Path to world model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory of processed latent .npz files",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="evaluation/tsne_real_vs_pred.png",
        help="Output plot file",
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        default="evaluation/tsne_stats.json",
        help="Output statistics file",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of latent pairs to sample",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="T-SNE perplexity parameter",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=SEQ_LEN,
        help="Sequence length for buffer sampling",
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    # Load model and buffer
    model = load_model(args.checkpoint, device)
    buffer = load_buffer(args.data_dir)
    
    # Collect latents
    real_latents, pred_latents, stats = collect_latents(
        model, buffer, device,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
    )
    
    # Plot T-SNE
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_tsne(real_latents, pred_latents, str(out_path), perplexity=args.perplexity)
    
    # Save stats
    stats_path = Path(args.stats_file)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Statistics saved to {stats_path}")
    
    print(f"\n[INFO] T-SNE visualization complete.")


if __name__ == "__main__":
    main()