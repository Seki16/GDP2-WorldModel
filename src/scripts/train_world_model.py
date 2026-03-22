"""
train_world_model.py  —  Member E: Tasks E.2 / E.3 / E.4
==========================================================
Main training script for the Latent World Model.

Usage
-----
# E.3 smoke-test (1 epoch, no crash verification)
python -m src.scripts.train_world_model --smoke_test

# E.4 full training run
python -m src.scripts.train_world_model --epochs 50 --batches_per_epoch 200

# KL regularisation run (this sprint — E.1)
python -m src.scripts.train_world_model --epochs 50 --batches_per_epoch 200 --kl_weight 0.01

# CDR baseline reproduction (KL disabled)
python -m src.scripts.train_world_model --epochs 50 --batches_per_epoch 200 --kl_weight 0.0

Interface Contract (GDP Plan §2.3)
------------------------------------
  Latent dim   : 384   (DINOv2 ViT-S/14)
  Seq length T : 24
  Latent tensor: (B, T, 384)
  Action tensor: (B, T)  — integer class ids [0..3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# ── KL REGULARISATION (E.1) ───────────────────────────────────────────────────
# Added for the final sprint KL reg fix.
# torch.distributions provides analytically exact KL(Normal || Normal).
from torch.distributions import Normal, kl_divergence

# ── Real imports (Members B & C deliverables) ─────────────────────────────────
try:
    from src.data.buffer import LatentReplayBuffer
    _REAL_BUFFER = True
except ImportError:
    _REAL_BUFFER = False
    print("[WARN] src.data.buffer not found – using MockBuffer")

try:
    from src.models.transformer import DinoWorldModel, latent_mse_loss
    from src.models.transformer_configuration import TransformerWMConfiguration as Config
    _REAL_MODEL = True
except ImportError:
    _REAL_MODEL = False
    print("[WARN] src.models.transformer not found – using MockModel")


# ── Fallback stubs ────────────────────────────────────────────────────────────

class _MockBuffer:
    """Mimics LatentReplayBuffer.sample() when buffer.py is unavailable."""
    def sample(self, batch_size: int, seq_len: int = 24):
        from collections import namedtuple
        Batch = namedtuple("Batch", ["latents", "actions", "rewards", "dones"])
        return Batch(
            latents=torch.randn(batch_size, seq_len, 384),
            actions=torch.randint(0, ACTION_DIM, (batch_size, seq_len)).float(),
            rewards=torch.randn(batch_size, seq_len),
            dones=torch.zeros(batch_size, seq_len)
        )

    # KL REGULARISATION (E.1): mock also needs episodes attribute so that
    # compute_prior_gaussian() doesn't crash in --mock mode.
    episodes = []


class _MockModel(nn.Module):
    """Minimal stand-in for DinoWorldModel."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(384, 384)

    def forward(self, z_in: torch.Tensor, a_in: torch.Tensor):
        pred_next = z_in + self.proj(z_in)
        pred_rew  = torch.zeros(*z_in.shape[:2], 1, device=z_in.device)
        pred_val  = torch.zeros(*z_in.shape[:2], 1, device=z_in.device)
        return pred_next, pred_rew, pred_val


def _mock_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN    = 24
LATENT_DIM = 384
ACTION_DIM = 4

# ── Loss weights ──────────────────────────────────────────────────────────────

LAMBDA_LATENT = 1.0
LAMBDA_REWARD = 0.5
LAMBDA_DONE   = 0.5

# KL REGULARISATION (E.1):
# Default kl_weight (β). Kept small so KL term does not overwhelm the
# reconstruction loss. Overridden at runtime by --kl_weight CLI arg.
# Set to 0.0 to reproduce the CDR baseline exactly.
KL_WEIGHT_DEFAULT = 0.01


# ── KL REGULARISATION (E.1): Prior fitting ────────────────────────────────────

def compute_prior_gaussian(
    buffer,
    device: torch.device,
    max_episodes: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a diagonal Gaussian prior p = Normal(mu, sigma) to the real buffer
    latents. This prior is used as the KL regularisation target during WM
    training — the WM's predicted latent distribution is pulled toward this
    prior, preventing autoregressive drift into out-of-distribution regions.

    Parameters
    ----------
    buffer       : LatentReplayBuffer (or _MockBuffer)
    device       : torch.device
    max_episodes : int — cap to avoid OOM on large buffers

    Returns
    -------
    mu_prior    : (384,) float32 tensor  — per-dimension mean
    std_prior   : (384,) float32 tensor  — per-dimension std (clamped ≥ 1e-4)

    Why diagonal Gaussian?
        A full covariance matrix over 384 dims would be 384×384 = 147k params
        just for the prior. Diagonal is the standard approximation used in
        DreamerV3 and most VAE-style world models. It assumes each latent
        dimension is independent, which is a reasonable approximation for
        DINOv2 CLS tokens.
    """
    episodes = getattr(buffer, "episodes", [])

    if not episodes:
        # No real data available (mock mode or empty buffer).
        # Fall back to standard Normal(0, 1) — a reasonable uninformed prior.
        print("[KL prior] No episodes found — using standard Normal(0,1) prior")
        mu_prior  = torch.zeros(LATENT_DIM, device=device)
        std_prior = torch.ones(LATENT_DIM,  device=device)
        return mu_prior, std_prior

    # Collect latents from up to max_episodes episodes
    all_latents = []
    for ep in episodes[:max_episodes]:
        # ep.latents is a numpy array of shape (T, 384)
        all_latents.append(
            torch.tensor(ep.latents, dtype=torch.float32)
        )

    # Stack into (N_total_steps, 384)
    all_latents = torch.cat(all_latents, dim=0).to(device)

    mu_prior  = all_latents.mean(dim=0)                         # (384,)
    std_prior = all_latents.std(dim=0).clamp(min=1e-4)          # (384,) ≥ 1e-4

    print(f"[KL prior] Fitted from {all_latents.shape[0]:,} real latent steps  |  "
          f"mu mean={mu_prior.mean().item():.4f}  "
          f"std mean={std_prior.mean().item():.4f}")

    return mu_prior, std_prior


def kl_loss_fn(
    pred_next:  torch.Tensor,
    mu_prior:   torch.Tensor,
    std_prior:  torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean KL divergence between the WM's predicted latent distribution
    and the real-data prior:

        KL( Normal(mu_q, sigma_q) || Normal(mu_prior, sigma_prior) )

    where mu_q and sigma_q are the empirical mean and std of pred_next across
    the batch dimension (treating each step as an independent sample).

    Parameters
    ----------
    pred_next  : (B, T-1, 384) — WM predicted next latents
    mu_prior   : (384,)        — prior mean fitted from real buffer
    std_prior  : (384,)        — prior std fitted from real buffer

    Returns
    -------
    kl_loss : scalar tensor

    Why this formulation?
        We treat the batch of WM predictions as samples from the WM's implicit
        output distribution q. We fit a diagonal Gaussian to those samples
        (empirical mean + std) and measure KL against the real-data prior p.
        This penalises the WM whenever its predicted latents drift away from
        the region where real DINOv2 latents live — which is precisely the
        mechanism that causes autoregressive drift.
    """
    # Flatten to (B*(T-1), 384) — treat every predicted step as one sample
    B, T_minus1, D = pred_next.shape
    flat = pred_next.reshape(-1, D)  # (N, 384)

    # Empirical mean and std of WM predictions
    mu_q  = flat.mean(dim=0)                   # (384,)
    std_q = flat.std(dim=0).clamp(min=1e-4)    # (384,)

    # Build distributions
    q = Normal(mu_q,      std_q)
    p = Normal(mu_prior,  std_prior)

    # KL(q || p): shape (384,), then mean over dimensions
    kl = kl_divergence(q, p).mean()

    return kl


# ── Components factory ────────────────────────────────────────────────────────

def build_components(device: torch.device, use_real: bool):
    """Return (buffer, model, optimizer, scheduler, loss_fn)."""
    buffer  = LatentReplayBuffer(capacity_steps=200_000) if (use_real and _REAL_BUFFER) \
              else _MockBuffer()

    if use_real and _REAL_MODEL:
        config  = Config()
        model   = DinoWorldModel(config).to(device)
        loss_fn = latent_mse_loss
    else:
        model   = _MockModel().to(device)
        loss_fn = _mock_mse

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    return buffer, model, optimizer, scheduler, loss_fn


def load_buffer_from_disk(buffer: "LatentReplayBuffer", processed_dir: Path) -> int:
    """
    Populate buffer from .npz latent files produced by Member C's pipeline.
    Returns number of episodes loaded.
    """
    import numpy as np

    files = sorted(processed_dir.glob("*.npz"))
    if not files:
        print(f"[WARN] No .npz files found in {processed_dir}")
        return 0

    loaded = 0

    for fpath in files:
        data = np.load(fpath)
        if "latents" not in data:
            print(f"[SKIP] {fpath.name} — missing 'latents' key")
            continue

        missing = [k for k in ("actions", "rewards", "dones") if k not in data]
        if missing:
            print(f"[SKIP] {fpath.name} — missing keys: {missing}")
            continue

        buffer.add_episode(
            latents = data["latents"],
            actions = data["actions"],
            rewards = data["rewards"],
            dones   = data["dones"]
        )
        loaded += 1

        data.close()

    print(f"[INFO] Loaded {loaded}/{len(files)} episodes | "
          f"Buffer steps: {buffer.total_steps}")
    return loaded


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(
    model, buffer, optimizer, loss_fn, device,
    batch_size: int,
    batches_per_epoch: int,
    # KL REGULARISATION (E.1): new parameters
    kl_weight:  float,
    mu_prior:   torch.Tensor,
    std_prior:  torch.Tensor,
) -> dict:
    """
    One full epoch. Returns loss statistics dict.

    KL REGULARISATION (E.1):
        kl_weight  — β coefficient. 0.0 disables KL term entirely (CDR baseline).
        mu_prior   — (384,) prior mean fitted from real buffer before training.
        std_prior  — (384,) prior std  fitted from real buffer before training.
    """
    model.train()

    losses        = []
    losses_latent = []
    losses_reward = []
    losses_done   = []
    # KL REGULARISATION (E.1): track KL loss separately for logging
    losses_kl     = []

    for _ in range(batches_per_epoch):
        # A. Sample latents from buffer ────────────────────────────────────────
        batch   = buffer.sample(batch_size, seq_len=SEQ_LEN)
        latents = batch.latents.to(device)
        actions = batch.actions.long().to(device)
        rewards = batch.rewards.to(device)
        dones   = batch.dones.to(device)

        # B. Shifted next-step prediction ─────────────────────────────────────
        z_in     = latents[:, :-1]                  # (B, T-1, 384)
        a_in     = actions[:, :-1]                  # (B, T-1)
        z_target = latents[:, 1:]                   # (B, T-1, 384)
        r_target = rewards[:, 1:].unsqueeze(-1)     # (B, T-1, 1)
        d_target = dones[:, 1:].unsqueeze(-1)       # (B, T-1, 1)

        # C. Forward + loss ────────────────────────────────────────────────────
        optimizer.zero_grad()
        pred_next, pred_rew, pred_done = model(z_in, a_in)

        loss_latent = loss_fn(pred_next, z_target)
        loss_reward = nn.functional.mse_loss(pred_rew, r_target)
        loss_done   = nn.functional.binary_cross_entropy_with_logits(
                          pred_done, d_target.clamp(0, 1)
                      )

        # KL REGULARISATION (E.1): ─────────────────────────────────────────────
        # Compute KL between WM predicted latent distribution and real prior.
        # When kl_weight=0.0 this branch is skipped entirely — zero overhead
        # for the CDR baseline reproduction run.
        if kl_weight > 0.0:
            loss_kl = kl_loss_fn(pred_next, mu_prior, std_prior)
        else:
            loss_kl = torch.tensor(0.0, device=device)

        loss = (LAMBDA_LATENT * loss_latent
              + LAMBDA_REWARD * loss_reward
              + LAMBDA_DONE   * loss_done
              + kl_weight     * loss_kl)       # KL REGULARISATION (E.1)

        # D. Backward ──────────────────────────────────────────────────────────
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        losses_latent.append(loss_latent.item())
        losses_reward.append(loss_reward.item())
        losses_done.append(loss_done.item())
        losses_kl.append(loss_kl.item())       # KL REGULARISATION (E.1)

    n = len(losses)
    return {
        "avg_loss":        sum(losses)        / n,
        "min_loss":        min(losses),
        "max_loss":        max(losses),
        "avg_loss_latent": sum(losses_latent) / n,
        "avg_loss_reward": sum(losses_reward) / n,
        "avg_loss_done":   sum(losses_done)   / n,
        "avg_loss_kl":     sum(losses_kl)     / n,   # KL REGULARISATION (E.1)
    }


def save_checkpoint(model, optimizer, epoch: int, metrics: dict,
                    save_dir: Path, tag: str = "last") -> Path:
    """Save model + optimizer state to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics":     metrics,
    }
    path = save_dir / f"world_model_{tag}.pt"
    torch.save(ckpt, path)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Latent World Model (E.1/E.2/E.3/E.4)")
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batches_per_epoch", type=int,   default=200)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--data_dir",          type=str,   default="data/processed")
    p.add_argument("--save_dir",          type=str,   default="checkpoints")
    p.add_argument("--save_every",        type=int,   default=10)
    p.add_argument("--log_file",          type=str,   default="training_log.json")
    p.add_argument("--smoke_test",        action="store_true")
    p.add_argument("--mock",              action="store_true")
    # KL REGULARISATION (E.1): ─────────────────────────────────────────────────
    # β coefficient for KL loss term.
    # 0.01  — default for the KL reg experiment run (E.1).
    # 0.0   — reproduces CDR baseline exactly (KL term disabled).
    # Try 0.001 and 0.1 if 0.01 does not reduce drift.
    p.add_argument("--kl_weight",         type=float, default=KL_WEIGHT_DEFAULT,
                   help="KL regularisation weight β. 0.0 = CDR baseline (no KL).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        print("=" * 58)
        print("  E.3 SMOKE TEST — 1 epoch, 10 batches (integration check)")
        print("=" * 58)
        args.epochs            = 1
        args.batches_per_epoch = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device        : {device}")
    print(f"[INFO] Epochs        : {args.epochs}")
    print(f"[INFO] Batches/epoch : {args.batches_per_epoch}")
    print(f"[INFO] Real buffer   : {_REAL_BUFFER and not args.mock}")
    print(f"[INFO] Real model    : {_REAL_MODEL and not args.mock}")
    # KL REGULARISATION (E.1)
    print(f"[INFO] KL weight (β) : {args.kl_weight}"
          + ("  ← KL DISABLED (CDR baseline)" if args.kl_weight == 0.0
             else "  ← KL ENABLED (E.1 fix)"))

    use_real = not args.mock
    buffer, model, optimizer, scheduler, loss_fn = build_components(device, use_real)

    if use_real and _REAL_BUFFER:
        load_buffer_from_disk(buffer, Path(args.data_dir))

    # KL REGULARISATION (E.1): fit prior from real buffer BEFORE training starts.
    # This is done once — the prior is fixed for the entire training run.
    # Fitting happens after the buffer is loaded so we use the actual data.
    mu_prior, std_prior = compute_prior_gaussian(buffer, device)

    save_dir  = Path(args.save_dir)
    log       = []
    best_loss = float("inf")

    print("\n─── Training Start " + "─" * 40)
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        stats = train_epoch(
            model, buffer, optimizer, loss_fn,
            device, args.batch_size, args.batches_per_epoch,
            # KL REGULARISATION (E.1)
            kl_weight = args.kl_weight,
            mu_prior  = mu_prior,
            std_prior = std_prior,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:4d}/{args.epochs}  |  "
            f"total={stats['avg_loss']:.4f}  "
            f"latent={stats['avg_loss_latent']:.4f}  "
            f"rew={stats['avg_loss_reward']:.4f}  "
            f"done={stats['avg_loss_done']:.4f}  "
            # KL REGULARISATION (E.1): KL loss printed every epoch
            f"kl={stats['avg_loss_kl']:.4f}  |  "
            f"lr={lr_now:.2e}  |  {elapsed:.1f}s"
        )

        if stats["avg_loss"] < best_loss:
            best_loss = stats["avg_loss"]
            p = save_checkpoint(model, optimizer, epoch, stats, save_dir, "best")
            print(f"  ✔ Best checkpoint → {p}")

        if epoch % args.save_every == 0:
            p = save_checkpoint(model, optimizer, epoch, stats, save_dir,
                                f"epoch{epoch:04d}")
            print(f"  ✔ Checkpoint      → {p}")

        log.append({"epoch": epoch, "lr": lr_now, **stats})

    final = save_checkpoint(model, optimizer, args.epochs, log[-1],
                            save_dir, "final")
    print(f"\n[INFO] Final weights saved → {final}")
    print(f"[INFO] Training time       → {(time.time()-t_start)/60:.1f} min")
    print(f"[INFO] Best avg loss       → {best_loss:.6f}")

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[INFO] Training log        → {log_path}")

    if args.smoke_test:
        print("\n" + "=" * 58)
        print("  ✅  E.3 PASSED — Pipeline ran 1 epoch without crash.")
        print("=" * 58)

    return model


if __name__ == "__main__":
    main()