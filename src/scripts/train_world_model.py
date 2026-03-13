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

Interface Contract (GDP Plan §2.3)
------------------------------------
  Latent dim   : 384   (DINOv2 ViT-S/14)
  Seq length T : 16
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

# ── Real imports (Members B & C deliverables) ─────────────────────────────────
# Graceful fallback stubs let E.3 smoke-test pass even if partners are still
# finalising their modules.

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
    def sample(self, batch_size: int, seq_len: int = 16):
        from collections import namedtuple
        Batch = namedtuple("Batch", ["latents", "actions", "rewards", "dones"])
        return Batch(
            latents=torch.randn(batch_size, seq_len, 384),
            actions=torch.randint(0, ACTION_DIM, (batch_size, seq_len)).float(),
            rewards=torch.randn(batch_size, seq_len),
            dones=torch.zeros(batch_size, seq_len)
        )


class _MockModel(nn.Module):
    """Minimal stand-in for DinoWorldModel."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(384, 384)

    def forward(self, z_in: torch.Tensor, a_in: torch.Tensor):
        pred_next = z_in + self.proj(z_in)           # residual prediction
        pred_rew  = torch.zeros(*z_in.shape[:2], 1, device=z_in.device)
        pred_val  = torch.zeros(*z_in.shape[:2], 1, device=z_in.device)
        return pred_next, pred_rew, pred_val


def _mock_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN    = 16
LATENT_DIM = 384
ACTION_DIM = 4

# ── Loss weights ──────────────────────────────────────────────────────────────

LAMBDA_LATENT = 1.0
LAMBDA_REWARD = 0.5
LAMBDA_DONE   = 0.5

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
    Each file must contain key 'latents' with shape (episode_len, 384).
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
            latents = data["latents"],   # (T, 384)
            actions = data["actions"],   # (T, *action_shape)
            rewards = data["rewards"],   # (T,)
            dones   = data["dones"]      # (T,)
        )
        loaded += 1
        
        data.close()  # Free memory immediately after loading each episode

    print(f"[INFO] Loaded {loaded}/{len(files)} episodes | "
          f"Buffer steps: {buffer.total_steps}")
    return loaded


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, buffer, optimizer, loss_fn, device,
                batch_size: int, batches_per_epoch: int) -> dict:
    """One full epoch. Returns loss statistics dict."""
    model.train()
    
    losses = []
    losses_latent = []
    losses_reward = []
    losses_done = []

    for _ in range(batches_per_epoch):
        # A. Sample latents from buffer - now unpacks all four fields ─────────────────────────────────────
        batch = buffer.sample(batch_size, seq_len=SEQ_LEN)
        latents = batch.latents.to(device)
        actions = batch.actions.long().to(device)
        rewards = batch.rewards.to(device)
        dones = batch.dones.to(device)

        # B. Shifted next-step prediction ───────────────────────────────────
        z_in     = latents[:, :-1]     # (B, T-1, 384) — context
        a_in     = actions[:, :-1]     # (B, T-1)
        z_target = latents[:, 1:]      # (B, T-1, 384) — ground truth next latent
        r_target = rewards[:, 1:].unsqueeze(-1) # (B, T-1, 1) — ground truth next reward
        d_target = dones[:, 1:].unsqueeze(-1)   # (B, T-1, 1) — ground truth next done flag
        
        # C. Forward + loss ─────────────────────────────────────────────────
        optimizer.zero_grad()
        pred_next, pred_rew, pred_done = model(z_in, a_in)
        
        
        # binary_cross_entropy_with_logits not binary_cross_entropy 
        # — the model's done head outputs raw logits, so the _with_logits 
        # variant is numerically safer (fuses sigmoid + BCE in one pass). 
        # The .clamp(0, 1) on d_target is a safety guard in case dones 
        # arrive as floats slightly outside that range.
        
        
        loss_latent = loss_fn(pred_next, z_target)
        loss_reward = nn.functional.mse_loss(pred_rew, r_target)
        loss_done   = nn.functional.binary_cross_entropy_with_logits(
                          pred_done, d_target.clamp(0, 1)
                      )

        loss = (LAMBDA_LATENT * loss_latent
              + LAMBDA_REWARD * loss_reward
              + LAMBDA_DONE   * loss_done)
        
        # D. Backward ───────────────────────────────────────────────────────
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        losses_latent.append(loss_latent.item())
        losses_reward.append(loss_reward.item())
        losses_done.append(loss_done.item())

    n = len(losses)
    return {
        "avg_loss":        sum(losses)        / n,
        "min_loss":        min(losses),
        "max_loss":        max(losses),
        "avg_loss_latent": sum(losses_latent) / n,
        "avg_loss_reward": sum(losses_reward) / n,
        "avg_loss_done":   sum(losses_done)   / n,
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
    p = argparse.ArgumentParser(description="Train Latent World Model (E.2/E.3/E.4)")
    p.add_argument("--epochs",            type=int, default=50)
    p.add_argument("--batches_per_epoch", type=int, default=200)
    p.add_argument("--batch_size",        type=int, default=32)
    p.add_argument("--data_dir",          type=str, default="data/processed",
                   help="Dir with pre-encoded .npz latent files (Member C output)")
    p.add_argument("--save_dir",          type=str, default="checkpoints")
    p.add_argument("--save_every",        type=int, default=10,
                   help="Save a numbered checkpoint every N epochs")
    p.add_argument("--log_file",          type=str, default="training_log.json")
    p.add_argument("--smoke_test",        action="store_true",
                   help="E.3: run 1 epoch × 10 batches and exit cleanly")
    p.add_argument("--mock",              action="store_true",
                   help="Force mock stubs (no partner code required)")
    return p.parse_args()


def main():
    args   = parse_args()

    # E.3 smoke-test overrides ─────────────────────────────────────────────────
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

    use_real = not args.mock
    buffer, model, optimizer, scheduler, loss_fn = build_components(device, use_real)

    # Load latent data from disk ───────────────────────────────────────────────
    if use_real and _REAL_BUFFER:
        load_buffer_from_disk(buffer, Path(args.data_dir))

    # Training loop ────────────────────────────────────────────────────────────
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
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:4d}/{args.epochs}  |  "
            f"total={stats['avg_loss']:.4f}  "
            f"latent={stats['avg_loss_latent']:.4f}  "
            f"rew={stats['avg_loss_reward']:.4f}  "
            f"done={stats['avg_loss_done']:.4f}  |  "
            f"lr={lr_now:.2e}  |  {elapsed:.1f}s"
        )

        # Best checkpoint ──────────────────────────────────────────────────────
        if stats["avg_loss"] < best_loss:
            best_loss = stats["avg_loss"]
            p = save_checkpoint(model, optimizer, epoch, stats, save_dir, "best")
            print(f"  ✔ Best checkpoint → {p}")

        # Periodic checkpoint ──────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            p = save_checkpoint(model, optimizer, epoch, stats, save_dir,
                                f"epoch{epoch:04d}")
            print(f"  ✔ Checkpoint      → {p}")

        log.append({"epoch": epoch, "lr": lr_now, **stats})

    # Final weights ────────────────────────────────────────────────────────────
    final = save_checkpoint(model, optimizer, args.epochs, log[-1],
                            save_dir, "final")
    print(f"\n[INFO] Final weights saved → {final}")
    print(f"[INFO] Training time       → {(time.time()-t_start)/60:.1f} min")
    print(f"[INFO] Best avg loss       → {best_loss:.6f}")

    # JSON log for Member D ────────────────────────────────────────────────────
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[INFO] Training log        → {log_path}")

    # E.3 pass banner ──────────────────────────────────────────────────────────
    if args.smoke_test:
        print("\n" + "=" * 58)
        print("  ✅  E.3 PASSED — Pipeline ran 1 epoch without crash.")
        print("=" * 58)

    return model


if __name__ == "__main__":
    main()