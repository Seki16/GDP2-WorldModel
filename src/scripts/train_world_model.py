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

# Member B: KL regularisation + rollout loss run
python -m src.scripts.train_world_model --epochs 150 --batches_per_epoch 200 \
    --kl_weight 5e-3 --rollout_steps 4 --rollout_weight 0.5

Interface Contract (GDP Plan §2.3)
------------------------------------
  Latent dim   : 384   (DINOv2 ViT-S/14)
  Seq length T : 24
  Latent tensor: (B, T, 384)
  Action tensor: (B, T)  — integer class ids [0..3]

─────────────────────────────────────────────────────────────
MODIFICATION — Member B (KL Regularisation, Final Sprint)
─────────────────────────────────────────────────────────────
Added KL divergence regularisation loss to train_epoch().

The KL loss fits a diagonal Gaussian prior to the real buffer
latents in each batch and penalises the predicted latents for
diverging from that distribution. This targets the root cause
of autoregressive drift: the world model learns to produce
latents that stray from the real DINOv2 distribution at
training time, causing compounding errors at rollout.

─────────────────────────────────────────────────────────────
MODIFICATION — Member B (Rollout Loss, Final Sprint)
─────────────────────────────────────────────────────────────
Added multi-step rollout loss to train_epoch().

Standard teacher forcing trains the WM on one-step predictions
using real latents as input at every step. This means the model
never learns to handle its own prediction errors, causing them
to compound catastrophically at rollout time (autoregressive
latent drift).

Rollout loss unrolls the WM for rollout_steps steps, feeding
predicted latents back as input rather than ground truth. The
MSE loss is computed at every intermediate step and summed.
This forces the model to be robust to its own errors, directly
attacking the exponential drift growth rate.

New CLI flags:
  --kl_weight β        (float, default 1e-3) — KL loss weight
  --rollout_steps N    (int,   default 4)    — unroll steps
  --rollout_weight λ   (float, default 0.5)  — rollout loss weight

Loss structure (full):
  loss = λ_latent  * loss_latent        (teacher-forced, 1-step)
       + λ_reward  * loss_reward
       + λ_done    * loss_done
       + β         * loss_kl
       + λ_rollout * loss_rollout       (multi-step, own predictions)
─────────────────────────────────────────────────────────────
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
LAMBDA_DONE   = 0.1
# KL weight (β) is passed in per-run via args.kl_weight — not a global constant,
# so different members can use different values without touching shared code.


# ── KL Divergence Loss ────────────────────────────────────────────────────────

def kl_divergence_loss(
    pred_latents: torch.Tensor,
    real_latents: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence regularisation loss (Member B — Final Sprint).

    Fits a diagonal Gaussian prior N(μ_real, σ²_real) to the real buffer
    latents in the current batch, then computes the KL divergence from the
    predicted latent distribution N(μ_pred, σ²_pred) to that prior.

    This penalises the world model for generating predicted latents that
    stray from the statistical distribution of real DINOv2 latents,
    directly targeting the root cause of autoregressive latent drift.

    KL(N(μ_pred, σ²_pred) || N(μ_real, σ²_real))
      = 0.5 * Σ [ log(σ²_real/σ²_pred)
                  + σ²_pred/σ²_real
                  + (μ_pred - μ_real)²/σ²_real
                  - 1 ]

    Both distributions are computed per latent dimension, then averaged.

    Args:
        pred_latents : (B, T, 384) — world model predicted next latents
        real_latents : (B, T, 384) — ground truth next latents from buffer

    Returns:
        Scalar KL divergence loss (mean over batch, time, and latent dims).
    """
    # Flatten to (N, 384) for per-dimension statistics
    pred_flat = pred_latents.reshape(-1, pred_latents.shape[-1])  # (N, 384)
    real_flat = real_latents.reshape(-1, real_latents.shape[-1])  # (N, 384)

    # Fit diagonal Gaussian to real buffer latents (the prior)
    mu_real  = real_flat.mean(dim=0)                  # (384,)
    var_real = real_flat.var(dim=0).clamp(min=1e-6)   # (384,) — clamp avoids div/0

    # Fit diagonal Gaussian to predicted latents
    mu_pred  = pred_flat.mean(dim=0)                  # (384,)
    var_pred = pred_flat.var(dim=0).clamp(min=1e-6)   # (384,)

    # Closed-form KL(N(μ_pred, σ²_pred) || N(μ_real, σ²_real))
    # per dimension, then mean over all 384 dims
    kl_per_dim = 0.5 * (
        torch.log(var_real / var_pred)              # log variance ratio
        + var_pred / var_real                       # variance ratio
        + (mu_pred - mu_real) ** 2 / var_real       # mean shift penalty
        - 1.0                                       # normalisation constant
    )

    return kl_per_dim.mean()


# ── Rollout Loss ─────────────────────────────────────────────────────────────

def rollout_loss(
    model,
    z_start:       torch.Tensor,
    a_seq:         torch.Tensor,
    z_targets:     torch.Tensor,
    rollout_steps: int,
    loss_fn,
) -> torch.Tensor:
    """
    Multi-step rollout loss (Member B — Final Sprint).

    Unrolls the world model for rollout_steps steps using its own
    predicted latents as input (no teacher forcing). Computes MSE
    against ground truth at every step and returns the mean.

    This forces the model to be robust to its own prediction errors,
    directly targeting the exponential drift growth rate that KL
    regularisation alone cannot fix.

    Standard teacher forcing:
        z_0 (real) → pred z_1,  input z_1 (real) → pred z_2, ...
        Model never sees its own errors during training.

    Rollout loss:
        z_0 (real) → pred z_1,  input z_1 (pred) → pred z_2, ...
        Model must handle compounding errors at training time.

    Args:
        model         : DinoWorldModel (in train mode)
        z_start       : (B, 1, 384) — real starting latent z_0
        a_seq         : (B, T, ) — action sequence for the rollout
        z_targets     : (B, T, 384) — ground truth latents for each step
        rollout_steps : number of steps to unroll (≤ T)
        loss_fn       : latent MSE loss function

    Returns:
        Scalar mean rollout MSE across all unrolled steps.
    """
    # Cap rollout_steps to available sequence length
    rollout_steps = min(rollout_steps, z_targets.shape[1])

    z_hist  = z_start                   # (B, 1, 384) — grows each step
    a_hist  = []
    step_losses = []

    for t in range(rollout_steps):
        # Append next action to history
        a_t = a_seq[:, t:t+1]           # (B, 1)
        a_hist.append(a_t)
        a_hist_cat = torch.cat(a_hist, dim=1)  # (B, t+1)

        # Forward pass using full latent/action history
        pred_seq, _, _ = model(z_hist, a_hist_cat)

        # Take last predicted latent — this is the t+1 prediction
        z_pred = pred_seq[:, -1:, :]    # (B, 1, 384)

        # Supervise against ground truth at this step
        z_gt   = z_targets[:, t:t+1, :]  # (B, 1, 384)
        step_losses.append(loss_fn(z_pred, z_gt))

        # Feed predicted latent back as input — no teacher forcing
        z_hist = torch.cat([z_hist, z_pred], dim=1)  # (B, t+2, 384)

    return torch.stack(step_losses).mean()


# ── Components factory ────────────────────────────────────────────────────────

def build_components(device: torch.device, use_real: bool):
    """Return (buffer, model, optimizer, scheduler, loss_fn)."""
    buffer  = LatentReplayBuffer(capacity_steps=200_000) if (use_real and _REAL_BUFFER) \
              else _MockBuffer()

    if use_real and _REAL_MODEL:
        config = Config.from_params(num_layers=8, mlp_ratio=4, num_heads=8, learning_rate=3e-4, sequence_length=24)
        model  = DinoWorldModel(config).to(device)
        loss_fn = latent_mse_loss
    else:
        model   = _MockModel().to(device)
        loss_fn = _mock_mse

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
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

        data.close()

    print(f"[INFO] Loaded {loaded}/{len(files)} episodes | "
          f"Buffer steps: {buffer.total_steps}")
    return loaded


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(
    model,
    buffer,
    optimizer,
    loss_fn,
    device,
    batch_size:        int,
    batches_per_epoch: int,
    kl_weight:         float = 0.0,
    rollout_steps:     int   = 0,
    rollout_weight:    float = 0.5,
) -> dict:
    """
    One full epoch. Returns loss statistics dict.

    Args:
        kl_weight      : β — weight for KL divergence regularisation loss.
                         0.0 disables KL loss, reproducing original CDR behaviour.
        rollout_steps  : N — number of steps to unroll for rollout loss.
                         0 disables rollout loss entirely.
        rollout_weight : λ — weight for rollout loss term.
    """
    model.train()

    losses         = []
    losses_latent  = []
    losses_reward  = []
    losses_done    = []
    losses_kl      = []
    losses_rollout = []

    for _ in range(batches_per_epoch):
        # A. Sample latents from buffer ────────────────────────────────────────
        batch   = buffer.sample(batch_size, seq_len=SEQ_LEN)
        latents = batch.latents.to(device)
        actions = batch.actions.long().to(device)
        rewards = batch.rewards.to(device)
        dones   = batch.dones.to(device)

        # B. Shifted next-step prediction ──────────────────────────────────────
        z_in     = latents[:, :-1]              # (B, T-1, 384) — context
        a_in     = actions[:, :-1]              # (B, T-1)
        z_target = latents[:, 1:]               # (B, T-1, 384) — ground truth next latent
        r_target = rewards[:, 1:].unsqueeze(-1) # (B, T-1, 1)
        d_target = dones[:, 1:].unsqueeze(-1)   # (B, T-1, 1)

        # C. Forward + loss ────────────────────────────────────────────────────
        optimizer.zero_grad()
        pred_next, pred_rew, pred_done = model(z_in, a_in)

        loss_latent = loss_fn(pred_next, z_target)
        loss_reward = nn.functional.mse_loss(pred_rew, r_target)
        loss_done   = nn.functional.binary_cross_entropy_with_logits(
                          pred_done, d_target.clamp(0, 1)
                      )

        # ── KL Regularisation (Member B) ──────────────────────────────────────
        # Penalises predicted latents for diverging from the real buffer
        # latent distribution. Targets autoregressive drift at training time.
        # Disabled when kl_weight=0.0 (no overhead in that case).
        if kl_weight > 0.0:
            loss_kl = kl_divergence_loss(pred_next, z_target)
        else:
            loss_kl = torch.tensor(0.0, device=device)

        # ── Rollout Loss (Member B) ────────────────────────────────────────────
        # Unrolls the WM for rollout_steps using its own predicted latents as
        # input rather than ground truth. Forces robustness to compounding
        # errors — directly targets the exponential drift growth rate.
        # Disabled when rollout_steps=0 (no overhead in that case).
        if rollout_steps > 0:
            # Start from the first real latent in the sequence
            z_start  = latents[:, 0:1, :]       # (B, 1, 384)
            loss_rol = rollout_loss(
                model, z_start, a_in, z_target,
                rollout_steps, loss_fn,
            )
        else:
            loss_rol = torch.tensor(0.0, device=device)

        loss = (LAMBDA_LATENT  * loss_latent
              + LAMBDA_REWARD  * loss_reward
              + LAMBDA_DONE    * loss_done
              + kl_weight      * loss_kl
              + rollout_weight * loss_rol)

        # D. Backward ──────────────────────────────────────────────────────────
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        losses_latent.append(loss_latent.item())
        losses_reward.append(loss_reward.item())
        losses_done.append(loss_done.item())
        losses_kl.append(loss_kl.item())
        losses_rollout.append(loss_rol.item())

    n = len(losses)
    return {
        "avg_loss":          sum(losses)         / n,
        "min_loss":          min(losses),
        "max_loss":          max(losses),
        "avg_loss_latent":   sum(losses_latent)  / n,
        "avg_loss_reward":   sum(losses_reward)  / n,
        "avg_loss_done":     sum(losses_done)    / n,
        "avg_loss_kl":       sum(losses_kl)      / n,
        "avg_loss_rollout":  sum(losses_rollout) / n,
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
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batches_per_epoch", type=int,   default=200)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--data_dir",          type=str,   default="data/processed",
                   help="Dir with pre-encoded .npz latent files (Member C output)")
    p.add_argument("--save_dir",          type=str,   default="checkpoints")
    p.add_argument("--save_every",        type=int,   default=10,
                   help="Save a numbered checkpoint every N epochs")
    p.add_argument("--log_file",          type=str,   default="training_log.json")
    p.add_argument("--smoke_test",        action="store_true",
                   help="E.3: run 1 epoch × 10 batches and exit cleanly")
    p.add_argument("--mock",              action="store_true",
                   help="Force mock stubs (no partner code required)")

    # ── KL Regularisation (Member B) ──────────────────────────────────────────
    p.add_argument(
        "--kl_weight", type=float, default=1e-3,
        help=(
            "β: weight for KL divergence regularisation loss. "
            "Penalises predicted latents for diverging from the real buffer "
            "latent distribution. 0.0 disables KL loss (CDR baseline behaviour). "
            "Recommended range: 1e-4 (gentle) to 1e-2 (aggressive). "
            "Default: 1e-3."
        ),
    )

    # ── Rollout Loss (Member B) ────────────────────────────────────────────────
    p.add_argument(
        "--rollout_steps", type=int, default=4,
        help=(
            "N: number of steps to unroll for rollout loss. "
            "0 disables rollout loss entirely (CDR baseline behaviour). "
            "Feeds predicted latents back as input rather than ground truth, "
            "forcing the model to handle compounding errors at training time. "
            "Recommended range: 3-5 steps. Default: 4."
        ),
    )
    p.add_argument(
        "--rollout_weight", type=float, default=0.5,
        help=(
            "λ: weight for rollout loss term. "
            "Scales the multi-step rollout loss relative to the teacher-forced "
            "latent MSE loss. Default: 0.5."
        ),
    )

    # ── Resume from checkpoint (Member B) ────────────────────────────────────
    p.add_argument(
        "--resume", type=str, default=None,
        help=(
            "Path to a checkpoint to resume training from. "
            "Loads model weights (and optionally optimizer state) before "
            "training begins. Use this to continue from joint training: "
            "--resume checkpoints/world_model_joint_best.pt"
        ),
    )
    p.add_argument(
        "--resume_optimizer", action="store_true",
        help=(
            "Also restore optimizer state from the resume checkpoint. "
            "Use this when resuming an interrupted run. "
            "Leave unset when resuming from joint training (different optimizer state)."
        ),
    )

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
    print(f"[INFO] KL weight (β) : {args.kl_weight}"
          + (" (disabled)" if args.kl_weight == 0.0 else ""))
    print(f"[INFO] Rollout steps : {args.rollout_steps}"
          + (" (disabled)" if args.rollout_steps == 0 else ""))
    if args.rollout_steps > 0:
        print(f"[INFO] Rollout weight: {args.rollout_weight}")

    use_real = not args.mock
    buffer, model, optimizer, scheduler, loss_fn = build_components(device, use_real)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )
        print(f"[INFO] Resuming from  : {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        # Load model weights — always
        state_dict = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys (random-init): {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys (ignored): {len(unexpected)}")

        # Load optimizer state — only if requested
        if args.resume_optimizer and "optim_state" in ckpt:
            optimizer.load_state_dict(ckpt["optim_state"])
            print(f"[INFO] Optimizer state restored")
        else:
            print(f"[INFO] Optimizer state reset (fresh LR schedule)")

        resumed_epoch = ckpt.get("epoch", "?")
        resumed_loss  = ckpt.get("metrics", {}).get("avg_loss", float("nan"))
        print(f"[INFO] Resumed from epoch={resumed_epoch}  loss={resumed_loss:.6f}")

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
            kl_weight      = args.kl_weight,
            rollout_steps  = args.rollout_steps,
            rollout_weight = args.rollout_weight,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        kl_str  = f"  kl={stats['avg_loss_kl']:.4f}"      if args.kl_weight     > 0.0 else ""
        rol_str = f"  rol={stats['avg_loss_rollout']:.4f}" if args.rollout_steps > 0   else ""
        print(
            f"Epoch {epoch:4d}/{args.epochs}  |  "
            f"total={stats['avg_loss']:.4f}  "
            f"latent={stats['avg_loss_latent']:.4f}  "
            f"rew={stats['avg_loss_reward']:.4f}  "
            f"done={stats['avg_loss_done']:.4f}"
            f"{kl_str}{rol_str}  |  "
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