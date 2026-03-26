"""
train_wm_scheduled_sampling.py  —  Member E: Architectural fix SS
==================================================================
World Model training with Scheduled Sampling (Bengio et al. 2015).

What this fixes
---------------
The standard training loop always feeds ground-truth latents z_t as input
at every step. At evaluation time (dreaming), the model feeds its own
predictions z_hat_t instead. This train/eval mismatch is one structural
cause of autoregressive drift — the model was never trained to handle
its own errors.

Scheduled sampling fixes this by randomly replacing ground-truth inputs
with the model's own previous predictions during training, with probability
p annealed from 0 → ss_ratio_max over the training run:

    z_in[t] = z_t       with prob (1 - p)     ← real latent
    z_in[t] = z_hat_t-1 with prob p            ← model's own prediction

The annealing schedule means the model first learns on clean data
(p=0, identical to baseline), then gradually faces its own errors as
training progresses, forcing it to become robust to drift.

This requires a step-by-step autoregressive forward pass during training
rather than the single batched forward pass used in the baseline. This
makes each epoch slower — expect ~2-3x training time vs baseline.

Usage
-----
# Smoke test
python -m src.scripts.train_wm_scheduled_sampling --smoke_test

# Full run (default: ss_ratio_max=0.5, KL weight=0.01)
python -m src.scripts.train_wm_scheduled_sampling --epochs 50 --batches_per_epoch 200 --save_dir checkpoints/wm_ss

# Ablation: SS only, no KL
python -m src.scripts.train_wm_scheduled_sampling --epochs 50 --batches_per_epoch 200 --kl_weight 0.0 --save_dir checkpoints/wm_ss_nokl

# Sweep ss_ratio_max
python -m src.scripts.train_wm_scheduled_sampling --epochs 50 --batches_per_epoch 200 --ss_ratio_max 0.1 --save_dir checkpoints/wm_ss_p01
python -m src.scripts.train_wm_scheduled_sampling --epochs 50 --batches_per_epoch 200 --ss_ratio_max 0.3 --save_dir checkpoints/wm_ss_p03
python -m src.scripts.train_wm_scheduled_sampling --epochs 50 --batches_per_epoch 200 --ss_ratio_max 0.5 --save_dir checkpoints/wm_ss_p05

Interface Contract (unchanged from GDP Plan §2.3)
--------------------------------------------------
  Latent dim   : 384   (DINOv2 ViT-S/14)
  Seq length T : 24
  Latent tensor: (B, T, 384)
  Action tensor: (B, T)  — integer class ids [0..3]

The saved checkpoint format is identical to train_world_model.py.
Pass any checkpoint from this script directly to train_dream_dqn.py
and evaluate_transfer.py — no changes needed downstream.
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
    def sample(self, batch_size: int, seq_len: int = 24):
        from collections import namedtuple
        Batch = namedtuple("Batch", ["latents", "actions", "rewards", "dones"])
        return Batch(
            latents=torch.randn(batch_size, seq_len, 384),
            actions=torch.randint(0, ACTION_DIM, (batch_size, seq_len)).float(),
            rewards=torch.randn(batch_size, seq_len),
            dones=torch.zeros(batch_size, seq_len)
        )
    episodes = []


class _MockModel(nn.Module):
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

LAMBDA_LATENT = 1.0
LAMBDA_REWARD = 0.5
LAMBDA_DONE   = 0.5
KL_WEIGHT_DEFAULT     = 0.01
SS_RATIO_MAX_DEFAULT  = 0.5   # peak scheduled-sampling probability


# ── KL prior (identical to train_world_model.py) ──────────────────────────────

def compute_prior_gaussian(buffer, device, max_episodes=500):
    """
    Fit diagonal Gaussian prior p = Normal(mu, sigma) to real buffer latents.
    Used as KL regularisation target. Identical to train_world_model.py.
    """
    episodes = getattr(buffer, "episodes", [])
    if not episodes:
        print("[KL prior] No episodes found — using standard Normal(0,1) prior")
        return torch.zeros(LATENT_DIM, device=device), torch.ones(LATENT_DIM, device=device)

    all_latents = []
    for ep in episodes[:max_episodes]:
        all_latents.append(torch.tensor(ep.latents, dtype=torch.float32))
    all_latents = torch.cat(all_latents, dim=0).to(device)

    mu_prior  = all_latents.mean(dim=0)
    std_prior = all_latents.std(dim=0).clamp(min=1e-4)
    print(f"[KL prior] Fitted from {all_latents.shape[0]:,} real latent steps  |  "
          f"mu mean={mu_prior.mean().item():.4f}  std mean={std_prior.mean().item():.4f}")
    return mu_prior, std_prior


def kl_loss_fn(pred_next, mu_prior, std_prior):
    """KL( Normal(mu_q, sigma_q) || Normal(mu_prior, sigma_prior) ). Scalar."""
    B, T_minus1, D = pred_next.shape
    flat  = pred_next.reshape(-1, D)
    mu_q  = flat.mean(dim=0)
    std_q = flat.std(dim=0).clamp(min=1e-4)
    q = Normal(mu_q, std_q)
    p = Normal(mu_prior, std_prior)
    return kl_divergence(q, p).mean()


# ── Scheduled sampling forward pass ──────────────────────────────────────────

def scheduled_sampling_forward(
    model,
    latents:   torch.Tensor,   # (B, T, 384) — full real sequence
    actions:   torch.Tensor,   # (B, T)      — integer actions
    ss_prob:   float,          # current sampling probability p ∈ [0, 1]
    loss_fn,
    device:    torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Autoregressive forward pass with scheduled sampling.

    At each step t, the input fed to the model is:
        z_in[t] = z_t         with probability (1 - ss_prob)  ← ground truth
        z_in[t] = z_hat[t-1]  with probability ss_prob        ← model's prediction

    At t=0 we always feed the real z_0 (no previous prediction exists).

    Why step-by-step instead of batched?
        The batched call model(z_in, a_in) processes all T-1 steps at once,
        which means we can't conditionally swap individual timestep inputs
        based on the model's own outputs — those outputs don't exist yet.
        Step-by-step lets us generate z_hat[t], decide whether to use it
        at step t+1, then continue.

    Returns
    -------
    all_pred_next : (B, T-1, 384) — predicted next latents (for latent loss)
    all_pred_rew  : (B, T-1, 1)   — predicted rewards
    all_pred_done : (B, T-1, 1)   — predicted done logits
    """
    B, T, D = latents.shape
    T_steps = T - 1   # number of prediction steps (= SEQ_LEN - 1 = 23)

    all_pred_next = []
    all_pred_rew  = []
    all_pred_done = []

    # z_ctx: the sequence of inputs we've built so far — grows step by step.
    # Starts with z_0 (always real) as context.
    # Shape at step t: (B, t+1, 384)
    z_ctx = latents[:, 0:1, :]   # (B, 1, 384) — seed with real z_0

    prev_pred = None   # z_hat from previous step; None at t=0

    for t in range(T_steps):
        # ── Decide input for this step ────────────────────────────────────────
        # At t=0, prev_pred is None so we always use the real latent.
        # At t>0, flip a coin: use real z_t with prob (1 - ss_prob),
        #                      use prev_pred with prob ss_prob.
        if prev_pred is not None and torch.rand(1).item() < ss_prob:
            # SCHEDULED SAMPLING: feed model's own previous prediction
            # Shape: (B, 1, 384)
            z_input = prev_pred.detach()   # detach to avoid backprop through
                                            # the previous step's graph
        else:
            # TEACHER FORCING: feed real ground-truth latent z_t
            z_input = latents[:, t:t+1, :]   # (B, 1, 384)

        # Append this step's input to the context
        if t == 0:
            z_ctx = z_input                           # (B, 1, 384)
        else:
            z_ctx = torch.cat([z_ctx, z_input], dim=1)  # (B, t+1, 384)

        # ── Forward pass over full context so far ────────────────────────────
        # The Transformer attends to all previous steps.
        # a_ctx: actions up to and including step t
        a_ctx = actions[:, :t+1]   # (B, t+1)

        pred_next_ctx, pred_rew_ctx, pred_done_ctx = model(z_ctx, a_ctx)

        # We only care about the prediction at the LAST position (step t)
        # Shape: (B, 1, 384), (B, 1, 1), (B, 1, 1)
        z_hat_t    = pred_next_ctx[:, -1:, :]
        rew_hat_t  = pred_rew_ctx[:,  -1:, :]
        done_hat_t = pred_done_ctx[:, -1:, :]

        all_pred_next.append(z_hat_t)
        all_pred_rew.append(rew_hat_t)
        all_pred_done.append(done_hat_t)

        # Store prediction to potentially use as next step's input
        prev_pred = z_hat_t

    # Stack along time dimension → (B, T-1, 384) / (B, T-1, 1)
    all_pred_next = torch.cat(all_pred_next, dim=1)
    all_pred_rew  = torch.cat(all_pred_rew,  dim=1)
    all_pred_done = torch.cat(all_pred_done, dim=1)

    return all_pred_next, all_pred_rew, all_pred_done


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(
    model, buffer, optimizer, loss_fn, device,
    batch_size:        int,
    batches_per_epoch: int,
    kl_weight:         float,
    mu_prior:          torch.Tensor,
    std_prior:         torch.Tensor,
    ss_prob:           float,          # SCHEDULED SAMPLING: current p for this epoch
) -> dict:
    """
    One full epoch with scheduled sampling.

    ss_prob is the probability of replacing a ground-truth input with the
    model's previous prediction at each step. Passed in from main() where
    it is annealed linearly from 0 → ss_ratio_max over training epochs.
    """
    model.train()

    losses        = []
    losses_latent = []
    losses_reward = []
    losses_done   = []
    losses_kl     = []

    for _ in range(batches_per_epoch):
        # A. Sample batch from buffer ──────────────────────────────────────────
        batch   = buffer.sample(batch_size, seq_len=SEQ_LEN)
        latents = batch.latents.to(device)   # (B, T, 384)
        actions = batch.actions.long().to(device)
        rewards = batch.rewards.to(device)
        dones   = batch.dones.to(device)

        # B. Targets (shifted by 1 — predicting next step) ────────────────────
        z_target = latents[:, 1:]                   # (B, T-1, 384)
        r_target = rewards[:, 1:].unsqueeze(-1)     # (B, T-1, 1)
        d_target = dones[:, 1:].unsqueeze(-1)       # (B, T-1, 1)

        # C. Forward pass ──────────────────────────────────────────────────────
        # SCHEDULED SAMPLING: step-by-step autoregressive forward.
        # When ss_prob=0.0 this is equivalent to teacher forcing (baseline).
        # When ss_prob>0, some inputs are replaced with the model's own
        # previous predictions — forcing the model to be robust to its own errors.
        optimizer.zero_grad()

        pred_next, pred_rew, pred_done = scheduled_sampling_forward(
            model, latents, actions, ss_prob, loss_fn, device
        )

        # D. Losses ────────────────────────────────────────────────────────────
        loss_latent = loss_fn(pred_next, z_target)
        loss_reward = nn.functional.mse_loss(pred_rew, r_target)
        loss_done   = nn.functional.binary_cross_entropy_with_logits(
                          pred_done, d_target.clamp(0, 1)
                      )

        if kl_weight > 0.0:
            loss_kl = kl_loss_fn(pred_next, mu_prior, std_prior)
        else:
            loss_kl = torch.tensor(0.0, device=device)

        loss = (LAMBDA_LATENT * loss_latent
              + LAMBDA_REWARD * loss_reward
              + LAMBDA_DONE   * loss_done
              + kl_weight     * loss_kl)

        # E. Backward ──────────────────────────────────────────────────────────
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        losses_latent.append(loss_latent.item())
        losses_reward.append(loss_reward.item())
        losses_done.append(loss_done.item())
        losses_kl.append(loss_kl.item())

    n = len(losses)
    return {
        "avg_loss":        sum(losses)        / n,
        "min_loss":        min(losses),
        "max_loss":        max(losses),
        "avg_loss_latent": sum(losses_latent) / n,
        "avg_loss_reward": sum(losses_reward) / n,
        "avg_loss_done":   sum(losses_done)   / n,
        "avg_loss_kl":     sum(losses_kl)     / n,
        "ss_prob":         ss_prob,            # log for analysis
    }


# ── Components factory (identical to train_world_model.py) ────────────────────

def build_components(device, use_real):
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


def load_buffer_from_disk(buffer, processed_dir: Path) -> int:
    import numpy as np
    files = sorted(processed_dir.glob("*.npz"))
    if not files:
        print(f"[WARN] No .npz files found in {processed_dir}")
        return 0
    loaded = 0
    for fpath in files:
        data = np.load(fpath)
        if "latents" not in data:
            continue
        missing = [k for k in ("actions", "rewards", "dones") if k not in data]
        if missing:
            continue
        buffer.add_episode(
            latents=data["latents"], actions=data["actions"],
            rewards=data["rewards"], dones=data["dones"]
        )
        loaded += 1
        data.close()
    print(f"[INFO] Loaded {loaded}/{len(files)} episodes | Buffer steps: {buffer.total_steps}")
    return loaded


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Latent World Model with Scheduled Sampling (Member E fix SS)"
    )
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batches_per_epoch", type=int,   default=200)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--data_dir",          type=str,   default="data/processed")
    p.add_argument("--save_dir",          type=str,   default="checkpoints/wm_ss")
    p.add_argument("--save_every",        type=int,   default=10)
    p.add_argument("--log_file",          type=str,   default="training_log_ss.json")
    p.add_argument("--smoke_test",        action="store_true")
    p.add_argument("--mock",              action="store_true")
    # KL regularisation (carried over from train_world_model.py)
    p.add_argument("--kl_weight",         type=float, default=KL_WEIGHT_DEFAULT,
                   help="KL regularisation weight β. 0.0 = no KL.")
    # SCHEDULED SAMPLING: ──────────────────────────────────────────────────────
    # ss_ratio_max — the peak probability of replacing a ground-truth input
    # with the model's own prediction. Annealed linearly from 0 → ss_ratio_max
    # over the full training run.
    #
    # Intuition: at epoch 1, p=0 (pure teacher forcing, identical to baseline).
    # At the final epoch, p=ss_ratio_max. So if ss_ratio_max=0.5, by epoch 50
    # half of all inputs are the model's own predictions.
    #
    # Recommended sweep: 0.1, 0.3, 0.5
    # Start with 0.5 (the paper default). Lower if loss fails to converge.
    p.add_argument("--ss_ratio_max",      type=float, default=SS_RATIO_MAX_DEFAULT,
                   help="Peak scheduled sampling probability (0=baseline, 0.5=default).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        print("=" * 62)
        print("  SMOKE TEST (SS) — 1 epoch, 10 batches (integration check)")
        print("=" * 62)
        args.epochs            = 1
        args.batches_per_epoch = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device          : {device}")
    print(f"[INFO] Epochs          : {args.epochs}")
    print(f"[INFO] Batches/epoch   : {args.batches_per_epoch}")
    print(f"[INFO] KL weight (β)   : {args.kl_weight}")
    print(f"[INFO] SS ratio max (p): {args.ss_ratio_max}"
          + ("  ← SS DISABLED (teacher forcing only)" if args.ss_ratio_max == 0.0
             else "  ← SCHEDULED SAMPLING ENABLED"))
    print(f"[INFO] Real buffer     : {_REAL_BUFFER and not args.mock}")
    print(f"[INFO] Real model      : {_REAL_MODEL and not args.mock}")

    use_real = not args.mock
    buffer, model, optimizer, scheduler, loss_fn = build_components(device, use_real)

    if use_real and _REAL_BUFFER:
        load_buffer_from_disk(buffer, Path(args.data_dir))

    mu_prior, std_prior = compute_prior_gaussian(buffer, device)

    save_dir  = Path(args.save_dir)
    log       = []
    best_loss = float("inf")

    print("\n─── Training Start (Scheduled Sampling) " + "─" * 22)
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # SCHEDULED SAMPLING: linear annealing of ss_prob
        # At epoch 1: ss_prob = 0 (pure teacher forcing)
        # At epoch N: ss_prob = ss_ratio_max
        # Formula: p(epoch) = ss_ratio_max * (epoch - 1) / (epochs - 1)
        # Guard against epochs=1 (smoke test) with max(1, ...) in denominator.
        ss_prob = args.ss_ratio_max * (epoch - 1) / max(1, args.epochs - 1)

        stats = train_epoch(
            model, buffer, optimizer, loss_fn,
            device, args.batch_size, args.batches_per_epoch,
            kl_weight = args.kl_weight,
            mu_prior  = mu_prior,
            std_prior = std_prior,
            ss_prob   = ss_prob,
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
            f"kl={stats['avg_loss_kl']:.4f}  |  "
            f"ss_p={ss_prob:.3f}  |  "
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

        log.append({"epoch": epoch, "lr": lr_now, "ss_prob": ss_prob, **stats})

    final = save_checkpoint(model, optimizer, args.epochs, log[-1], save_dir, "final")
    print(f"\n[INFO] Final weights saved → {final}")
    print(f"[INFO] Training time       → {(time.time()-t_start)/60:.1f} min")
    print(f"[INFO] Best avg loss       → {best_loss:.6f}")

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[INFO] Training log        → {log_path}")

    if args.smoke_test:
        print("\n" + "=" * 62)
        print("  ✅  SMOKE TEST PASSED — Scheduled Sampling pipeline OK.")
        print("=" * 62)

    return model


def save_checkpoint(model, optimizer, epoch, metrics, save_dir, tag="last"):
    save_dir = Path(save_dir)
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


if __name__ == "__main__":
    main()