"""
train_wm_ss_tbptt.py  —  Member E: Architectural fix SS + Truncated BPTT
=========================================================================
World Model training combining:
  1. Scheduled Sampling (Bengio et al. 2015)
  2. Truncated Backpropagation Through Time with real latent re-anchoring

Note: Residual prediction (predicting Δz instead of z) is already
implemented in DinoWorldModel.delta_head — no changes needed there.

Why combine SS + Truncated BPTT?
---------------------------------
SS trains the model to be robust to its own prediction errors by
occasionally feeding its own outputs back as inputs. This reduces
drift but doesn't cap how far errors can compound within a sequence.

Truncated BPTT caps error accumulation by re-anchoring to a real latent
every k steps during training. The model learns dynamics over windows of
length k, which matches the reliable rollout horizon we want at eval time.

Together:
  - SS: model learns to handle noisy inputs (robust to drift)
  - TBPTT: caps maximum error accumulation to k steps (bounds drift)
  - Residual (existing): predicts small Δz, lower per-step error magnitude

How Truncated BPTT works here
------------------------------
The sequence of length T=24 is split into windows of length k.
At the start of each window, the context is reset to a real latent
from the buffer (the ground-truth latent at that position). Within
each window, the forward pass is autoregressive with optional SS.

Example with T=24, k=6:
  Window 1: real z_0  → predict z_1..z_6   (ss active within window)
  Window 2: real z_6  → predict z_7..z_12  (re-anchored to real z_6)
  Window 3: real z_12 → predict z_13..z_18
  Window 4: real z_18 → predict z_19..z_24

Key difference from fix 3b (which we disabled):
  Fix 3b re-anchored at EVALUATION time → agent teleports, no navigation.
  TBPTT re-anchors at TRAINING time only → WM learns short-horizon
  dynamics accurately. At eval time, rollout is still fully autoregressive.

Interface Contract (unchanged from GDP Plan §2.3)
--------------------------------------------------
  Latent dim   : 384
  Seq length T : 24
  Checkpoint format: identical to train_world_model.py
  Downstream scripts: train_dream_dqn.py, evaluate_transfer.py unchanged

Usage
-----
# Smoke test
python -m src.scripts.train_wm_ss_tbptt --smoke_test

# Full run (recommended defaults)
python -m src.scripts.train_wm_ss_tbptt --epochs 50 --batches_per_epoch 200 --ss_ratio_max 0.5 --tbptt_k 6 --save_dir checkpoints/wm_ss_tbptt

# Ablation: TBPTT only (SS disabled)
python -m src.scripts.train_wm_ss_tbptt --epochs 50 --batches_per_epoch 200 --ss_ratio_max 0.0 --tbptt_k 6 --save_dir checkpoints/wm_tbptt

# Sweep tbptt_k
python -m src.scripts.train_wm_ss_tbptt --epochs 50 --batches_per_epoch 200 --ss_ratio_max 0.5 --tbptt_k 4  --save_dir checkpoints/wm_ss_tbptt_k4
python -m src.scripts.train_wm_ss_tbptt --epochs 50 --batches_per_epoch 200 --ss_ratio_max 0.5 --tbptt_k 8  --save_dir checkpoints/wm_ss_tbptt_k8
python -m src.scripts.train_wm_ss_tbptt --epochs 50 --batches_per_epoch 200 --ss_ratio_max 0.5 --tbptt_k 12 --save_dir checkpoints/wm_ss_tbptt_k12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

# ── Real imports ───────────────────────────────────────────────────────────────
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


# ── Fallback stubs ─────────────────────────────────────────────────────────────

class _MockBuffer:
    def sample(self, batch_size: int, seq_len: int = 24):
        from collections import namedtuple
        Batch = namedtuple("Batch", ["latents", "actions", "rewards", "dones"])
        return Batch(
            latents=torch.randn(batch_size, seq_len, 384),
            actions=torch.randint(0, 4, (batch_size, seq_len)).float(),
            rewards=torch.randn(batch_size, seq_len),
            dones=torch.zeros(batch_size, seq_len),
        )
    episodes = []


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(384, 384)

    def forward(self, z_in, a_in):
        pred_next = z_in + self.proj(z_in)
        pred_rew  = torch.zeros(*z_in.shape[:2], 1, device=z_in.device)
        pred_val  = torch.zeros(*z_in.shape[:2], 1, device=z_in.device)
        return pred_next, pred_rew, pred_val


def _mock_mse(pred, target):
    return nn.functional.mse_loss(pred, target)


# ── Constants ──────────────────────────────────────────────────────────────────

SEQ_LEN    = 24
LATENT_DIM = 384
ACTION_DIM = 4

LAMBDA_LATENT        = 1.0
LAMBDA_REWARD        = 0.5
LAMBDA_DONE          = 0.5
KL_WEIGHT_DEFAULT    = 0.0    # off by default — SS already reduces drift
SS_RATIO_MAX_DEFAULT = 0.5
TBPTT_K_DEFAULT      = 6     # window length — 6 steps reliable horizon from SS run


# ── KL prior (identical to train_world_model.py) ───────────────────────────────

def compute_prior_gaussian(buffer, device, max_episodes=500):
    episodes = getattr(buffer, "episodes", [])
    if not episodes:
        print("[KL prior] No episodes — using Normal(0,1)")
        return torch.zeros(LATENT_DIM, device=device), torch.ones(LATENT_DIM, device=device)
    all_latents = []
    for ep in episodes[:max_episodes]:
        all_latents.append(torch.tensor(ep.latents, dtype=torch.float32))
    all_latents = torch.cat(all_latents, dim=0).to(device)
    mu  = all_latents.mean(dim=0)
    std = all_latents.std(dim=0).clamp(min=1e-4)
    print(f"[KL prior] Fitted from {all_latents.shape[0]:,} steps  |  "
          f"mu mean={mu.mean().item():.4f}  std mean={std.mean().item():.4f}")
    return mu, std


def kl_loss_fn(pred_next, mu_prior, std_prior):
    B, T1, D = pred_next.shape
    flat  = pred_next.reshape(-1, D)
    mu_q  = flat.mean(dim=0)
    std_q = flat.std(dim=0).clamp(min=1e-4)
    return kl_divergence(Normal(mu_q, std_q), Normal(mu_prior, std_prior)).mean()


# ── SS + TBPTT forward pass ────────────────────────────────────────────────────

def ss_tbptt_forward(
    model,
    latents:  torch.Tensor,   # (B, T, 384) — full real sequence
    actions:  torch.Tensor,   # (B, T)      — integer actions
    ss_prob:  float,           # scheduled sampling probability
    tbptt_k:  int,             # window length for truncated BPTT
    loss_fn,
    device:   torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Autoregressive forward pass with Scheduled Sampling + Truncated BPTT.

    The sequence of T-1 prediction steps is divided into windows of
    length tbptt_k. At the start of each window, the latent context
    is re-anchored to the real ground-truth latent at that position.
    Within each window, SS applies as normal.

    Why re-anchor at window boundaries?
        Without re-anchoring, errors compound over the full T-1 steps.
        With re-anchoring every k steps, maximum error accumulation
        is bounded to k steps. The model learns accurate k-step dynamics
        rather than increasingly noisy long-horizon predictions.

    Why detach at window boundaries?
        At each window boundary we detach the context from the
        computation graph. This implements true truncated BPTT —
        gradients only flow within each window of length k, not
        across the full sequence. This prevents exploding gradients
        over long sequences and speeds up training.

    Returns
    -------
    all_pred_next : (B, T-1, 384)
    all_pred_rew  : (B, T-1, 1)
    all_pred_done : (B, T-1, 1)
    """
    B, T, D = latents.shape
    T_steps = T - 1   # = SEQ_LEN - 1 = 23

    all_pred_next = []
    all_pred_rew  = []
    all_pred_done = []

    prev_pred = None   # z_hat from previous step for SS

    for t in range(T_steps):

        # ── TBPTT: re-anchor at window boundaries ─────────────────────────────
        # At t=0 and at every multiple of tbptt_k, reset context to real latent.
        # This bounds error accumulation to at most tbptt_k steps.
        if t % tbptt_k == 0:
            # Re-anchor: use real latent z_t as the start of a fresh window.
            # Detach to cut gradient flow from previous window.
            z_ctx  = latents[:, t:t+1, :].detach()   # (B, 1, 384)
            a_ctx  = actions[:, t:t+1]                # (B, 1)
            prev_pred = None   # reset SS state — no previous prediction in new window
        else:
            # ── SS: decide input for this step ────────────────────────────────
            if prev_pred is not None and torch.rand(1).item() < ss_prob:
                # Use model's own previous prediction (scheduled sampling)
                z_input = prev_pred.detach()
            else:
                # Use real ground-truth latent (teacher forcing)
                z_input = latents[:, t:t+1, :]

            z_ctx = torch.cat([z_ctx, z_input], dim=1)   # (B, t_in_window+1, 384)
            a_ctx = torch.cat([a_ctx, actions[:, t:t+1]], dim=1)

        # ── Forward pass over current window context ──────────────────────────
        pred_next_ctx, pred_rew_ctx, pred_done_ctx = model(z_ctx, a_ctx)

        # Take prediction at the last position in the current context
        z_hat_t    = pred_next_ctx[:, -1:, :]    # (B, 1, 384)
        rew_hat_t  = pred_rew_ctx[:,  -1:, :]    # (B, 1, 1)
        done_hat_t = pred_done_ctx[:, -1:, :]    # (B, 1, 1)

        all_pred_next.append(z_hat_t)
        all_pred_rew.append(rew_hat_t)
        all_pred_done.append(done_hat_t)

        prev_pred = z_hat_t   # store for potential SS use next step

    all_pred_next = torch.cat(all_pred_next, dim=1)   # (B, T-1, 384)
    all_pred_rew  = torch.cat(all_pred_rew,  dim=1)   # (B, T-1, 1)
    all_pred_done = torch.cat(all_pred_done, dim=1)   # (B, T-1, 1)

    return all_pred_next, all_pred_rew, all_pred_done


# ── Training epoch ─────────────────────────────────────────────────────────────

def train_epoch(
    model, buffer, optimizer, loss_fn, device,
    batch_size:        int,
    batches_per_epoch: int,
    kl_weight:         float,
    mu_prior:          torch.Tensor,
    std_prior:         torch.Tensor,
    ss_prob:           float,
    tbptt_k:           int,
) -> dict:
    model.train()

    losses        = []
    losses_latent = []
    losses_reward = []
    losses_done   = []
    losses_kl     = []

    for _ in range(batches_per_epoch):
        batch   = buffer.sample(batch_size, seq_len=SEQ_LEN)
        latents = batch.latents.to(device)
        actions = batch.actions.long().to(device)
        rewards = batch.rewards.to(device)
        dones   = batch.dones.to(device)

        z_target = latents[:, 1:]                   # (B, T-1, 384)
        r_target = rewards[:, 1:].unsqueeze(-1)     # (B, T-1, 1)
        d_target = dones[:, 1:].unsqueeze(-1)       # (B, T-1, 1)

        optimizer.zero_grad()

        pred_next, pred_rew, pred_done = ss_tbptt_forward(
            model, latents, actions, ss_prob, tbptt_k, loss_fn, device
        )

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
        "ss_prob":         ss_prob,
        "tbptt_k":         tbptt_k,
    }


# ── Components factory ─────────────────────────────────────────────────────────

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
        with open(fpath, "rb") as f:
            import numpy as np
            data = np.load(f)
            missing = [k for k in ("latents", "actions", "rewards", "dones") if k not in data]
            if missing:
                continue
            buffer.add_episode(
                latents=data["latents"], actions=data["actions"],
                rewards=data["rewards"], dones=data["dones"],
            )
            loaded += 1
    print(f"[INFO] Loaded {loaded}/{len(files)} episodes | Buffer steps: {buffer.total_steps}")
    return loaded


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


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train WM with Scheduled Sampling + Truncated BPTT (Member E)"
    )
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batches_per_epoch", type=int,   default=200)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--data_dir",          type=str,   default="data/processed")
    p.add_argument("--save_dir",          type=str,   default="checkpoints/wm_ss_tbptt")
    p.add_argument("--save_every",        type=int,   default=10)
    p.add_argument("--log_file",          type=str,   default="training_log_ss_tbptt.json")
    p.add_argument("--smoke_test",        action="store_true")
    p.add_argument("--mock",              action="store_true")
    p.add_argument("--kl_weight",         type=float, default=KL_WEIGHT_DEFAULT,
                   help="KL regularisation weight. 0.0=disabled (default).")
    p.add_argument("--ss_ratio_max",      type=float, default=SS_RATIO_MAX_DEFAULT,
                   help="Peak scheduled sampling probability. 0.0=TBPTT only.")
    # TRUNCATED BPTT: ──────────────────────────────────────────────────────────
    # tbptt_k — window length in steps. The sequence is re-anchored to a real
    # latent every k steps during training. Gradients are truncated at window
    # boundaries (detach), so the effective BPTT horizon is k steps.
    #
    # Choosing k:
    #   k=4  — very short windows, very stable, may miss medium-range dependencies
    #   k=6  — default, matches ~reliable rollout horizon from SS drift diagnostic
    #   k=8  — longer windows, more context, slightly more drift risk
    #   k=12 — half the sequence length, minimal truncation
    #
    # Rule of thumb: k should match the step at which DDA drops below 0.6
    # in the drift diagnostic. From our SS run: DDA threshold = step 16,
    # so k=6 is conservative, k=12 is more aggressive.
    p.add_argument("--tbptt_k",           type=int,   default=TBPTT_K_DEFAULT,
                   help="TBPTT window length in steps. Re-anchors to real latent every k steps.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        print("=" * 66)
        print("  SMOKE TEST (SS + TBPTT) — 1 epoch, 10 batches")
        print("=" * 66)
        args.epochs            = 1
        args.batches_per_epoch = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device          : {device}")
    print(f"[INFO] Epochs          : {args.epochs}")
    print(f"[INFO] Batches/epoch   : {args.batches_per_epoch}")
    print(f"[INFO] KL weight (β)   : {args.kl_weight}")
    print(f"[INFO] SS ratio max (p): {args.ss_ratio_max}"
          + ("  ← SS DISABLED" if args.ss_ratio_max == 0.0 else "  ← SS ENABLED"))
    print(f"[INFO] TBPTT window (k): {args.tbptt_k}  ← re-anchor every {args.tbptt_k} steps")
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

    print("\n─── Training Start (SS + Truncated BPTT) " + "─" * 20)
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Linear annealing of ss_prob: 0 → ss_ratio_max over training
        ss_prob = args.ss_ratio_max * (epoch - 1) / max(1, args.epochs - 1)

        stats = train_epoch(
            model, buffer, optimizer, loss_fn,
            device, args.batch_size, args.batches_per_epoch,
            kl_weight = args.kl_weight,
            mu_prior  = mu_prior,
            std_prior = std_prior,
            ss_prob   = ss_prob,
            tbptt_k   = args.tbptt_k,
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
            f"ss_p={ss_prob:.3f}  k={args.tbptt_k}  |  "
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

        log.append({"epoch": epoch, "lr": lr_now,
                    "ss_prob": ss_prob, "tbptt_k": args.tbptt_k, **stats})

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
        print("\n" + "=" * 66)
        print("  ✅  SMOKE TEST PASSED — SS + TBPTT pipeline OK.")
        print("=" * 66)

    return model


if __name__ == "__main__":
    main()