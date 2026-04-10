"""
train_world_model_joint.py  —  Member B: Joint DINOv2 + World Model Training
=============================================================================
Trains the World Model jointly with the DINOv2 encoder, allowing the encoder
to reshape the latent space for maze navigation.

Standard pipeline (pre-encoded):
    raw obs → [frozen DINOv2] → latents (stored) → [WM] → pred latents

Joint training pipeline (on-the-fly encoding):
    raw obs → [fine-tuned DINOv2] → latents → [WM] → pred latents
                      ↑                                      |
                      └──────── backprop through both ───────┘

WHY THIS IS NEEDED:
    T-SNE analysis showed the goal state forms a completely isolated cluster
    in DINOv2 latent space, 3.4 units away from normal states (vs 2.15 for
    normal transitions). DINOv2 was trained on ImageNet, not mazes — the
    goal pixel (green, bottom-right) produces a dramatically different
    embedding unrelated to its navigational meaning. Joint training reshapes
    the latent space so that goal proximity is encoded smoothly, making the
    world model's dynamics learnable and the transfer gap closeable.

OPTIMISATIONS vs original:
    - Lazy loading: only file paths stored in memory, episodes loaded on demand
    - Single encode pass per batch: observations encoded once, reused for all losses
    - Rollout loss uses pre-computed latents, no second encoder call
    - Smaller default batch size (8) to reduce peak RAM
    - Batched DINOv2 forward pass over all T frames at once

TWO FINE-TUNING MODES:
    --finetune_blocks N  : unfreeze last N transformer blocks of DINOv2
                           N=2 (default) — safe, fast, recommended first try
                           N=12 (all)    — full fine-tune, slower, more powerful

COMMANDS:
    # Smoke test first
    python -m src.scripts.train_world_model_joint --smoke_test

    # Option A: Fine-tune last 2 blocks (safer, recommended)
    python -m src.scripts.train_world_model_joint \\
        --raw_dir data/raw \\
        --epochs 100 \\
        --batches_per_epoch 200 \\
        --kl_weight 5e-3 \\
        --rollout_steps 8 \\
        --rollout_weight 1.0 \\
        --finetune_blocks 2 \\
        --encoder_lr 1e-5

    # Option B: Fine-tune all 12 blocks (more powerful, slower)
    python -m src.scripts.train_world_model_joint \\
        --raw_dir data/raw \\
        --epochs 100 \\
        --batches_per_epoch 200 \\
        --kl_weight 5e-3 \\
        --rollout_steps 8 \\
        --rollout_weight 1.0 \\
        --finetune_blocks 12 \\
        --encoder_lr 1e-6

OUTPUTS:
    checkpoints/world_model_joint_best.pt  — WM weights (compatible with pipeline)
    checkpoints/encoder_finetuned.pt       — fine-tuned encoder weights
    training_log_joint.json                — training log
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── Imports ───────────────────────────────────────────────────────────────────

try:
    from src.models.transformer import DinoWorldModel, latent_mse_loss
    from src.models.transformer_configuration import TransformerWMConfiguration as Config
    _REAL_MODEL = True
except ImportError:
    _REAL_MODEL = False
    print("[WARN] src.models.transformer not found")

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN    = 24
LATENT_DIM = 384
ACTION_DIM = 4

LAMBDA_LATENT = 1.0
LAMBDA_REWARD = 0.5
LAMBDA_DONE   = 0.1

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ── Fine-tuneable DINOv2 Encoder ──────────────────────────────────────────────

class FineTunableDinoV2(nn.Module):
    """
    DINOv2 ViT-S/14 with selective layer unfreezing for joint training.
    All layers frozen by default. Call unfreeze_last_n_blocks(N) to enable
    fine-tuning of the last N transformer blocks.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        print("[INFO] Loading DINOv2 ViT-S/14...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14"
        )
        self.backbone.to(device)

        # Freeze everything initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.register_buffer("mean", IMAGENET_MEAN.to(device))
        self.register_buffer("std",  IMAGENET_STD.to(device))

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze last N transformer blocks + final layer norm."""
        if n <= 0:
            print("[INFO] Encoder fully frozen")
            return

        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        total_blocks = len(self.backbone.blocks)
        start_block  = max(0, total_blocks - n)
        for i in range(start_block, total_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.backbone.parameters()
                        if p.requires_grad)
        total     = sum(p.numel() for p in self.backbone.parameters())
        print(f"[INFO] Unfroze last {n}/{total_blocks} DINOv2 blocks  "
              f"({trainable:,} / {total:,} params trainable)")

    def encode_batch(self, obs_BT: torch.Tensor) -> torch.Tensor:
        """
        Encode a flat batch of observations in one DINOv2 forward pass.

        Optimisations:
        - Always divides by 255 (inputs are always uint8 — no max() scan)
        - Frozen blocks run inside torch.no_grad() to skip gradient tracking

        Args:
            obs_BT : (N, 64, 64, 3) uint8

        Returns:
            latents : (N, 384)
        """
        x = obs_BT.permute(0, 3, 1, 2).float() / 255.0   # (N, 3, 64, 64)
        x = (x - self.mean) / self.std
        x = F.interpolate(x, size=(224, 224),
                          mode="bilinear", align_corners=False)
        return self.backbone(x)                             # (N, 384)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence batch in one DINOv2 call.

        Args:
            obs : (B, T, 64, 64, 3) uint8

        Returns:
            latents : (B, T, 384)
        """
        B, T, H, W, C = obs.shape
        flat    = obs.reshape(B * T, H, W, C)        # (B*T, 64, 64, 3)
        latents = self.encode_batch(flat)             # (B*T, 384)
        return latents.reshape(B, T, LATENT_DIM)     # (B, T, 384)


# ── Lazy Episode Index ────────────────────────────────────────────────────────

class LazyEpisodeIndex:
    """
    Stores only file paths and metadata in memory.
    Loads raw observations from disk only when sampled.

    Memory usage: ~1KB per episode (path + metadata) vs ~786KB per episode
    for full in-memory storage. For 3000 episodes: ~3MB vs ~2.4GB.

    Caches goal_pool and any_pool as numpy arrays so sample_batch
    does not rebuild them from scratch on every call.
    """

    def __init__(self):
        self.paths      = []   # Path to each .npz file
        self.lengths    = []   # Episode length (T)
        self.has_goal   = []   # Whether episode contains a goal transition
        self.goal_idxs  = []   # Index of goal transition (-1 if none)

        # Cached numpy index arrays — built once in finalise()
        self.goal_pool: np.ndarray | None = None  # indices of goal episodes
        self.any_pool:  np.ndarray | None = None  # indices of all episodes

    def add(self, path: Path, length: int, has_goal: bool, goal_idx: int) -> None:
        self.paths.append(path)
        self.lengths.append(length)
        self.has_goal.append(has_goal)
        self.goal_idxs.append(goal_idx)

    def finalise(self) -> None:
        """
        Build cached numpy index arrays after all episodes are added.
        Call once after build_index() completes.
        Avoids rebuilding these lists 200× per epoch inside sample_batch.
        """
        all_i       = np.arange(len(self.paths), dtype=np.int32)
        goal_mask   = np.array(self.has_goal, dtype=bool)
        self.goal_pool = all_i[goal_mask]
        self.any_pool  = all_i

    def load_episode(self, i: int) -> dict:
        """Load a single episode from disk."""
        d = np.load(self.paths[i])
        ep = {
            "obs":     d["obs"].astype(np.uint8),
            "actions": d["actions"].astype(np.int64),
            "rewards": d["rewards"].astype(np.float32),
            "dones":   d["dones"].astype(np.float32),
        }
        d.close()
        return ep

    def __len__(self) -> int:
        return len(self.paths)


def build_index(raw_dir: Path, seq_len: int) -> LazyEpisodeIndex:
    """
    Scan raw episode files and build a lazy index.
    Only reads rewards (tiny) to identify goal episodes — no obs loaded.
    """
    idx   = LazyEpisodeIndex()
    files = sorted(raw_dir.glob("ep_*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No episode files in {raw_dir}. "
            f"Run collect_data_dqn.py first."
        )

    print(f"[INFO] Indexing {len(files)} episodes...")
    goal_count = 0

    for fpath in files:
        d = np.load(fpath)
        missing = [k for k in ("obs", "actions", "rewards", "dones")
                   if k not in d]
        if missing:
            d.close()
            continue

        rewards  = d["rewards"].astype(np.float32)
        length   = len(rewards)
        has_goal = bool(rewards.max() > 0.5)
        goal_idx = int(np.argmax(rewards > 0.5)) if has_goal else -1
        d.close()

        if length < seq_len:
            continue

        idx.add(fpath, length, has_goal, goal_idx)
        if has_goal:
            goal_count += 1

    print(f"[INFO] Indexed {len(idx)} episodes  "
          f"(goal: {goal_count}/{len(idx)}, "
          f"{100*goal_count/max(1,len(idx)):.1f}%)")
    idx.finalise()
    return idx


# ── Batch Sampler ─────────────────────────────────────────────────────────────

def sample_batch(
    index:         LazyEpisodeIndex,
    batch_size:    int,
    seq_len:       int,
    device:        torch.device,
    goal_fraction: float = 0.25,
) -> dict:
    """
    Sample a batch with stratified goal sampling.
    Loads only the required episodes from disk — no full buffer in RAM.
    Uses cached numpy index arrays — no per-call list reconstruction.

    Returns dict with tensors on device:
        obs     : (B, T, 64, 64, 3) uint8
        actions : (B, T) int64
        rewards : (B, T) float32
        dones   : (B, T) float32
    """
    n_goal    = min(int(batch_size * goal_fraction), len(index.goal_pool))
    n_nongoal = batch_size - n_goal

    obs_out     = np.zeros((batch_size, seq_len, 64, 64, 3), dtype=np.uint8)
    actions_out = np.zeros((batch_size, seq_len),            dtype=np.int64)
    rewards_out = np.zeros((batch_size, seq_len),            dtype=np.float32)
    dones_out   = np.zeros((batch_size, seq_len),            dtype=np.float32)

    def fill(b: int, ep_i: int, force_goal: bool) -> None:
        ep = index.load_episode(ep_i)
        T  = len(ep["obs"])
        if force_goal:
            gi        = index.goal_idxs[ep_i]
            start_min = max(0, gi - seq_len + 1)
            start_max = min(gi, T - seq_len)
            start     = np.random.randint(start_min, start_max + 1)
        else:
            start = np.random.randint(0, T - seq_len + 1)
        sl              = slice(start, start + seq_len)
        obs_out[b]     = ep["obs"][sl]
        actions_out[b] = ep["actions"][sl]
        rewards_out[b] = ep["rewards"][sl]
        dones_out[b]   = ep["dones"][sl]

    # Goal slots — use cached goal_pool numpy array
    chosen_goal = np.random.choice(index.goal_pool, size=n_goal, replace=True)
    for b, ep_i in enumerate(chosen_goal):
        fill(b, int(ep_i), force_goal=True)

    # Non-goal slots — use cached any_pool numpy array
    chosen_any = np.random.choice(index.any_pool, size=n_nongoal, replace=True)
    for b, ep_i in enumerate(chosen_any, start=n_goal):
        fill(b, int(ep_i), force_goal=False)

    return {
        "obs":     torch.from_numpy(obs_out).to(device),
        "actions": torch.from_numpy(actions_out).to(device),
        "rewards": torch.from_numpy(rewards_out).to(device),
        "dones":   torch.from_numpy(dones_out).to(device),
    }


# ── Loss functions ────────────────────────────────────────────────────────────

def kl_divergence_loss(
    pred_latents: torch.Tensor,
    real_latents: torch.Tensor,
) -> torch.Tensor:
    """KL divergence from predicted to real latent distribution."""
    pred_flat = pred_latents.reshape(-1, LATENT_DIM)
    real_flat = real_latents.reshape(-1, LATENT_DIM)

    mu_real  = real_flat.mean(dim=0)
    var_real = real_flat.var(dim=0).clamp(min=1e-6)
    mu_pred  = pred_flat.mean(dim=0)
    var_pred = pred_flat.var(dim=0).clamp(min=1e-6)

    kl_per_dim = 0.5 * (
        torch.log(var_real / var_pred)
        + var_pred / var_real
        + (mu_pred - mu_real) ** 2 / var_real
        - 1.0
    )
    return kl_per_dim.mean()


def rollout_loss_from_latents(
    model:         "DinoWorldModel",
    latents:       torch.Tensor,
    a_seq:         torch.Tensor,
    rollout_steps: int,
    loss_fn,
) -> torch.Tensor:
    """
    Multi-step rollout loss using pre-computed latents as supervision targets.

    Uses the already-encoded latents — no second encoder call needed.
    Pre-allocates z_hist to avoid repeated torch.cat allocations in the loop.

    Args:
        latents       : (B, T, 384) — already encoded, grad-enabled
        a_seq         : (B, T-1) actions
        rollout_steps : steps to unroll (≤ T-1)
    """
    rollout_steps = min(rollout_steps, latents.shape[1] - 1)
    B             = latents.shape[0]
    device        = latents.device

    # Pre-allocate full history buffer — avoids N torch.cat calls in loop
    # Shape: (B, rollout_steps+1, 384)
    z_hist = torch.zeros(B, rollout_steps + 1, LATENT_DIM,
                         device=device, dtype=latents.dtype)
    z_hist[:, 0, :] = latents[:, 0, :].detach()  # seed with real z_0

    step_losses = []

    for t in range(rollout_steps):
        # Use history up to current step
        z_in_t  = z_hist[:, :t+1, :]              # (B, t+1, 384)
        a_in_t  = a_seq[:, :t+1]                  # (B, t+1)

        pred_seq, _, _ = model(z_in_t, a_in_t)
        z_pred = pred_seq[:, -1, :]               # (B, 384)

        # Supervise against ground truth at this step
        z_gt = latents[:, t+1, :].detach()        # (B, 384)
        step_losses.append(loss_fn(z_pred, z_gt))

        # Write prediction into pre-allocated buffer
        z_hist[:, t+1, :] = z_pred.detach()

    return torch.stack(step_losses).mean()


# ── Training epoch ────────────────────────────────────────────────────────────

def train_epoch(
    model:             "DinoWorldModel",
    encoder:           FineTunableDinoV2,
    index:             LazyEpisodeIndex,
    wm_optimizer:      torch.optim.Optimizer,
    enc_optimizer:     torch.optim.Optimizer | None,
    loss_fn,
    device:            torch.device,
    batch_size:        int,
    batches_per_epoch: int,
    kl_weight:         float = 5e-3,
    rollout_steps:     int   = 8,
    rollout_weight:    float = 1.0,
) -> dict:
    """One full joint training epoch."""
    model.train()
    encoder.backbone.train()

    losses         = []
    losses_latent  = []
    losses_reward  = []
    losses_done    = []
    losses_kl      = []
    losses_rollout = []

    for _ in range(batches_per_epoch):

        # A. Sample raw observations from disk (lazy) ──────────────────────────
        batch   = sample_batch(index, batch_size, SEQ_LEN, device,
                               goal_fraction=0.25)
        obs     = batch["obs"]       # (B, T, 64, 64, 3) on device
        actions = batch["actions"]   # (B, T)
        rewards = batch["rewards"]   # (B, T)
        dones   = batch["dones"]     # (B, T)

        # B. Encode ONCE — reused for all losses ───────────────────────────────
        # This is the key optimisation — single DINOv2 forward pass per batch.
        # Gradients flow through encoder for unfrozen blocks.
        latents = encoder(obs)       # (B, T, 384)

        # C. Shifted sequences ─────────────────────────────────────────────────
        z_in     = latents[:, :-1]                   # (B, T-1, 384)
        a_in     = actions[:, :-1]                   # (B, T-1)
        z_target = latents[:, 1:].detach()           # (B, T-1, 384)
        r_target = rewards[:, 1:].unsqueeze(-1)      # (B, T-1, 1)
        d_target = dones[:, 1:].unsqueeze(-1)        # (B, T-1, 1)

        # D. Zero gradients (set_to_none=True is faster than zeroing) ──────────
        wm_optimizer.zero_grad(set_to_none=True)
        if enc_optimizer is not None:
            enc_optimizer.zero_grad(set_to_none=True)

        # E. Forward + losses ──────────────────────────────────────────────────
        pred_next, pred_rew, pred_done = model(z_in, a_in)

        loss_latent = loss_fn(pred_next, z_target)

        # Weighted reward loss — upweight goal transitions 50x
        goal_mask   = (r_target > 0.5).float()
        weights     = 1.0 + goal_mask * 49.0
        loss_reward = (weights * (pred_rew - r_target) ** 2).mean()

        loss_done   = nn.functional.binary_cross_entropy_with_logits(
                          pred_done, d_target.clamp(0, 1)
                      )

        # KL regularisation
        if kl_weight > 0.0:
            loss_kl = kl_divergence_loss(pred_next, z_target)
        else:
            loss_kl = torch.tensor(0.0, device=device)

        # Rollout loss — uses pre-computed latents, NO second encoder call
        if rollout_steps > 0:
            loss_rol = rollout_loss_from_latents(
                model, latents, a_in,
                rollout_steps, loss_fn,
            )
        else:
            loss_rol = torch.tensor(0.0, device=device)

        loss = (LAMBDA_LATENT  * loss_latent
              + LAMBDA_REWARD  * loss_reward
              + LAMBDA_DONE    * loss_done
              + kl_weight      * loss_kl
              + rollout_weight * loss_rol)

        # F. Backward ──────────────────────────────────────────────────────────
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),    max_norm=1.0)
        if enc_optimizer is not None:
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=0.5)

        wm_optimizer.step()
        if enc_optimizer is not None:
            enc_optimizer.step()

        losses.append(loss.item())
        losses_latent.append(loss_latent.item())
        losses_reward.append(loss_reward.item())
        losses_done.append(loss_done.item())
        losses_kl.append(loss_kl.item())
        losses_rollout.append(loss_rol.item())

    n = len(losses)
    return {
        "avg_loss":         sum(losses)         / n,
        "min_loss":         min(losses),
        "max_loss":         max(losses),
        "avg_loss_latent":  sum(losses_latent)  / n,
        "avg_loss_reward":  sum(losses_reward)  / n,
        "avg_loss_done":    sum(losses_done)    / n,
        "avg_loss_kl":      sum(losses_kl)      / n,
        "avg_loss_rollout": sum(losses_rollout) / n,
    }


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:    "DinoWorldModel",
    encoder:  FineTunableDinoV2,
    wm_opt:   torch.optim.Optimizer,
    epoch:    int,
    metrics:  dict,
    save_dir: Path,
    tag:      str = "best",
) -> Path:
    """Save world model + encoder. WM checkpoint compatible with pipeline."""
    save_dir.mkdir(parents=True, exist_ok=True)

    wm_path = save_dir / f"world_model_joint_{tag}.pt"
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "optim_state":  wm_opt.state_dict(),
        "metrics":      metrics,
    }, wm_path)

    enc_path = save_dir / "encoder_finetuned.pt"
    torch.save({
        "epoch":         epoch,
        "encoder_state": encoder.state_dict(),
    }, enc_path)

    return wm_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Joint DINOv2 + World Model Training (Member B)"
    )

    p.add_argument("--raw_dir",           type=str,   default="data/raw")
    p.add_argument("--save_dir",          type=str,   default="checkpoints")
    p.add_argument("--log_file",          type=str,   default="training_log_joint.json")
    p.add_argument("--epochs",            type=int,   default=100)
    p.add_argument("--batches_per_epoch", type=int,   default=200)
    p.add_argument("--batch_size",        type=int,   default=8,
                   help="Keep low (4-8) for CPU — each sample encodes T frames through DINOv2")
    p.add_argument("--save_every",        type=int,   default=10)
    p.add_argument("--kl_weight",         type=float, default=5e-3)
    p.add_argument("--rollout_steps",     type=int,   default=8)
    p.add_argument("--rollout_weight",    type=float, default=1.0)
    p.add_argument(
        "--finetune_blocks", type=int, default=2,
        help="DINOv2 blocks to unfreeze from end. 0=frozen, 2=safe default, 12=full"
    )
    p.add_argument(
        "--encoder_lr", type=float, default=1e-5,
        help="Encoder LR — must be much lower than WM LR to avoid forgetting"
    )
    p.add_argument("--wm_lr",    type=float, default=1e-4)
    p.add_argument("--smoke_test", action="store_true",
                   help="1 epoch × 5 batches — verify no crash")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use all available CPU cores for torch operations
    n_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)
    print(f"[INFO] CPU threads       : {n_threads}")

    if args.smoke_test:
        print("=" * 58)
        print("  SMOKE TEST — 1 epoch, 5 batches")
        print("=" * 58)
        args.epochs            = 1
        args.batches_per_epoch = 5

    print(f"[INFO] Device            : {device}")
    print(f"[INFO] Epochs            : {args.epochs}")
    print(f"[INFO] Batches/epoch     : {args.batches_per_epoch}")
    print(f"[INFO] Batch size        : {args.batch_size}")
    print(f"[INFO] Finetune blocks   : {args.finetune_blocks}")
    print(f"[INFO] Encoder LR        : {args.encoder_lr}")
    print(f"[INFO] WM LR             : {args.wm_lr}")
    print(f"[INFO] KL weight         : {args.kl_weight}")
    print(f"[INFO] Rollout steps     : {args.rollout_steps}")
    print(f"[INFO] Rollout weight    : {args.rollout_weight}")

    # ── Build lazy index — no obs loaded yet ──────────────────────────────────
    index = build_index(Path(args.raw_dir), seq_len=SEQ_LEN)

    # ── Build encoder ──────────────────────────────────────────────────────────
    encoder = FineTunableDinoV2(device)
    encoder.unfreeze_last_n_blocks(args.finetune_blocks)

    # ── Build world model ──────────────────────────────────────────────────────
    if not _REAL_MODEL:
        raise RuntimeError("src.models.transformer required.")

    config = Config.from_params(num_layers=8, mlp_ratio=4, num_heads=8, learning_rate=3e-4, sequence_length=24)
    model  = DinoWorldModel(config).to(device)

    # ── Optimizers ────────────────────────────────────────────────────────────
    wm_optimizer = optim.Adam(model.parameters(), lr=args.wm_lr)
    wm_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        wm_optimizer, T_max=args.epochs, eta_min=1e-6
    )

    enc_trainable = [p for p in encoder.parameters() if p.requires_grad]
    if enc_trainable and args.finetune_blocks > 0:
        enc_optimizer = optim.Adam(enc_trainable, lr=args.encoder_lr)
        enc_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            enc_optimizer, T_max=args.epochs, eta_min=1e-8
        )
        print(f"[INFO] Encoder optimizer : Adam lr={args.encoder_lr} "
              f"({sum(p.numel() for p in enc_trainable):,} params)")
    else:
        enc_optimizer = None
        enc_scheduler = None
        print("[INFO] Encoder fully frozen")

    # ── Training loop ──────────────────────────────────────────────────────────
    save_dir  = Path(args.save_dir)
    log       = []
    best_loss = float("inf")

    print("\n─── Joint Training Start " + "─" * 34)
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        stats = train_epoch(
            model, encoder, index,
            wm_optimizer, enc_optimizer,
            latent_mse_loss, device,
            batch_size        = args.batch_size,
            batches_per_epoch = args.batches_per_epoch,
            kl_weight         = args.kl_weight,
            rollout_steps     = args.rollout_steps,
            rollout_weight    = args.rollout_weight,
        )

        wm_scheduler.step()
        if enc_scheduler is not None:
            enc_scheduler.step()

        elapsed = time.time() - t0
        lr_wm   = wm_optimizer.param_groups[0]["lr"]
        lr_enc  = enc_optimizer.param_groups[0]["lr"] if enc_optimizer else 0.0

        print(
            f"Epoch {epoch:4d}/{args.epochs}  |  "
            f"total={stats['avg_loss']:.4f}  "
            f"latent={stats['avg_loss_latent']:.4f}  "
            f"rew={stats['avg_loss_reward']:.4f}  "
            f"done={stats['avg_loss_done']:.4f}  "
            f"kl={stats['avg_loss_kl']:.4f}  "
            f"rol={stats['avg_loss_rollout']:.4f}  |  "
            f"lr_wm={lr_wm:.2e}  lr_enc={lr_enc:.2e}  |  "
            f"{elapsed:.1f}s"
        )

        if stats["avg_loss"] < best_loss:
            best_loss = stats["avg_loss"]
            p = save_checkpoint(model, encoder, wm_optimizer,
                                epoch, stats, save_dir, "best")
            print(f"  ✔ Best checkpoint → {p}")

        if epoch % args.save_every == 0:
            save_checkpoint(model, encoder, wm_optimizer,
                            epoch, stats, save_dir, f"epoch{epoch:04d}")

        log.append({"epoch": epoch, "lr_wm": lr_wm, "lr_enc": lr_enc, **stats})

    final_path = save_checkpoint(model, encoder, wm_optimizer,
                                 args.epochs, log[-1], save_dir, "final")
    print(f"\n[INFO] Final weights → {final_path}")
    print(f"[INFO] Training time → {(time.time()-t_start)/60:.1f} min")
    print(f"[INFO] Best avg loss → {best_loss:.6f}")

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[INFO] Training log  → {log_path}")

    if args.smoke_test:
        print("\n" + "=" * 58)
        print("  ✅  SMOKE TEST PASSED")
        print("=" * 58)


if __name__ == "__main__":
    main()