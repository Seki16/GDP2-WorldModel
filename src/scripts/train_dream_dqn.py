"""
src/scripts/train_dream_dqn.py  —  Member E: Task 3 (Dreaming Loop)
====================================================================
Trains a DQN agent entirely inside the latent World Model environment.
The agent NEVER touches the real MazeEnv during training — it learns
exclusively from imagined transitions produced by the World Model.

This is the core CDR deliverable: closing the dreaming loop.

Architecture
------------
  WorldModelEnv  →  obs: np.ndarray (384,)  —  latent vector z_t
  DreamQNet      →  MLP: 384 → 256 → 128 → 4  —  Q(z, a) for a ∈ {0,1,2,3}

The trained policy weights are saved to:
  checkpoints/dqn_dream.pt

These weights are then loaded by the transfer experiment (Task 4) to
execute the dream-trained policy in the real MazeEnv.

Usage
-----
  # Smoke test (10 episodes, no crash check):
  python -m src.scripts.train_dream_dqn --smoke_test

  # Full training run (pure dream, CDR baseline):
  python -m src.scripts.train_dream_dqn --steps 200000

  # Member B: Dyna hybrid run:
  python -m src.scripts.train_dream_dqn --steps 200000 \\
      --dyna_real_ratio 0.5 --max_rollout_steps 5

  # With a specific world model checkpoint:
  python -m src.scripts.train_dream_dqn --wm_checkpoint checkpoints/world_model_best.pt

Interface Contract (GDP Plan §2.3)
------------------------------------
  Latent dim   : 384   (DINOv2 ViT-S/14)
  Action dim   : 4     (Discrete)
  Max steps    : 64    (WorldModelEnv hard limit)

─────────────────────────────────────────────────────────────
MODIFICATION — Member B (Dyna Hybrid, Final Sprint)
─────────────────────────────────────────────────────────────
Added Dyna-style hybrid dream training.

At each learning step the replay buffer is sampled as a mix of:
  - Real transitions: (z, a, r, z', done) taken directly from the
    LatentReplayBuffer (real DINOv2-encoded transitions).
  - Dream transitions: short WM rollouts of ≤ max_rollout_steps
    steps before re-anchoring z_0 to a real buffer latent,
    capping drift accumulation per rollout.

This addresses autoregressive latent drift at inference time by
limiting how far the WM rolls out before returning to a real latent.

New CLI flags:
  --dyna_real_ratio    fraction of each learning batch drawn from
                       real buffer (0.0 = pure dream, 1.0 = pure real).
                       Default: 0.5
  --max_rollout_steps  maximum WM steps before re-anchoring to a real
                       latent. Caps drift accumulation.
                       Default: 5
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── WorldModelEnv (already written — Member B + E) ────────────────────────────
from src.env.world_model_env import WorldModelEnv

# ── Real World Model imports (graceful fallback if not yet available) ─────────
try:
    from src.models.transformer import DinoWorldModel
    from src.models.transformer_configuration import TransformerWMConfiguration as Config
    _REAL_MODEL = True
except ImportError:
    _REAL_MODEL = False
    print("[WARN] src.models.transformer not found — WorldModelEnv will use random stub")

# ── Real Buffer (needed to seed the env's initial latent z_0 and Dyna mixing) ─
try:
    from src.data.buffer import LatentReplayBuffer
    _REAL_BUFFER = True
except ImportError:
    _REAL_BUFFER = False
    print("[WARN] src.data.buffer not found — WorldModelEnv will use random z_0")


# ─────────────────────────────────────────────────────────────────────────────
# Constants (interface contract — do not change)
# ─────────────────────────────────────────────────────────────────────────────

LATENT_DIM = 384
ACTION_DIM = 4


# ─────────────────────────────────────────────────────────────────────────────
# Q-Network: MLP over latent space
# ─────────────────────────────────────────────────────────────────────────────

class DreamQNet(nn.Module):
    """
    3-layer MLP Q-network for the latent action space.

    Input  : z ∈ R^384  (latent observation from WorldModelEnv)
    Output : Q(z, a) for each of the 4 discrete actions

    Why MLP (not CNN)?
      The obs is already a 384-dim semantic latent vector from DINOv2.
      Spatial convolutions are irrelevant — the latent is not an image.
      An MLP is faster, simpler, and well-matched to the input structure.
    """
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_dim: int = ACTION_DIM,
        hidden1:    int = 256,
        hidden2:    int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer (simple, self-contained — independent of LatentReplayBuffer)
# ─────────────────────────────────────────────────────────────────────────────

Transition = collections.namedtuple(
    "Transition", ["z", "a", "r", "z2", "done"]
)


class DreamReplayBuffer:
    """
    Simple circular replay buffer for (z, a, r, z', done) dream transitions.
    Separate from LatentReplayBuffer (which stores world model training data).
    """
    def __init__(self, capacity: int = 100_000):
        self._buf: list[Transition] = []
        self._pos = 0
        self._cap = capacity

    def push(self, t: Transition) -> None:
        if len(self._buf) < self._cap:
            self._buf.append(t)
        else:
            self._buf[self._pos] = t
        self._pos = (self._pos + 1) % self._cap

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        z   = torch.tensor(np.array([t.z  for t in batch]), dtype=torch.float32)
        a   = torch.tensor([t.a  for t in batch], dtype=torch.long)
        r   = torch.tensor([t.r  for t in batch], dtype=torch.float32)
        z2  = torch.tensor(np.array([t.z2 for t in batch]), dtype=torch.float32)
        d   = torch.tensor([t.done for t in batch], dtype=torch.float32)
        return z, a, r, z2, d

    def __len__(self) -> int:
        return len(self._buf)


# ─────────────────────────────────────────────────────────────────────────────
# Dyna Hybrid: short WM rollouts + real buffer mixing  (Member B)
# ─────────────────────────────────────────────────────────────────────────────

def collect_dyna_transitions(
    wm_model:          "DinoWorldModel",
    real_buffer:       "LatentReplayBuffer",
    replay_buf:        DreamReplayBuffer,
    device:            torch.device,
    n_transitions:     int,
    max_rollout_steps: int = 5,
) -> None:
    """
    Collect n_transitions Dyna-style transitions and push them into replay_buf.

    Each rollout:
      1. Sample a real latent z_0 from the LatentReplayBuffer (re-anchor).
      2. Roll the world model forward for ≤ max_rollout_steps steps,
         sampling a random action at each step.
      3. Push each (z_t, a_t, r_t, z_{t+1}, done_t) into replay_buf.
      4. If the rollout reaches max_rollout_steps without done, re-anchor
         z_0 to another real latent for the next rollout.

    This caps drift accumulation to at most max_rollout_steps steps,
    matching the Dyna hybrid design in the sprint plan.

    Args:
        wm_model          : trained DinoWorldModel (eval mode)
        real_buffer       : LatentReplayBuffer with real DINOv2 latents
        replay_buf        : DreamReplayBuffer to push transitions into
        device            : torch device
        n_transitions     : how many transitions to collect this call
        max_rollout_steps : WM steps before re-anchoring (default 5)
    """
    wm_model.eval()
    collected = 0

    with torch.no_grad():
        while collected < n_transitions:
            # ── 1. Re-anchor: sample a real starting latent ───────────────────
            ep  = real_buffer.episodes[
                random.randint(0, len(real_buffer.episodes) - 1)]
            idx = random.randint(0, len(ep.latents) - 1)
            z_t = torch.tensor(
                ep.latents[idx], dtype=torch.float32, device=device
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, 384)

            # ── 2. Short rollout of ≤ max_rollout_steps ───────────────────────
            for _ in range(max_rollout_steps):
                if collected >= n_transitions:
                    break

                # Sample random action
                a_t = random.randint(0, ACTION_DIM - 1)
                a_tensor = torch.tensor(
                    [[a_t]], dtype=torch.long, device=device
                )  # (1, 1)

                # Single WM step
                pred_next, pred_rew, pred_done = wm_model(z_t, a_tensor)
                z_next = pred_next[:, -1:, :]   # (1, 1, 384)
                r_next = float(pred_rew[:, -1, 0].item())
                d_next = float(torch.sigmoid(pred_done[:, -1, 0]).item())
                done   = d_next > 0.9

                # Push transition
                replay_buf.push(Transition(
                    z    = z_t.squeeze().cpu().numpy(),
                    a    = a_t,
                    r    = r_next,
                    z2   = z_next.squeeze().cpu().numpy(),
                    done = float(done),
                ))
                collected += 1

                if done:
                    break   # re-anchor on next outer loop iteration

                z_t = z_next


def collect_real_transitions(
    real_buffer: "LatentReplayBuffer",
    replay_buf:  DreamReplayBuffer,
    n_transitions: int,
) -> None:
    """
    Sample n_transitions directly from the LatentReplayBuffer and push
    them into the DreamReplayBuffer as real (z, a, r, z', done) tuples.

    Real transitions have no drift by definition — mixing them into the
    replay buffer grounds the Q-network in the actual DINOv2 distribution.

    Args:
        real_buffer    : LatentReplayBuffer with real DINOv2 latents
        replay_buf     : DreamReplayBuffer to push transitions into
        n_transitions  : how many real transitions to collect
    """
    for _ in range(n_transitions):
        ep  = real_buffer.episodes[
            random.randint(0, len(real_buffer.episodes) - 1)]
        # Need at least 2 steps for a (z, a, r, z', done) tuple
        if len(ep.latents) < 2:
            continue
        t = random.randint(0, len(ep.latents) - 2)

        replay_buf.push(Transition(
            z    = ep.latents[t].astype(np.float32),
            a    = int(ep.actions[t]),
            r    = float(ep.rewards[t]),
            z2   = ep.latents[t + 1].astype(np.float32),
            done = float(ep.dones[t]),
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Epsilon schedule  (mirrors train_baseline.py)
# ─────────────────────────────────────────────────────────────────────────────

def epsilon_schedule(step: int, eps_start: float, eps_end: float, eps_decay: int) -> float:
    """Linear decay from eps_start → eps_end over eps_decay steps."""
    if step >= eps_decay:
        return eps_end
    return eps_start + (eps_end - eps_start) * step / eps_decay


# ─────────────────────────────────────────────────────────────────────────────
# World Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_world_model(checkpoint_path: str | None, device: torch.device):
    """
    Load DinoWorldModel from checkpoint, or return None to use the random stub.
    """
    if checkpoint_path is None or not _REAL_MODEL:
        print("[WARN] No world model checkpoint — WorldModelEnv will use random stub")
        return None

    path = Path(checkpoint_path)
    if not path.exists():
        print(f"[WARN] Checkpoint not found at {path} — using random stub")
        return None

    config = Config.from_params(num_layers=8, mlp_ratio=4, num_heads=8, learning_rate=3e-4, sequence_length=24)
    model  = DinoWorldModel(config).to(device)
    ckpt   = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:    print(f"[WARN] Missing keys (random-init): {missing}")
    if unexpected: print(f"[WARN] Unexpected keys (ignored): {unexpected}")
    model.eval()
    print(f"[INFO] Loaded world model from {path}  "
          f"(epoch={ckpt.get('epoch', '?')}, "
          f"loss={ckpt.get('metrics', {}).get('avg_loss', '?')})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Buffer loader
# ─────────────────────────────────────────────────────────────────────────────

def load_latent_buffer(data_dir: str | None):
    """
    Load LatentReplayBuffer from processed latent files, or return None.
    Used by WorldModelEnv to seed z_0, and by Dyna hybrid for real transitions.
    """
    if data_dir is None or not _REAL_BUFFER:
        print("[WARN] No data_dir / buffer — WorldModelEnv will use random z_0")
        return None

    buf   = LatentReplayBuffer(capacity_steps=200_000)
    files = sorted(Path(data_dir).glob("*.npz"))
    if not files:
        print(f"[WARN] No .npz files in {data_dir} — using random z_0")
        return None

    for f in files:
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

    print(f"[INFO] Buffer loaded: {len(buf.episodes)} episodes, {buf.total_steps} steps")
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Dyna hybrid config summary ────────────────────────────────────────────
    dyna_enabled = args.dyna_real_ratio > 0.0 and _REAL_BUFFER
    if dyna_enabled:
        print(f"[INFO] Dyna hybrid ENABLED")
        print(f"[INFO]   dyna_real_ratio    : {args.dyna_real_ratio}")
        print(f"[INFO]   max_rollout_steps  : {args.max_rollout_steps}")
    else:
        print(f"[INFO] Dyna hybrid DISABLED (pure dream training)")

    # ── World model + env ─────────────────────────────────────────────────────
    wm_model = load_world_model(args.wm_checkpoint, device)
    buf      = load_latent_buffer(args.data_dir)

    env = WorldModelEnv(
        model    = wm_model,
        buffer   = buf,
        device   = device,
        max_steps= 64,
    )
    print(f"[INFO] WorldModelEnv ready: {env}")

    # ── DQN components ────────────────────────────────────────────────────────
    q     = DreamQNet().to(device)
    q_tgt = DreamQNet().to(device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    optimizer  = optim.Adam(q.parameters(), lr=args.lr)
    replay_buf = DreamReplayBuffer(capacity=args.buffer_size)

    # ── Output setup ──────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "dream_dqn_metrics.csv"
    out_path.write_text("step,episode,episode_return,episode_len,epsilon,success\n")

    # ── Training state ────────────────────────────────────────────────────────
    obs, _         = env.reset()
    episode_return = 0.0
    episode_len    = 0
    episode_idx    = 0
    success_count  = 0
    recent_returns: list[float] = []
    metrics_log    = []

    t_start = time.time()
    print(f"\n{'─'*60}")
    print(f"  Dream DQN Training  |  steps={args.steps}  |  device={device}")
    print(f"{'─'*60}")

    for t in range(1, args.steps + 1):

        # ── Epsilon-greedy action ─────────────────────────────────────────────
        eps = epsilon_schedule(t, args.eps_start, args.eps_end, args.eps_decay)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            z_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(q(z_t), dim=1).item())

        # ── Step inside the dreamed world ─────────────────────────────────────
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ── Store dream transition ────────────────────────────────────────────
        replay_buf.push(Transition(
            z    = obs,
            a    = action,
            r    = float(reward),
            z2   = obs_next,
            done = float(done),
        ))

        episode_return += float(reward)
        episode_len    += 1
        obs             = obs_next

        # ── Dyna hybrid: populate replay buffer with real + short-rollout data ─
        # Done before the learning step so the mixed buffer is ready immediately.
        # Only runs once we have a real buffer and Dyna is enabled.
        if (dyna_enabled
                and wm_model is not None
                and buf is not None
                and len(buf.episodes) > 0
                and t % args.dyna_collect_every == 0):

            # How many transitions to add this step
            n_real  = max(1, int(args.dyna_collect_n * args.dyna_real_ratio))
            n_dream = max(0, args.dyna_collect_n - n_real)

            # Real transitions — zero drift by definition
            collect_real_transitions(buf, replay_buf, n_real)

            # Short WM rollouts — capped at max_rollout_steps
            if n_dream > 0:
                collect_dyna_transitions(
                    wm_model, buf, replay_buf, device,
                    n_transitions     = n_dream,
                    max_rollout_steps = args.max_rollout_steps,
                )

        # ── Episode end ───────────────────────────────────────────────────────
        if done:
            recent_returns.append(episode_return)
            if len(recent_returns) > 100:
                recent_returns.pop(0)

            success = episode_return > 0.5
            if success:
                success_count += 1

            out_path.open("a").write(
                f"{t},{episode_idx},{episode_return:.4f},"
                f"{episode_len},{eps:.4f},{int(success)}\n"
            )
            episode_idx   += 1
            episode_return = 0.0
            episode_len    = 0
            obs, _         = env.reset()

        # ── Learning step ─────────────────────────────────────────────────────
        if (t >= args.learn_starts
                and t % args.train_every == 0
                and len(replay_buf) >= args.batch_size):

            z_b, a_b, r_b, z2_b, d_b = replay_buf.sample(args.batch_size)
            z_b  = z_b.to(device)
            a_b  = a_b.to(device)
            r_b  = r_b.to(device)
            z2_b = z2_b.to(device)
            d_b  = d_b.to(device)

            # 3d — Noise injection: forces robustness to latent distribution shift
            z_b_noisy  = z_b  + torch.randn_like(z_b)  * args.noise_std
            z2_b_noisy = z2_b + torch.randn_like(z2_b) * args.noise_std

            q_sa = q(z_b_noisy).gather(1, a_b.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                target = r_b + (1.0 - d_b) * args.gamma * q_tgt(z2_b_noisy).max(1).values

            loss = nn.functional.smooth_l1_loss(q_sa, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            optimizer.step()

        # ── Target network sync ───────────────────────────────────────────────
        if t % args.target_update == 0:
            q_tgt.load_state_dict(q.state_dict())

        # ── Logging ───────────────────────────────────────────────────────────
        if t % args.log_every == 0:
            avg = float(np.mean(recent_returns)) if recent_returns else 0.0
            sr  = 100.0 * success_count / max(1, episode_idx)
            elapsed = time.time() - t_start
            print(
                f"[Dream DQN] step={t:7d}  eps={eps:.3f}  "
                f"ep={episode_idx:5d}  avg_ret={avg:+.3f}  "
                f"success={sr:.1f}%  ({elapsed:.0f}s)"
            )
            metrics_log.append({
                "step":        t,
                "episode":     episode_idx,
                "avg_return":  avg,
                "success_pct": sr,
                "epsilon":     eps,
            })

        # ── Smoke-test early exit ─────────────────────────────────────────────
        if args.smoke_test and episode_idx >= 10:
            print("[SMOKE TEST] 10 episodes completed without crash ✅")
            break

    # ── Save weights ──────────────────────────────────────────────────────────
    final_sr = 100.0 * success_count / max(1, episode_idx)
    ckpt_path = save_dir / "dqn_dream.pt"
    torch.save({
        "model_state":        q.state_dict(),
        "latent_dim":         LATENT_DIM,
        "action_dim":         ACTION_DIM,
        "wm_checkpoint":      args.wm_checkpoint,
        "steps_trained":      args.steps,
        "final_success_pct":  final_sr,
        "dyna_real_ratio":    args.dyna_real_ratio,
        "max_rollout_steps":  args.max_rollout_steps,
        "args":               vars(args),
    }, ckpt_path)

    metrics_path = save_dir / "dream_dqn_training_log.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"\n[Dream DQN] DONE")
    print(f"[Dream DQN] Weights     → {ckpt_path}")
    print(f"[Dream DQN] Metrics CSV → {out_path}")
    print(f"[Dream DQN] Metrics JSON→ {metrics_path}")
    print(f"[Dream DQN] Episodes: {episode_idx}  |  Success rate: {final_sr:.1f}%")
    if recent_returns:
        print(f"[Dream DQN] Avg return (last 100 ep): {np.mean(recent_returns):+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DQN inside the Latent World Model (Dreaming Loop)"
    )

    # World model
    p.add_argument("--wm_checkpoint", type=str,
                   default="checkpoints/world_model_best.pt",
                   help="Path to trained world model checkpoint")
    p.add_argument("--data_dir", type=str,
                   default="data/processed",
                   help="Directory of processed latent .npz files (for z_0 seeding + Dyna)")

    # Training
    p.add_argument("--steps",        type=int,   default=200_000)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--buffer_size",  type=int,   default=100_000)
    p.add_argument("--learn_starts", type=int,   default=1_000)
    p.add_argument("--train_every",  type=int,   default=1)
    p.add_argument("--target_update",type=int,   default=1_000)

    # Epsilon schedule
    p.add_argument("--eps_start",    type=float, default=1.0)
    p.add_argument("--eps_end",      type=float, default=0.05)
    p.add_argument("--eps_decay",    type=int,   default=100_000)

    # Logging / output
    p.add_argument("--log_every",    type=int,   default=5_000)
    p.add_argument("--save_dir",     type=str,   default="checkpoints")

    # Dev / noise
    p.add_argument("--noise_std",    type=float, default=0.01,
                   help="3d: Gaussian noise std added to latents during learning")
    p.add_argument("--smoke_test",   action="store_true",
                   help="Run 10 dream episodes then exit (no crash = pass)")

    # ── Dyna hybrid (Member B) ────────────────────────────────────────────────
    p.add_argument(
        "--dyna_real_ratio", type=float, default=0.5,
        help=(
            "Fraction of Dyna batch drawn from real buffer transitions. "
            "0.0 = pure dream (no Dyna), 1.0 = all real transitions. "
            "Default: 0.5 (equal mix of real and short WM rollouts)."
        ),
    )
    p.add_argument(
        "--max_rollout_steps", type=int, default=5,
        help=(
            "Maximum WM rollout steps before re-anchoring to a real latent. "
            "Caps autoregressive drift accumulation per rollout. "
            "Sprint plan target: ≤5 steps. Default: 5."
        ),
    )
    p.add_argument(
        "--dyna_collect_every", type=int, default=4,
        help=(
            "Collect Dyna transitions every N environment steps. "
            "Lower = more frequent mixing, higher = less overhead. "
            "Default: 4."
        ),
    )
    p.add_argument(
        "--dyna_collect_n", type=int, default=8,
        help=(
            "Number of Dyna transitions to collect each time. "
            "Splits by dyna_real_ratio into real vs dream. "
            "Default: 8."
        ),
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
