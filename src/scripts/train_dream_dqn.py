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

  # Full training run:
  python -m src.scripts.train_dream_dqn --steps 200000

  # With a specific world model checkpoint:
  python -m src.scripts.train_dream_dqn --wm_checkpoint checkpoints/world_model_best.pt

Interface Contract (GDP Plan §2.3)
------------------------------------
  Latent dim   : 384   (DINOv2 ViT-S/14)
  Action dim   : 4     (Discrete)
  Max steps    : 64    (WorldModelEnv hard limit)
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

# ── Real Buffer (needed to seed the env's initial latent z_0) ────────────────
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
        """
        Parameters
        ----------
        z : (B, 384) float tensor

        Returns
        -------
        q : (B, 4) float tensor  —  Q-values for each action
        """
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

    Returns
    -------
    model : DinoWorldModel | None
        None → WorldModelEnv will fall back to _RandomWorldModel (smoke-test mode)
    """
    if checkpoint_path is None or not _REAL_MODEL:
        print("[WARN] No world model checkpoint — WorldModelEnv will use random stub")
        return None

    path = Path(checkpoint_path)
    if not path.exists():
        print(f"[WARN] Checkpoint not found at {path} — using random stub")
        return None

    config = Config()
    model  = DinoWorldModel(config).to(device)
    ckpt   = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:   print(f"[WARN] Missing keys (random-init): {missing}")
    if unexpected: print(f"[WARN] Unexpected keys (ignored): {unexpected}")
    model.eval()
    print(f"[INFO] Loaded world model from {path}  "
          f"(epoch={ckpt.get('epoch', '?')}, "
          f"loss={ckpt.get('metrics', {}).get('avg_loss', '?')})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Buffer loader (for WorldModelEnv initial latent z_0)
# ─────────────────────────────────────────────────────────────────────────────

def load_latent_buffer(data_dir: str | None):
    """
    Load LatentReplayBuffer from processed latent files, or return None.
    Used by WorldModelEnv to seed the initial state z_0.
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

        # ── Store transition ──────────────────────────────────────────────────
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

        # ── Episode end ───────────────────────────────────────────────────────
        if done:
            recent_returns.append(episode_return)
            if len(recent_returns) > 100:
                recent_returns.pop(0)

            # True success = world model predicted done BEFORE the step limit
            # If terminated fires exactly at max_steps, it's the done_head
            # misfiring at truncation, not a genuine goal-reach
            success = episode_return > 0.5  # real goal gives +1.0 reward
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

            # Q(z, a) for the taken action
            q_sa = q(z_b_noisy).gather(1, a_b.view(-1, 1)).squeeze(1)

            # Target: r + γ * max_a' Q_tgt(z', a')
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
        "model_state":       q.state_dict(),
        "latent_dim":        LATENT_DIM,
        "action_dim":        ACTION_DIM,
        "wm_checkpoint":     args.wm_checkpoint,
        "steps_trained":     args.steps,
        "final_success_pct": final_sr,
        "args":              vars(args),
    }, ckpt_path)

    # ── Save metrics JSON (for CDR plots) ──────────────────────────────────────
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
                   help="Directory of processed latent .npz files (for z_0 seeding)")

    # Training
    p.add_argument("--steps",        type=int,   default=200_000)
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Adam learning rate (lower than baseline: latent space is smoother)")
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--buffer_size",  type=int,   default=100_000,
                   help="Replay buffer capacity (larger than baseline: no env resets)")
    p.add_argument("--learn_starts", type=int,   default=1_000,
                   help="Steps before learning begins (fill buffer first)")
    p.add_argument("--train_every",  type=int,   default=1,
                   help="Gradient update every N steps")
    p.add_argument("--target_update",type=int,   default=1_000,
                   help="Hard target network sync every N steps")

    # Epsilon schedule
    p.add_argument("--eps_start",    type=float, default=1.0)
    p.add_argument("--eps_end",      type=float, default=0.05)
    p.add_argument("--eps_decay",    type=int,   default=100_000)

    # Logging / output
    p.add_argument("--log_every",    type=int,   default=5_000)
    p.add_argument("--save_dir",     type=str,   default="checkpoints")

    # Dev mode
    p.add_argument("--noise_std",    type=float, default=0.01,
                   help="3d: Gaussian noise std added to latents during learning")
    p.add_argument("--smoke_test",   action="store_true",
                   help="Run 10 dream episodes then exit (no crash = pass)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)