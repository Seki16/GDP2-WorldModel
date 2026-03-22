"""
src/scripts/train_dream_dqn.py  —  Member E: Task 3 (Dreaming Loop)
====================================================================
Trains a DQN agent inside the latent World Model environment.

CDR baseline: agent trains exclusively on dream transitions (pure WM rollouts).
E.1 Dyna hybrid: each learning batch is a mix of:
    - Real transitions sampled directly from the LatentReplayBuffer (real latents)
    - Short WM rollouts of at most --dyna_max_rollout_steps steps, starting from
      a real latent anchor sampled from the buffer.

Set --dyna_real_ratio 0.0 to reproduce the CDR baseline exactly (no real transitions,
unlimited rollout length — pure dream training as before).

Architecture
------------
  WorldModelEnv  →  obs: np.ndarray (384,)  —  latent vector z_t
  DreamQNet      →  MLP: 384 → 256 → 128 → 4  —  Q(z, a) for a ∈ {0,1,2,3}

Usage
-----
  # Smoke test:
  python -m src.scripts.train_dream_dqn --smoke_test

  # CDR baseline reproduction (Dyna disabled):
  python -m src.scripts.train_dream_dqn --steps 200000 --dyna_real_ratio 0.0

  # Dyna hybrid run (E.1):
  python -m src.scripts.train_dream_dqn --steps 200000 --dyna_real_ratio 0.3 --dyna_max_rollout_steps 5

  # Dyna hybrid with KL-trained WM (recommended E.1 run):
  python -m src.scripts.train_dream_dqn --steps 200000 --wm_checkpoint checkpoints/kl_run/world_model_best.pt --dyna_real_ratio 0.3 --dyna_max_rollout_steps 5 --save_dir checkpoints/dyna_run

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

# ── WorldModelEnv ─────────────────────────────────────────────────────────────
from src.env.world_model_env import WorldModelEnv

# ── Real World Model imports ──────────────────────────────────────────────────
try:
    from src.models.transformer import DinoWorldModel
    from src.models.transformer_configuration import TransformerWMConfiguration as Config
    _REAL_MODEL = True
except ImportError:
    _REAL_MODEL = False
    print("[WARN] src.models.transformer not found — WorldModelEnv will use random stub")

# ── Real Buffer ───────────────────────────────────────────────────────────────
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

# DYNA HYBRID (E.1): defaults
# dyna_real_ratio        — fraction of each training batch drawn from the real
#                          buffer. 0.3 means 30% real, 70% dream.
# dyna_max_rollout_steps — WM rollouts are capped at this many steps before
#                          re-anchoring to a new real latent. Keeps drift small.
DYNA_REAL_RATIO_DEFAULT        = 0.3
DYNA_MAX_ROLLOUT_STEPS_DEFAULT = 5


# ─────────────────────────────────────────────────────────────────────────────
# Q-Network: MLP over latent space
# ─────────────────────────────────────────────────────────────────────────────

class DreamQNet(nn.Module):
    """
    3-layer MLP Q-network for the latent action space.

    Input  : z ∈ R^384  (latent observation from WorldModelEnv)
    Output : Q(z, a) for each of the 4 discrete actions
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
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

Transition = collections.namedtuple(
    "Transition", ["z", "a", "r", "z2", "done"]
)


class DreamReplayBuffer:
    """
    Circular replay buffer for (z, a, r, z', done) transitions.
    Stores both dream transitions (from WM rollouts) and real transitions
    (from the LatentReplayBuffer, added during Dyna hybrid batch construction).
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
# Epsilon schedule
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
    if missing:    print(f"[WARN] Missing keys (random-init): {missing}")
    if unexpected: print(f"[WARN] Unexpected keys (ignored):  {unexpected}")
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
    Load LatentReplayBuffer from processed latent files.
    Used both by WorldModelEnv (z_0 seeding) and by the Dyna hybrid
    (real transition sampling).
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
# DYNA HYBRID (E.1): Real transition sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_real_transitions(
    buf,
    n: int,
    device: torch.device,
) -> list[Transition]:
    """
    Sample n individual (z, a, r, z', done) transitions directly from the
    LatentReplayBuffer. These are real DINOv2-encoded latents — guaranteed
    to be in-distribution. Used to fill the real portion of Dyna batches.

    Each transition is drawn by:
      1. Picking a random episode.
      2. Picking a random step t within that episode (not the last step,
         so z' = latents[t+1] is always available).

    Parameters
    ----------
    buf    : LatentReplayBuffer
    n      : number of transitions to sample
    device : torch.device (unused here — tensors are built later in the
             learning step, kept as numpy for consistency with DreamReplayBuffer)

    Returns
    -------
    list of n Transition namedtuples with numpy arrays for z and z2
    """
    transitions = []
    episodes = buf.episodes

    for _ in range(n):
        # Pick a random episode that has at least 2 steps (need z and z')
        ep = episodes[np.random.randint(len(episodes))]
        T  = ep.latents.shape[0]

        if T < 2:
            # Edge case: single-step episode — use step 0 with z'=z (no motion)
            t = 0
            z  = ep.latents[0].astype(np.float32)
            z2 = ep.latents[0].astype(np.float32)
        else:
            t  = np.random.randint(0, T - 1)    # t in [0, T-2] so t+1 is valid
            z  = ep.latents[t].astype(np.float32)
            z2 = ep.latents[t + 1].astype(np.float32)

        a    = int(ep.actions[t])
        r    = float(ep.rewards[t])
        done = float(ep.dones[t])

        transitions.append(Transition(z=z, a=a, r=r, z2=z2, done=done))

    return transitions


def sample_dyna_rollout(
    wm_model,
    buf,
    q_net:         DreamQNet,
    device:        torch.device,
    max_steps:     int,
    eps:           float,
    noise_std:     float,
) -> list[Transition]:
    """
    Generate one short WM rollout of at most max_steps steps, starting from
    a real latent anchor sampled from the buffer.

    The rollout uses the current DQN policy (epsilon-greedy) to select actions,
    exactly as the main dreaming loop does. This means the Dyna rollouts explore
    the same latent regions the DQN is currently valuing.

    After max_steps steps the rollout stops — z_final is discarded and never
    fed back into the WM. This caps drift accumulation to at most max_steps
    compounding errors, as opposed to 64 in the original dreaming loop.

    Parameters
    ----------
    wm_model   : DinoWorldModel (eval mode, no grad)
    buf        : LatentReplayBuffer — provides the real anchor z_0
    q_net      : DreamQNet — used for action selection (epsilon-greedy)
    device     : torch.device
    max_steps  : int — hard cap on rollout length (default 5)
    eps        : float — current epsilon for action selection
    noise_std  : float — noise std for robustness (applied to z when selecting
                 action, NOT to the z stored in the transition — we want the
                 stored latents to be as clean as possible)

    Returns
    -------
    list of Transition namedtuples (at most max_steps items)
    """
    transitions = []
    episodes = buf.episodes

    # ── Sample a real anchor latent z_0 ──────────────────────────────────────
    ep  = episodes[np.random.randint(len(episodes))]
    idx = np.random.randint(ep.latents.shape[0])
    z_anchor = ep.latents[idx].astype(np.float32)      # (384,) numpy

    # Convert to WM input format: (1, 1, 384) tensor
    z_t = torch.tensor(z_anchor, dtype=torch.float32, device=device)
    z_t = z_t.unsqueeze(0).unsqueeze(0)                # (1, 1, 384)

    for _ in range(max_steps):
        # ── Epsilon-greedy action (same logic as main loop) ───────────────────
        z_obs = z_t.squeeze().cpu().numpy()            # (384,) for DQN input
        if random.random() < eps:
            action = random.randint(0, ACTION_DIM - 1)
        else:
            z_input = torch.tensor(z_obs, dtype=torch.float32, device=device)
            z_noisy = z_input + torch.randn_like(z_input) * noise_std
            with torch.no_grad():
                action = int(torch.argmax(q_net(z_noisy.unsqueeze(0)), dim=1).item())

        # ── WM forward step ───────────────────────────────────────────────────
        a_tensor = torch.tensor([[action]], dtype=torch.long, device=device)
        with torch.no_grad():
            pred_next, pred_rew, pred_done = wm_model(z_t, a_tensor)

        # Extract outputs — match WorldModelEnv.step() exactly
        z_next     = pred_next[:, -1:, :]              # (1, 1, 384)
        wm_reward  = float(pred_rew[:, -1, 0].item())
        # done_head disabled (same as CDR final run) — truncation handled
        # externally by rollout length cap
        terminated = False
        done       = float(terminated)

        z_next_obs = z_next.squeeze().cpu().numpy()    # (384,) numpy

        transitions.append(Transition(
            z    = z_obs,
            a    = action,
            r    = wm_reward,
            z2   = z_next_obs,
            done = done,
        ))

        # Advance state
        z_t = z_next

        if terminated:
            break

    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── World model + buffer + env ────────────────────────────────────────────
    wm_model = load_world_model(args.wm_checkpoint, device)
    buf      = load_latent_buffer(args.data_dir)

    env = WorldModelEnv(
        model     = wm_model,
        buffer    = buf,
        device    = device,
        max_steps = 64,
    )
    print(f"[INFO] WorldModelEnv ready: {env}")

    # DYNA HYBRID (E.1): check if Dyna is enabled and buffer is available
    dyna_enabled = (
        args.dyna_real_ratio > 0.0
        and buf is not None
        and len(buf.episodes) > 0
        and wm_model is not None
    )
    if args.dyna_real_ratio > 0.0 and not dyna_enabled:
        print("[WARN] Dyna hybrid requested but buffer or WM not available — "
              "falling back to pure dream training")

    print(f"[INFO] Dyna hybrid        : {'ENABLED' if dyna_enabled else 'DISABLED (CDR baseline)'}")
    if dyna_enabled:
        print(f"[INFO]   real_ratio        : {args.dyna_real_ratio}")
        print(f"[INFO]   max_rollout_steps : {args.dyna_max_rollout_steps}")

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

        # ── Store dream transition in replay buffer ────────────────────────────
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

            # DYNA HYBRID (E.1): ───────────────────────────────────────────────
            # Build a mixed batch: some transitions from the real buffer (real
            # DINOv2 latents, guaranteed in-distribution), the rest from short
            # WM rollouts (capped at dyna_max_rollout_steps to limit drift).
            #
            # When dyna_enabled=False this entire block is skipped and we fall
            # through to the standard replay_buf.sample() — exact CDR baseline.
            if dyna_enabled:
                n_real  = max(1, int(args.batch_size * args.dyna_real_ratio))
                n_dream = args.batch_size - n_real

                # Real portion: individual transitions from LatentReplayBuffer
                real_transitions = sample_real_transitions(buf, n_real, device)

                # Dream portion: short WM rollouts from real anchors.
                # We generate rollouts until we have enough transitions.
                dream_transitions = []
                while len(dream_transitions) < n_dream:
                    rollout = sample_dyna_rollout(
                        wm_model   = wm_model,
                        buf        = buf,
                        q_net      = q,
                        device     = device,
                        max_steps  = args.dyna_max_rollout_steps,
                        eps        = eps,
                        noise_std  = args.noise_std,
                    )
                    dream_transitions.extend(rollout)

                # Trim dream to exactly n_dream (rollouts may overshoot slightly)
                dream_transitions = dream_transitions[:n_dream]

                # Combine and convert to tensors
                all_transitions = real_transitions + dream_transitions
                z_b  = torch.tensor(
                    np.array([t.z    for t in all_transitions]),
                    dtype=torch.float32, device=device)
                a_b  = torch.tensor(
                    [t.a             for t in all_transitions],
                    dtype=torch.long, device=device)
                r_b  = torch.tensor(
                    [t.r             for t in all_transitions],
                    dtype=torch.float32, device=device)
                z2_b = torch.tensor(
                    np.array([t.z2   for t in all_transitions]),
                    dtype=torch.float32, device=device)
                d_b  = torch.tensor(
                    [t.done          for t in all_transitions],
                    dtype=torch.float32, device=device)

            else:
                # CDR baseline: pure dream replay sample (unchanged)
                z_b, a_b, r_b, z2_b, d_b = replay_buf.sample(args.batch_size)
                z_b  = z_b.to(device)
                a_b  = a_b.to(device)
                r_b  = r_b.to(device)
                z2_b = z2_b.to(device)
                d_b  = d_b.to(device)

            # ── Noise injection (fix 3d — unchanged) ──────────────────────────
            z_b_noisy  = z_b  + torch.randn_like(z_b)  * args.noise_std
            z2_b_noisy = z2_b + torch.randn_like(z2_b) * args.noise_std

            # ── Q-learning update (unchanged) ─────────────────────────────────
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
    final_sr  = 100.0 * success_count / max(1, episode_idx)
    ckpt_path = save_dir / "dqn_dream.pt"
    torch.save({
        "model_state":             q.state_dict(),
        "latent_dim":              LATENT_DIM,
        "action_dim":              ACTION_DIM,
        "wm_checkpoint":           args.wm_checkpoint,
        "steps_trained":           args.steps,
        "final_success_pct":       final_sr,
        # DYNA HYBRID (E.1): record Dyna config in checkpoint for traceability
        "dyna_real_ratio":         args.dyna_real_ratio,
        "dyna_max_rollout_steps":  args.dyna_max_rollout_steps,
        "args":                    vars(args),
    }, ckpt_path)

    metrics_path = save_dir / "dream_dqn_training_log.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"\n[Dream DQN] DONE")
    print(f"[Dream DQN] Weights      → {ckpt_path}")
    print(f"[Dream DQN] Metrics CSV  → {out_path}")
    print(f"[Dream DQN] Metrics JSON → {metrics_path}")
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
                   default="checkpoints/world_model_best.pt")
    p.add_argument("--data_dir", type=str,
                   default="data/processed")

    # Training
    p.add_argument("--steps",         type=int,   default=200_000)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--buffer_size",   type=int,   default=100_000)
    p.add_argument("--learn_starts",  type=int,   default=1_000)
    p.add_argument("--train_every",   type=int,   default=1)
    p.add_argument("--target_update", type=int,   default=1_000)

    # Epsilon schedule
    p.add_argument("--eps_start",     type=float, default=1.0)
    p.add_argument("--eps_end",       type=float, default=0.05)
    p.add_argument("--eps_decay",     type=int,   default=100_000)

    # Logging / output
    p.add_argument("--log_every",     type=int,   default=5_000)
    p.add_argument("--save_dir",      type=str,   default="checkpoints")

    # Fix 3d
    p.add_argument("--noise_std",     type=float, default=0.01)

    # DYNA HYBRID (E.1): ───────────────────────────────────────────────────────
    # dyna_real_ratio        — fraction of each training batch drawn from the
    #                          real LatentReplayBuffer. 0.0 = CDR baseline.
    # dyna_max_rollout_steps — WM rollout length cap. Keeps drift bounded.
    #                          Ignored when dyna_real_ratio=0.0.
    p.add_argument("--dyna_real_ratio",
                   type=float, default=DYNA_REAL_RATIO_DEFAULT,
                   help="Fraction of batch from real buffer. 0.0 = CDR baseline.")
    p.add_argument("--dyna_max_rollout_steps",
                   type=int,   default=DYNA_MAX_ROLLOUT_STEPS_DEFAULT,
                   help="Max WM rollout steps per Dyna rollout. Default 5.")

    # Dev mode
    p.add_argument("--smoke_test",    action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)