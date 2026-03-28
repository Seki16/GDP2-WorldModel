"""
train_dream_dqn_near_goal.py — Near-Goal Dream DQN with optional Dyna
=======================================================================
Research question:
    If we limit WM rollouts to 3-5 steps near the goal (where drift is
    still small), can the Dream DQN learn a transferable policy?

Two modes:
    Pure dream:  --dyna_real_ratio 0.0  (all transitions from WM)
    Dyna hybrid: --dyna_real_ratio 0.3  (30% real, 70% dream)

Key differences from train_dream_dqn.py:
    1. Episodes start from near-goal DINOv2 latents (not [0,0])
    2. Max episode length = 5 steps (not 64)
    3. Dyna real transitions are filtered to near-goal positions only

Usage:
    # Pure dream, SS+TBPTT world model:
    python -m src.scripts.train_dream_dqn_near_goal \
        --wm_checkpoint checkpoints/wm_ss_tbptt/world_model_best.pt \
        --steps 200000

    # Dyna hybrid (recommended):
    python -m src.scripts.train_dream_dqn_near_goal \
        --wm_checkpoint checkpoints/wm_ss_tbptt/world_model_best.pt \
        --steps 200000 --dyna_real_ratio 0.3

    # Quick smoke test:
    python -m src.scripts.train_dream_dqn_near_goal --smoke_test
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

from src.env.maze_env import MazeEnv, MazeConfig
from src.env.world_model_env import WorldModelEnv
from src.models.encoder import DinoV2Encoder

try:
    from src.models.transformer import DinoWorldModel
    from src.models.transformer_configuration import TransformerWMConfiguration as Config
    _REAL_MODEL = True
except ImportError:
    _REAL_MODEL = False

try:
    from src.data.buffer import LatentReplayBuffer
    _REAL_BUFFER = True
except ImportError:
    _REAL_BUFFER = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
LATENT_DIM = 384
ACTION_DIM = 4


# ─────────────────────────────────────────────────────────────────────────────
# Q-Network (same architecture as train_dream_dqn.py)
# ─────────────────────────────────────────────────────────────────────────────
class DreamQNet(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
                 hidden1=256, hidden2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, z):
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────
Transition = collections.namedtuple("Transition", ["z", "a", "r", "z2", "done"])


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self._buf = []
        self._pos = 0
        self._cap = capacity

    def push(self, t):
        if len(self._buf) < self._cap:
            self._buf.append(t)
        else:
            self._buf[self._pos] = t
        self._pos = (self._pos + 1) % self._cap

    def sample(self, batch_size):
        batch = random.sample(self._buf, batch_size)
        z  = torch.tensor(np.array([t.z  for t in batch]), dtype=torch.float32)
        a  = torch.tensor([t.a  for t in batch], dtype=torch.long)
        r  = torch.tensor([t.r  for t in batch], dtype=torch.float32)
        z2 = torch.tensor(np.array([t.z2 for t in batch]), dtype=torch.float32)
        d  = torch.tensor([t.done for t in batch], dtype=torch.float32)
        return z, a, r, z2, d

    def __len__(self):
        return len(self._buf)


def epsilon_schedule(step, eps_start, eps_end, eps_decay):
    if step >= eps_decay:
        return eps_end
    return eps_start + (eps_end - eps_start) * step / eps_decay


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Pre-compute near-goal anchor latents from the REAL environment
# ─────────────────────────────────────────────────────────────────────────────
def compute_near_goal_anchors(
    encoder: DinoV2Encoder,
    device: torch.device,
    grid_size: int = 10,
    obs_size: int = 64,
    max_distance: int = 4,
    maze_seed: int = 0,
    wall_prob: float = 0.20,
) -> list[dict]:
    """
    For every non-wall cell within Manhattan distance `max_distance` of
    the goal [9,9], render the real observation, encode through DINOv2,
    and store as an anchor.

    Returns list of dicts: {pos, distance, latent (384,) numpy}
    """
    env = MazeEnv(MazeConfig(
        grid_size=grid_size, max_steps=64, obs_size=obs_size,
        wall_prob=wall_prob, seed=maze_seed,
    ))
    env.reset()
    grid = env.grid.copy()
    goal = (grid_size - 1, grid_size - 1)

    anchors = []
    for r in range(grid_size):
        for c in range(grid_size):
            dist = abs(r - goal[0]) + abs(c - goal[1])
            if dist > max_distance or dist == 0:  # skip goal itself
                continue
            if grid[r, c] == 1:  # wall
                continue

            # Render real observation from this position
            env.grid = grid.copy()
            env.agent_pos = [r, c]
            env.steps = 0
            obs = env._get_obs()  # (64, 64, 3) uint8

            # Encode through DINOv2 → real latent
            obs_tensor = torch.from_numpy(obs).to(device)
            with torch.no_grad():
                z = encoder.encode(obs_tensor)  # (384,)

            anchors.append({
                "pos": (r, c),
                "distance": dist,
                "latent": z.cpu().numpy(),
            })

    return anchors


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b: Extract near-goal REAL transitions from the latent buffer
#           (for Dyna hybrid mode)
# ─────────────────────────────────────────────────────────────────────────────
def extract_near_goal_real_transitions(
    data_dir: str,
    grid_size: int = 10,
    max_distance: int = 4,
    maze_seed: int = 0,
    wall_prob: float = 0.20,
    obs_size: int = 64,
) -> list[Transition]:
    """
    Load the latent buffer and extract transitions where the agent was
    within `max_distance` of the goal.

    Problem: the buffer stores latents but not agent positions directly.
    Solution: we reconstruct positions from the MazeEnv by replaying
    the stored actions from the episode start position [0,0].
    """
    if not _REAL_BUFFER:
        print("[WARN] Buffer not available — Dyna real transitions disabled")
        return []

    buf = LatentReplayBuffer(capacity_steps=200_000)
    files = sorted(Path(data_dir).glob("*.npz"))
    for f in files:
        d = np.load(f)
        missing = [k for k in ("latents", "actions", "rewards", "dones") if k not in d]
        if missing:
            continue
        buf.add_episode(
            latents=d["latents"], actions=d["actions"],
            rewards=d["rewards"], dones=d["dones"],
        )
        d.close()

    # Reconstruct positions by replaying actions in the maze
    env = MazeEnv(MazeConfig(
        grid_size=grid_size, max_steps=64, obs_size=obs_size,
        wall_prob=wall_prob, seed=maze_seed,
    ))
    env.reset()
    fixed_grid = env.grid.copy()
    goal = (grid_size - 1, grid_size - 1)

    near_goal_transitions = []

    for ep in buf.episodes:
        T = ep.latents.shape[0]
        if T < 2:
            continue

        # Replay actions to reconstruct positions
        env.grid = fixed_grid.copy()
        env.agent_pos = [0, 0]
        env.steps = 0
        positions = [(0, 0)]

        for t in range(T - 1):
            action = int(ep.actions[t])
            env.step(action)
            positions.append(tuple(env.agent_pos))

            # Reset grid each step (step() doesn't change grid, but be safe)
            env.grid = fixed_grid.copy()

        # Now extract transitions where agent was near goal
        for t in range(T - 1):
            pos = positions[t]
            dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            if dist <= max_distance and dist > 0:
                near_goal_transitions.append(Transition(
                    z=ep.latents[t],
                    a=int(ep.actions[t]),
                    r=float(ep.rewards[t]),
                    z2=ep.latents[t + 1],
                    done=float(ep.dones[t]),
                ))

    print(f"[INFO] Extracted {len(near_goal_transitions)} near-goal real "
          f"transitions (distance <= {max_distance})")
    return near_goal_transitions


# ─────────────────────────────────────────────────────────────────────────────
# Loaders (reused from train_dream_dqn.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_world_model(checkpoint_path, device):
    if checkpoint_path is None or not _REAL_MODEL:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"[WARN] Checkpoint not found at {path}")
        return None
    config = Config()
    model = DinoWorldModel(config).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"[INFO] Loaded WM from {path} "
          f"(epoch={ckpt.get('epoch', '?')}, "
          f"loss={ckpt.get('metrics', {}).get('avg_loss', '?')})")
    return model


def load_latent_buffer(data_dir):
    if data_dir is None or not _REAL_BUFFER:
        return None
    buf = LatentReplayBuffer(capacity_steps=200_000)
    files = sorted(Path(data_dir).glob("*.npz"))
    for f in files:
        d = np.load(f)
        missing = [k for k in ("latents", "actions", "rewards", "dones") if k not in d]
        if missing:
            continue
        buf.add_episode(
            latents=d["latents"], actions=d["actions"],
            rewards=d["rewards"], dones=d["dones"],
        )
        d.close()
    print(f"[INFO] Buffer loaded: {len(buf.episodes)} episodes, {buf.total_steps} steps")
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Step 1: Pre-compute near-goal anchors ─────────────────────────────────
    print("\n[STEP 1] Computing near-goal anchor latents from real env...")
    encoder = DinoV2Encoder(device=device)
    anchors = compute_near_goal_anchors(
        encoder, device,
        max_distance=args.max_anchor_distance,
        maze_seed=args.maze_seed,
    )
    print(f"[INFO] {len(anchors)} anchor latents computed:")
    for a in anchors:
        print(f"  pos={a['pos']}, distance={a['distance']}")

    if len(anchors) == 0:
        print("[ERROR] No valid near-goal cells found. Exiting.")
        return

    # We can free the encoder now — it's only needed for anchor computation
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Step 1b: Extract real near-goal transitions (for Dyna) ────────────────
    real_transitions = []
    if args.dyna_real_ratio > 0:
        print(f"\n[STEP 1b] Extracting near-goal real transitions for Dyna "
              f"(ratio={args.dyna_real_ratio})...")
        real_transitions = extract_near_goal_real_transitions(
            data_dir=args.data_dir,
            max_distance=args.max_anchor_distance,
            maze_seed=args.maze_seed,
        )
        if len(real_transitions) == 0:
            print("[WARN] No near-goal real transitions found — "
                  "falling back to pure dream training")
            args.dyna_real_ratio = 0.0

    # ── Load WM + buffer (buffer needed for WorldModelEnv constructor) ────────
    print("\n[STEP 2] Loading world model...")
    wm_model = load_world_model(args.wm_checkpoint, device)
    wm_buffer = load_latent_buffer(args.data_dir)

    env = WorldModelEnv(
        model=wm_model,
        buffer=wm_buffer,
        device=device,
        max_steps=args.max_episode_steps,  # SHORT episodes (5 steps, not 64)
    )

    # ── DQN components ────────────────────────────────────────────────────────
    q = DreamQNet().to(device)
    q_tgt = DreamQNet().to(device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    optimizer = optim.Adam(q.parameters(), lr=args.lr)
    replay_buf = ReplayBuffer(capacity=args.buffer_size)

    # ── Pre-seed replay buffer with real transitions (Dyna) ───────────────────
    if args.dyna_real_ratio > 0 and real_transitions:
        n_seed = min(args.learn_starts, len(real_transitions))
        for i in range(n_seed):
            replay_buf.push(real_transitions[i % len(real_transitions)])
        print(f"[INFO] Pre-seeded replay buffer with {n_seed} real transitions")

    # ── Output setup ──────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "dream_dqn_near_goal_metrics.csv"
    metrics_path.write_text(
        "step,episode,episode_return,episode_len,epsilon,"
        "success,dream_frac,avg_loss\n"
    )

    # ── Reset env with a random near-goal anchor ──────────────────────────────
    def reset_near_goal():
        anchor = random.choice(anchors)
        z_init = torch.tensor(anchor["latent"], dtype=torch.float32,
                              device=device)
        obs, info = env.reset(z_init=z_init)
        return obs, anchor

    obs, current_anchor = reset_near_goal()
    episode_return = 0.0
    episode_len = 0
    episode_idx = 0
    success_count = 0
    recent_returns = []
    recent_losses = []
    metrics_log = []
    dream_transitions_count = 0
    real_transitions_count = 0

    t_start = time.time()
    mode_str = f"Dyna {args.dyna_real_ratio:.0%} real" if args.dyna_real_ratio > 0 else "Pure dream"
    print(f"\n{'─'*60}")
    print(f"  Near-Goal Dream DQN Training")
    print(f"  Mode: {mode_str}")
    print(f"  Steps: {args.steps} | Max episode: {args.max_episode_steps}")
    print(f"  Anchors: {len(anchors)} | Real transitions: {len(real_transitions)}")
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

        # ── Step in WM ────────────────────────────────────────────────────────
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store dream transition
        replay_buf.push(Transition(
            z=obs, a=action, r=float(reward), z2=obs_next, done=float(done),
        ))
        dream_transitions_count += 1

        # ── Dyna: also inject a real transition ───────────────────────────────
        if args.dyna_real_ratio > 0 and real_transitions:
            # For every dream transition, probabilistically add a real one
            # to maintain the desired ratio in the buffer over time
            if random.random() < args.dyna_real_ratio / (1 - args.dyna_real_ratio + 1e-8):
                real_t = random.choice(real_transitions)
                replay_buf.push(real_t)
                real_transitions_count += 1

        episode_return += float(reward)
        episode_len += 1
        obs = obs_next

        # ── Episode end ───────────────────────────────────────────────────────
        if done:
            recent_returns.append(episode_return)
            if len(recent_returns) > 100:
                recent_returns.pop(0)

            # Success = reward head predicted positive reward (genuine goal)
            ep_success = episode_return > 0.0
            if ep_success:
                success_count += 1

            episode_idx += 1

            # Log
            total_trans = dream_transitions_count + real_transitions_count
            dream_frac = dream_transitions_count / max(total_trans, 1)
            avg_loss = np.mean(recent_losses[-100:]) if recent_losses else 0.0

            with open(metrics_path, "a") as f:
                f.write(f"{t},{episode_idx},{episode_return:.4f},"
                        f"{episode_len},{eps:.4f},{int(ep_success)},"
                        f"{dream_frac:.3f},{avg_loss:.6f}\n")

            # Print progress
            if episode_idx % 200 == 0:
                mean_ret = np.mean(recent_returns) if recent_returns else 0
                success_pct = success_count / episode_idx * 100
                elapsed = time.time() - t_start
                print(f"  step {t:>7d} | ep {episode_idx:>5d} | "
                      f"ret {mean_ret:>+6.2f} | success {success_pct:>5.1f}% | "
                      f"eps {eps:.3f} | loss {avg_loss:.4f} | "
                      f"dream/real {dream_transitions_count}/{real_transitions_count} | "
                      f"{elapsed:.0f}s")

            # Reset with new near-goal anchor
            episode_return = 0.0
            episode_len = 0
            obs, current_anchor = reset_near_goal()

        # ── DQN training step ─────────────────────────────────────────────────
        if t >= args.learn_starts and t % args.train_every == 0:
            if len(replay_buf) >= args.batch_size:
                z, a, r, z2, d = replay_buf.sample(args.batch_size)
                z  = z.to(device)
                a  = a.to(device)
                r  = r.to(device)
                z2 = z2.to(device)
                d  = d.to(device)

                # Noise injection (fix 3d — matches train_dream_dqn.py)
                z_noisy  = z  + torch.randn_like(z)  * args.noise_std
                z2_noisy = z2 + torch.randn_like(z2) * args.noise_std

                # Q_theta(z_t, a_t) — learning network prediction
                q_vals = q(z_noisy).gather(1, a.unsqueeze(1)).squeeze(1)

                # max_a' Q_target(z_{t+1}, a') — target network estimate
                with torch.no_grad():
                    q_next = q_tgt(z2_noisy).max(dim=1).values

                # Bellman target: y = r + gamma * max Q_target * (1 - done)
                target = r + args.gamma * q_next * (1 - d)

                # Loss: smooth L1 (Huber) — matches train_dream_dqn.py
                loss = nn.functional.smooth_l1_loss(q_vals, target)

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — matches train_dream_dqn.py
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                optimizer.step()

                recent_losses.append(float(loss.item()))
                if len(recent_losses) > 1000:
                    recent_losses.pop(0)

        # ── Sync target network ───────────────────────────────────────────────
        if t % args.target_update == 0:
            q_tgt.load_state_dict(q.state_dict())

    # ── Save checkpoint ───────────────────────────────────────────────────────
    final_success_pct = success_count / max(episode_idx, 1) * 100
    ckpt = {
        "model_state": q.state_dict(),
        "latent_dim": LATENT_DIM,
        "action_dim": ACTION_DIM,
        "steps_trained": args.steps,
        "final_success_pct": final_success_pct,
        "episodes": episode_idx,
        "mode": "dyna" if args.dyna_real_ratio > 0 else "pure_dream",
        "dyna_real_ratio": args.dyna_real_ratio,
        "max_episode_steps": args.max_episode_steps,
        "max_anchor_distance": args.max_anchor_distance,
        "dream_transitions": dream_transitions_count,
        "real_transitions": real_transitions_count,
    }
    ckpt_path = save_dir / "dqn_dream.pt"
    torch.save(ckpt, ckpt_path)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Episodes       : {episode_idx}")
    print(f"  In-dream success: {final_success_pct:.1f}%")
    print(f"  Dream transitions: {dream_transitions_count}")
    print(f"  Real transitions : {real_transitions_count}")
    print(f"  Time           : {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Near-goal Dream DQN training with optional Dyna hybrid"
    )

    # WM and data
    parser.add_argument("--wm_checkpoint", type=str,
                        default="checkpoints/wm_ss_tbptt/world_model_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--save_dir", type=str,
                        default="checkpoints/dqn_near_goal")

    # Near-goal config
    parser.add_argument("--max_anchor_distance", type=int, default=4,
                        help="Max Manhattan distance from goal for anchors")
    parser.add_argument("--max_episode_steps", type=int, default=5,
                        help="Max steps per dream episode (limits drift)")
    parser.add_argument("--maze_seed", type=int, default=0)

    # Dyna hybrid
    parser.add_argument("--dyna_real_ratio", type=float, default=0.0,
                        help="Fraction of transitions from real buffer "
                             "(0.0 = pure dream, 0.3 = 30%% real)")

    # DQN hyperparameters (matched to train_dream_dqn.py defaults)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--learn_starts", type=int, default=1_000)
    parser.add_argument("--train_every", type=int, default=4,
                        help="Match old code: default 1, but override to 4 as used in practice")
    parser.add_argument("--target_update", type=int, default=1_000)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=int, default=100_000)
    parser.add_argument("--noise_std", type=float, default=0.01,
                        help="Gaussian noise std on latents during training (fix 3d)")

    # Smoke test
    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        args.steps = 500
        args.learn_starts = 50
        args.eps_decay = 200
        print("[SMOKE TEST] Running 500 steps")

    train(args)


if __name__ == "__main__":
    main()