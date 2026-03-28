"""
train_dream_dqn_near_goal_latent_reward.py — Fix B: Latent-distance reward
============================================================================
Bypasses the broken WM reward head entirely. Instead of relying on the
reward head to detect the goal, we compute reward directly from L2
distance between the current dreamed latent and the known goal latent:

    r_t = +1.0  if ||z_t - z_goal||_2 < tau    (goal reached)
    r_t = -0.01 otherwise                       (step penalty)

This tests: if we fix the reward signal, can the WM dynamics support RL?

The goal latent z_goal is computed once at startup by encoding the real
goal observation (agent at [9,9]) through DINOv2.

Usage:
    # Near-goal Dyna with latent reward (recommended):
    python -m src.scripts.train_dream_dqn_near_goal_latent_reward \
        --wm_checkpoint checkpoints/wm_ss_tbptt/world_model_best.pt \
        --steps 200000 --dyna_real_ratio 0.3

    # Pure dream (no real transitions):
    python -m src.scripts.train_dream_dqn_near_goal_latent_reward \
        --wm_checkpoint checkpoints/wm_ss_tbptt/world_model_best.pt \
        --steps 200000

    # Smoke test:
    python -m src.scripts.train_dream_dqn_near_goal_latent_reward --smoke_test
"""

from __future__ import annotations

import argparse
import collections
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


LATENT_DIM = 384
ACTION_DIM = 4


# ─────────────────────────────────────────────────────────────────────────────
# Q-Network (identical to train_dream_dqn.py)
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
# Compute near-goal anchors + goal latent
# ─────────────────────────────────────────────────────────────────────────────
def compute_anchors_and_goal(
    encoder, device, grid_size=10, obs_size=64,
    max_distance=4, maze_seed=0, wall_prob=0.20,
):
    env = MazeEnv(MazeConfig(
        grid_size=grid_size, max_steps=64, obs_size=obs_size,
        wall_prob=wall_prob, seed=maze_seed,
    ))
    env.reset()
    grid = env.grid.copy()
    goal = (grid_size - 1, grid_size - 1)

    # Encode goal observation
    env.grid = grid.copy()
    env.agent_pos = list(goal)
    env.steps = 0
    goal_obs = env._get_obs()
    goal_tensor = torch.from_numpy(goal_obs).to(device)
    with torch.no_grad():
        z_goal = encoder.encode(goal_tensor).cpu().numpy()  # (384,)

    # Encode near-goal anchors
    anchors = []
    for r in range(grid_size):
        for c in range(grid_size):
            dist = abs(r - goal[0]) + abs(c - goal[1])
            if dist > max_distance or dist == 0:
                continue
            if grid[r, c] == 1:
                continue

            env.grid = grid.copy()
            env.agent_pos = [r, c]
            env.steps = 0
            obs = env._get_obs()
            obs_t = torch.from_numpy(obs).to(device)
            with torch.no_grad():
                z = encoder.encode(obs_t).cpu().numpy()

            anchors.append({"pos": (r, c), "distance": dist, "latent": z})

    return anchors, z_goal


# ─────────────────────────────────────────────────────────────────────────────
# Extract near-goal real transitions WITH latent-distance reward override
# ─────────────────────────────────────────────────────────────────────────────
def extract_near_goal_real_transitions(
    data_dir, z_goal, tau, grid_size=10, max_distance=4,
    maze_seed=0, wall_prob=0.20, obs_size=64,
):
    if not _REAL_BUFFER:
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

        env.grid = fixed_grid.copy()
        env.agent_pos = [0, 0]
        env.steps = 0
        positions = [(0, 0)]

        for t in range(T - 1):
            action = int(ep.actions[t])
            env.step(action)
            positions.append(tuple(env.agent_pos))
            env.grid = fixed_grid.copy()

        for t in range(T - 1):
            pos = positions[t]
            dist_to_goal = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            if dist_to_goal <= max_distance and dist_to_goal > 0:
                # Override reward with latent-distance reward
                z_next = ep.latents[t + 1]
                l2_dist = float(np.sqrt(np.mean((z_next - z_goal) ** 2)))
                if l2_dist < tau:
                    override_reward = 1.0
                    override_done = 1.0
                else:
                    override_reward = -0.01
                    override_done = 0.0

                near_goal_transitions.append(Transition(
                    z=ep.latents[t],
                    a=int(ep.actions[t]),
                    r=override_reward,
                    z2=z_next,
                    done=override_done,
                ))

    n_goal = sum(1 for t in near_goal_transitions if t.r > 0.5)
    print(f"[INFO] Extracted {len(near_goal_transitions)} near-goal real "
          f"transitions ({n_goal} with goal reward)")
    return near_goal_transitions


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_world_model(checkpoint_path, device):
    if checkpoint_path is None or not _REAL_MODEL:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    config = Config()
    model = DinoWorldModel(config).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"[INFO] Loaded WM from {path} "
          f"(epoch={ckpt.get('epoch', '?')})")
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
    print(f"[INFO] Buffer: {len(buf.episodes)} episodes, {buf.total_steps} steps")
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Step 1: Compute anchors and goal latent ───────────────────────────────
    print("\n[STEP 1] Computing anchors and goal latent...")
    encoder = DinoV2Encoder(device=device)
    anchors, z_goal_np = compute_anchors_and_goal(
        encoder, device, max_distance=args.max_anchor_distance,
        maze_seed=args.maze_seed,
    )
    z_goal_tensor = torch.tensor(z_goal_np, dtype=torch.float32, device=device)
    print(f"[INFO] {len(anchors)} anchors, tau={args.tau}")

    # Compute L2 distances from each anchor to goal for reference
    for a in anchors:
        l2 = float(np.sqrt(np.mean((a["latent"] - z_goal_np) ** 2)))
        print(f"  pos={a['pos']}, manhattan={a['distance']}, "
              f"L2_to_goal={l2:.4f} {'< tau ✓' if l2 < args.tau else '>= tau'}")

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 1b: Extract real transitions with overridden reward ───────────────
    real_transitions = []
    if args.dyna_real_ratio > 0:
        print(f"\n[STEP 1b] Extracting near-goal real transitions (Dyna)...")
        real_transitions = extract_near_goal_real_transitions(
            data_dir=args.data_dir, z_goal=z_goal_np, tau=args.tau,
            max_distance=args.max_anchor_distance, maze_seed=args.maze_seed,
        )
        if len(real_transitions) == 0:
            args.dyna_real_ratio = 0.0

    # ── Load WM + buffer ──────────────────────────────────────────────────────
    print("\n[STEP 2] Loading world model...")
    wm_model = load_world_model(args.wm_checkpoint, device)
    wm_buffer = load_latent_buffer(args.data_dir)

    env = WorldModelEnv(
        model=wm_model, buffer=wm_buffer, device=device,
        max_steps=args.max_episode_steps,
    )

    # ── DQN components ────────────────────────────────────────────────────────
    q = DreamQNet().to(device)
    q_tgt = DreamQNet().to(device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    optimizer = optim.Adam(q.parameters(), lr=args.lr)
    replay_buf = ReplayBuffer(capacity=args.buffer_size)

    # Pre-seed with real transitions
    if args.dyna_real_ratio > 0 and real_transitions:
        n_seed = min(args.learn_starts, len(real_transitions))
        for i in range(n_seed):
            replay_buf.push(real_transitions[i % len(real_transitions)])
        print(f"[INFO] Pre-seeded buffer with {n_seed} real transitions")

    # ── Output setup ──────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.csv"
    metrics_path.write_text(
        "step,episode,episode_return,episode_len,epsilon,"
        "success,goal_rewards_seen\n"
    )

    # ── Training state ────────────────────────────────────────────────────────
    def reset_near_goal():
        anchor = random.choice(anchors)
        z_init = torch.tensor(anchor["latent"], dtype=torch.float32, device=device)
        obs, info = env.reset(z_init=z_init)
        return obs, anchor

    obs, current_anchor = reset_near_goal()
    episode_return = 0.0
    episode_len = 0
    episode_idx = 0
    success_count = 0
    goal_rewards_total = 0
    recent_returns = []
    recent_losses = []
    dream_count = 0
    real_count = 0

    t_start = time.time()
    mode_str = f"Dyna {args.dyna_real_ratio:.0%}" if args.dyna_real_ratio > 0 else "Pure dream"
    print(f"\n{'─'*60}")
    print(f"  Near-Goal Dream DQN — LATENT REWARD (Fix B)")
    print(f"  Mode: {mode_str} | tau={args.tau}")
    print(f"  Steps: {args.steps} | Max ep: {args.max_episode_steps}")
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
        obs_next, _wm_reward, _terminated, truncated, info = env.step(action)

        # ── OVERRIDE REWARD with latent-distance check ────────────────────────
        # This is Fix B: bypass the reward head entirely.
        z_next_tensor = torch.tensor(obs_next, dtype=torch.float32, device=device)
        l2_dist = torch.sqrt(torch.mean((z_next_tensor - z_goal_tensor) ** 2)).item()

        if l2_dist < args.tau:
            reward = 1.0
            terminated = True
            goal_rewards_total += 1
        else:
            reward = -0.01
            terminated = False

        done = terminated or truncated

        # Store transition with overridden reward
        replay_buf.push(Transition(
            z=obs, a=action, r=reward, z2=obs_next, done=float(done),
        ))
        dream_count += 1

        # ── Dyna: inject real transition ──────────────────────────────────────
        if args.dyna_real_ratio > 0 and real_transitions:
            if random.random() < args.dyna_real_ratio / (1 - args.dyna_real_ratio + 1e-8):
                replay_buf.push(random.choice(real_transitions))
                real_count += 1

        episode_return += reward
        episode_len += 1
        obs = obs_next

        # ── Episode end ───────────────────────────────────────────────────────
        if done:
            recent_returns.append(episode_return)
            if len(recent_returns) > 100:
                recent_returns.pop(0)

            ep_success = episode_return > 0.5
            if ep_success:
                success_count += 1

            episode_idx += 1

            avg_loss = np.mean(recent_losses[-100:]) if recent_losses else 0.0
            with open(metrics_path, "a") as f:
                f.write(f"{t},{episode_idx},{episode_return:.4f},"
                        f"{episode_len},{eps:.4f},{int(ep_success)},"
                        f"{goal_rewards_total}\n")

            if episode_idx % 200 == 0:
                mean_ret = np.mean(recent_returns) if recent_returns else 0
                success_pct = success_count / episode_idx * 100
                elapsed = time.time() - t_start
                print(f"  step {t:>7d} | ep {episode_idx:>5d} | "
                      f"ret {mean_ret:>+6.2f} | success {success_pct:>5.1f}% | "
                      f"eps {eps:.3f} | goals_seen {goal_rewards_total} | "
                      f"{elapsed:.0f}s")

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

                # Noise injection (matches train_dream_dqn.py)
                z_noisy  = z  + torch.randn_like(z)  * args.noise_std
                z2_noisy = z2 + torch.randn_like(z2) * args.noise_std

                q_vals = q(z_noisy).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = q_tgt(z2_noisy).max(dim=1).values

                target = r + args.gamma * q_next * (1 - d)

                loss = nn.functional.smooth_l1_loss(q_vals, target)

                optimizer.zero_grad()
                loss.backward()
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
        "tau": args.tau,
        "goal_rewards_total": goal_rewards_total,
        "mode": "latent_reward_dyna" if args.dyna_real_ratio > 0 else "latent_reward_pure",
    }
    ckpt_path = save_dir / "dqn_dream.pt"
    torch.save(ckpt, ckpt_path)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — LATENT REWARD (Fix B)")
    print(f"{'='*60}")
    print(f"  Checkpoint       : {ckpt_path}")
    print(f"  Episodes         : {episode_idx}")
    print(f"  In-dream success : {final_success_pct:.1f}%")
    print(f"  Goal rewards seen: {goal_rewards_total}")
    print(f"  Dream/Real trans : {dream_count}/{real_count}")
    print(f"  Time             : {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Near-goal Dream DQN with latent-distance reward (Fix B)"
    )

    parser.add_argument("--wm_checkpoint", type=str,
                        default="checkpoints/wm_ss_tbptt/world_model_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--save_dir", type=str,
                        default="checkpoints/dqn_near_goal_latent_reward")

    # Near-goal config
    parser.add_argument("--max_anchor_distance", type=int, default=4)
    parser.add_argument("--max_episode_steps", type=int, default=5)
    parser.add_argument("--maze_seed", type=int, default=0)

    # Latent reward config
    parser.add_argument("--tau", type=float, default=0.15,
                        help="L2 distance threshold for goal detection. "
                             "From reward head diagnostic: MSE_to_goal ≈ 0.05-0.07 "
                             "at 1-4 steps, so sqrt(0.07) ≈ 0.26. Use 0.15 for "
                             "a tighter threshold that still catches real goals.")

    # Dyna hybrid
    parser.add_argument("--dyna_real_ratio", type=float, default=0.0)

    # DQN hyperparameters (matched to train_dream_dqn.py)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--learn_starts", type=int, default=1_000)
    parser.add_argument("--train_every", type=int, default=4)
    parser.add_argument("--target_update", type=int, default=1_000)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=int, default=100_000)
    parser.add_argument("--noise_std", type=float, default=0.01)

    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        args.steps = 500
        args.learn_starts = 50
        args.eps_decay = 200

    train(args)


if __name__ == "__main__":
    main()