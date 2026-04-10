"""
src/data/buffer.py
===================
Member C — Task C.2, Task B.1 in CDR

Latent Replay Buffer: stores encoded latent episodes and samples
random sequences for training the World Model.

Interface Contract (GDP Plan §2.3):
    buffer.sample(batch_size, seq_len) → Batch namedtuple with fields:
        .latents  (batch_size, seq_len, 384) float32 Tensor
        .actions  (batch_size, seq_len)      int64 Tensor
        .rewards  (batch_size, seq_len)      float32 Tensor
        .dones    (batch_size, seq_len)      float32 Tensor

Usage:
    buf = LatentReplayBuffer(capacity_steps=200_000)
    buf.add_episode(latents, actions, rewards, dones)
    batch = buf.sample(32, seq_len=16)
    # → batch.latents (32, 16, 384), batch.actions (32, 16), etc.

─────────────────────────────────────────────────────────────
MODIFICATION — Member B (Stratified Sampling, Final Sprint)
─────────────────────────────────────────────────────────────
Added stratified sampling to sample() to guarantee a fraction
of each batch contains sequences with goal transitions.

Goal transitions are only 1.18% of the buffer (1482 out of
125724 transitions). Pure random sampling means most batches
contain zero goal transitions, so the reward head never learns
to predict the +1.0 goal reward regardless of loss weighting.

Stratified sampling fixes this by:
  1. Splitting episodes into goal and non-goal pools
  2. Guaranteeing goal_fraction of each batch comes from
     goal episodes, with windows that contain the goal step
  3. Filling remaining slots with normal random sampling

New parameter:
  goal_fraction : float — fraction of batch guaranteed to
                  contain a goal transition. Default: 0.25.
                  0.0 reproduces original random sampling.
─────────────────────────────────────────────────────────────
"""

import numpy as np
import torch
from collections import namedtuple

Episode = namedtuple("Episode", ["latents", "actions", "rewards", "dones"])
Batch   = namedtuple("Batch",   ["latents", "actions", "rewards", "dones"])


class LatentReplayBuffer:
    def __init__(self, capacity_steps: int):
        """
        :param capacity_steps: Maximum total number of latent steps to store.
                               Oldest episodes are evicted once capacity is exceeded.
        """
        self.episodes       = []
        self.capacity_steps = capacity_steps
        self.total_steps    = 0

    def add_episode(
        self,
        latents: np.ndarray,
        actions: np.ndarray = None,
        rewards: np.ndarray = None,
        dones:   np.ndarray = None,
    ):
        """
        Add a full episode of latent vectors to the buffer.

        :param latents: np.ndarray of shape (episode_length, 384)
        :param actions: np.ndarray of shape (episode_length, *action_shape)
        :param rewards: np.ndarray of shape (episode_length,)
        :param dones:   np.ndarray of shape (episode_length,) bool or float32
        :raises ValueError: If arrays have inconsistent leading dimensions.
        """
        T = latents.shape[0]
        if actions is not None and rewards is not None and dones is not None:
            if not (actions.shape[0] == rewards.shape[0] == dones.shape[0] == T):
                raise ValueError(
                    f"All arrays must have the same leading dimension. "
                    f"Got latents={T}, actions={actions.shape[0]}, "
                    f"rewards={rewards.shape[0]}, dones={dones.shape[0]}."
                )

        self.episodes.append(Episode(
            latents = latents.astype(np.float32),
            actions = actions.astype(np.int64)   if actions is not None else np.zeros(T, dtype=np.int64),
            rewards = rewards.astype(np.float32) if rewards is not None else np.zeros(T, dtype=np.float32),
            dones   = dones.astype(np.float32)   if dones   is not None else np.zeros(T, dtype=np.float32),
        ))

        self.total_steps += T

        # Evict oldest episodes to stay within capacity
        while self.total_steps > self.capacity_steps:
            popped = self.episodes.pop(0)
            self.total_steps -= popped.latents.shape[0]

    def sample(
        self,
        batch_size:    int,
        seq_len:       int,
        goal_fraction: float = 0.25,
    ) -> "Batch":
        """
        Sample a batch of random latent sequences with stratified goal sampling.

        A fraction (goal_fraction) of the batch is guaranteed to contain
        sequences that include a goal transition (+1.0 reward). This prevents
        the reward head from ignoring the sparse goal signal during training.

        :param batch_size:    Number of sequences in the batch.
        :param seq_len:       Length of each sequence (must be <= episode length).
        :param goal_fraction: Fraction of batch guaranteed to contain a goal
                              transition. 0.0 reproduces original random sampling.
        :returns:             Batch namedtuple of float32 Tensors:
                                  .latents (batch_size, seq_len, 384)
                                  .actions (batch_size, seq_len, *action_shape)
                                  .rewards (batch_size, seq_len)
                                  .dones   (batch_size, seq_len)
        :raises ValueError:   If no episodes are long enough for seq_len.
        """
        candidates = [ep for ep in self.episodes if ep.latents.shape[0] >= seq_len]

        if not candidates:
            raise ValueError(
                f"No episodes in buffer long enough for seq_len={seq_len}. "
                f"Buffer has {len(self.episodes)} episodes, "
                f"total_steps={self.total_steps}."
            )

        # ── Split into goal and non-goal episode pools ────────────────────────
        goal_eps = [ep for ep in candidates if ep.rewards.max() > 0.5]

        # How many goal slots to guarantee this batch
        # If no goal episodes available, fall back to pure random sampling
        n_goal    = min(int(batch_size * goal_fraction), len(goal_eps))
        n_nongoal = batch_size - n_goal

        # ── Allocate output arrays ────────────────────────────────────────────
        action_shape = candidates[0].actions.shape[1:]
        latents_out  = np.zeros((batch_size, seq_len, 384),           dtype=np.float32)
        actions_out  = np.zeros((batch_size, seq_len, *action_shape), dtype=np.float32)
        rewards_out  = np.zeros((batch_size, seq_len),                dtype=np.float32)
        dones_out    = np.zeros((batch_size, seq_len),                dtype=np.float32)

        # ── Helper: fill one batch slot ───────────────────────────────────────
        def fill_slot(b: int, ep: Episode, force_goal: bool = False) -> None:
            """
            Fill batch slot b from episode ep.

            If force_goal=True, sample a window guaranteed to contain the
            goal transition. The window is chosen so the goal step falls
            somewhere within [start, start+seq_len), ensuring the reward
            head sees the +1.0 signal in this sequence.
            """
            if force_goal:
                # Find the goal step index
                goal_idx  = int(np.argmax(ep.rewards > 0.5))
                # Window must start early enough to include goal_idx
                # and late enough that the window fits in the episode
                start_min = max(0, goal_idx - seq_len + 1)
                start_max = min(goal_idx, ep.latents.shape[0] - seq_len)
                start     = np.random.randint(start_min, start_max + 1)
            else:
                start = np.random.randint(0, ep.latents.shape[0] - seq_len + 1)

            sl             = slice(start, start + seq_len)
            latents_out[b] = ep.latents[sl]
            actions_out[b] = ep.actions[sl]
            rewards_out[b] = ep.rewards[sl]
            dones_out[b]   = ep.dones[sl]

        # ── Fill goal slots ───────────────────────────────────────────────────
        for b in range(n_goal):
            ep = goal_eps[np.random.randint(len(goal_eps))]
            fill_slot(b, ep, force_goal=True)

        # ── Fill remaining slots with normal random sampling ──────────────────
        for b in range(n_goal, batch_size):
            ep = candidates[np.random.randint(len(candidates))]
            fill_slot(b, ep, force_goal=False)

        return Batch(
            latents = torch.from_numpy(latents_out),  # (B, seq_len, 384)
            actions = torch.from_numpy(actions_out),  # (B, seq_len, *action_shape)
            rewards = torch.from_numpy(rewards_out),  # (B, seq_len)
            dones   = torch.from_numpy(dones_out),    # (B, seq_len)
        )

    def __len__(self) -> int:
        """Return the number of episodes currently stored."""
        return len(self.episodes)

    def __repr__(self) -> str:
        return (
            f"LatentReplayBuffer("
            f"episodes={len(self.episodes)}, "
            f"total_steps={self.total_steps}, "
            f"capacity={self.capacity_steps})"
        )
