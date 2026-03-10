"""
src/data/buffer.py
===================
Member C — Task C.2, Task B.1 in CDR

Latent Replay Buffer: stores encoded latent episodes and samples
random sequences for training the World Model.

Interface Contract (GDP Plan §2.3):
    buffer.sample(batch_size, seq_len) → (batch_size, seq_len, 384) float32 Tensor

Usage:
    buf = LatentReplayBuffer(capacity_steps=200_000)
    buf.add_episode(latents, actions, rewards, dones)
    batch = buf.sample(32, seq_len=16)
    # -> batch.latents (32, 16, 384)
    # -> batch.actions (32, 16, *action_shape)
    # -> batch.rewards (32, 16)
    # -> batch.dones   (32, 16)
"""

import numpy as np
import torch
from collections import namedtuple

Episode = namedtuple("Episode", ["latents", "actions", "rewards", "dones"])
Batch = namedtuple("Batch", ["latents", "actions", "rewards", "dones"])

class LatentReplayBuffer:
    def __init__(self, capacity_steps: int):
        """
        :param capacity_steps: Maximum total number of latent steps to store.
                               Oldest episodes are evicted once capacity is exceeded.
        """
        self.episodes       = []   # list of np.ndarray, each shape (T, 384)
        self.capacity_steps = capacity_steps
        self.total_steps    = 0

    def add_episode(self, 
                    latents: np.ndarray,
                    actions: np.ndarray,
                    rewards: np.ndarray,
                    dones: np.ndarray):
        """
        Add a full episode of latent vectors to the buffer.

        :param latents: np.ndarray of shape (episode_length, 384)
        :param actions: np.ndarray of shape (episode_length, *action_shape)
        :param rewards: np.ndarray of shape (episode_length,)
        :param dones: np.ndarray of shape (episode_length,) bool or float32
        :raises ValueError: If arrays have inconsistent leading dimensions.
        """
        T = latents.shape[0]
        if not (actions.shape[0] == rewards.shape[0] == dones.shape[0] == T):
            raise ValueError(
                f"All arrays must have the same leading dimension. "
                f"Got latents={T}, actions={actions.shape[0]}, "
                f"rewards={rewards.shape[0]}, dones={dones.shape[0]}."
            )
        
        self.episodes.append(Episode(
            latents = latents.astype(np.float32),
            actions = actions.astype(np.float32),
            rewards = rewards.astype(np.float32).reshape(T),
            dones = dones.astype(np.float32).reshape(T)
        ))
        
        self.total_steps += T

        # Evict oldest episodes to stay within capacity
        while self.total_steps > self.capacity_steps:
            self.total_steps -= self.episodes.pop(0).shape[0]

    def sample(self, batch_size: int, seq_len: int) -> Batch:
        """
        Sample a batch of random latent sequences.

        :param batch_size: Number of sequences in the batch.
        :param seq_len:    Length of each sequence (must be <= episode length).
        :returns:          Batch namedtuple of float32 Tensors:
                               .latents (batch_size, seq_len, 384)
                               .actions (batch_size, seq_len, *action_shape)
                               .rewards (batch_size, seq_len)
                               .dones   (batch_size, seq_len)
        :raises ValueError: If no episodes are long enough for the requested seq_len.
        """
        candidates = [ep for ep in self.episodes if ep.latents.shape[0] >= seq_len]

        if not candidates:
            raise ValueError(
                f"No episodes in buffer long enough for seq_len={seq_len}. "
                f"Buffer has {len(self.episodes)} episodes, "
                f"total_steps={self.total_steps}."
            )

        #out = np.zeros((batch_size, seq_len, 384), dtype=np.float32)
        action_shape = candidates[0].actions.shape[1:]  # infer action shape from first candidate
        
        latents_out = np.zeros((batch_size, seq_len, 384), dtype=np.float32)
        actions_out = np.zeros((batch_size, seq_len, *action_shape), dtype=np.float32)
        rewards_out = np.zeros((batch_size, seq_len), dtype=np.float32)
        dones_out   = np.zeros((batch_size, seq_len), dtype=np.float32)

        for b in range(batch_size):
            ep    = candidates[np.random.randint(len(candidates))]
            start = np.random.randint(0, ep.latents.shape[0] - seq_len + 1)
            #out[b] = ep[start : start + seq_len]
            sl = slice(start, start+seq_len)
            
            latents_out[b] = ep.latents[sl]
            actions_out[b] = ep.actions[sl]
            rewards_out[b] = ep.rewards[sl]
            dones_out[b]   = ep.dones[sl]

        return Batch(
            latents = torch.from_numpy(latents_out),  # (B, seq_len, 384)
            actions = torch.from_numpy(actions_out),  # (B, seq_len, *action_shape)
            rewards = torch.from_numpy(rewards_out),  # (B, seq_len)
            dones   = torch.from_numpy(dones_out)     # (B, seq_len)
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