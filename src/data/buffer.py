"""
src/data/buffer.py
===================
Member C — Task C.2

Latent Replay Buffer: stores encoded latent episodes and samples
random sequences for training the World Model.

Interface Contract (GDP Plan §2.3):
    buffer.sample(batch_size, seq_len) → (batch_size, seq_len, 384) float32 Tensor

Usage:
    buf = LatentReplayBuffer(capacity_steps=200_000)
    buf.add_episode(latents)          # latents: np.ndarray (T, 384)
    batch = buf.sample(32, seq_len=16) # → torch.Tensor (32, 16, 384)
"""

import numpy as np
import torch


class LatentReplayBuffer:
    def __init__(self, capacity_steps: int):
        """
        :param capacity_steps: Maximum total number of latent steps to store.
                               Oldest episodes are evicted once capacity is exceeded.
        """
        self.episodes       = []   # list of np.ndarray, each shape (T, 384)
        self.capacity_steps = capacity_steps
        self.total_steps    = 0

    def add_episode(self, latents: np.ndarray):
        """
        Add a full episode of latent vectors to the buffer.

        :param latents: np.ndarray of shape (episode_length, 384)
        """
        self.episodes.append(latents.astype(np.float32))
        self.total_steps += latents.shape[0]

        # Evict oldest episodes to stay within capacity
        while self.total_steps > self.capacity_steps:
            self.total_steps -= self.episodes.pop(0).shape[0]

    def sample(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Sample a batch of random latent sequences.

        :param batch_size: Number of sequences in the batch.
        :param seq_len:    Length of each sequence (must be <= episode length).
        :returns:          torch.Tensor of shape (batch_size, seq_len, 384), dtype float32.
        :raises ValueError: If no episodes are long enough for the requested seq_len.
        """
        candidates = [ep for ep in self.episodes if ep.shape[0] >= seq_len]

        if not candidates:
            raise ValueError(
                f"No episodes in buffer long enough for seq_len={seq_len}. "
                f"Buffer has {len(self.episodes)} episodes, "
                f"total_steps={self.total_steps}."
            )

        out = np.zeros((batch_size, seq_len, 384), dtype=np.float32)

        for b in range(batch_size):
            ep    = candidates[np.random.randint(len(candidates))]
            start = np.random.randint(0, ep.shape[0] - seq_len + 1)
            out[b] = ep[start : start + seq_len]

        return torch.from_numpy(out)

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