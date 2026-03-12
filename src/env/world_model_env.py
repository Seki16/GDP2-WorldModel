"""
src/env/world_model_env.py
===========================
A gymnasium.Env wrapper around the trained Latent World Model.

Provides a standard RL environment interface where:
  - State space  : latent vectors z ∈ R^384  (DINOv2 ViT-S/14)
  - Action space : Discrete(4)               (integer class ids [0..3])
  - Dynamics     : DinoWorldModel forward pass (or random stub)

Usage
-----
    # With trained model
    from src.env.world_model_env import WorldModelEnv
    env = WorldModelEnv(model=model, buffer=buffer, device=device)

    # With random stub (no weights needed — for development / smoke-tests)
    env = WorldModelEnv(model=None, buffer=None, device="cpu")

    z, info = env.reset()
    z_next, reward, done, truncated, info = env.step(action=0)

Interface Contract (GDP Plan §2.3)
------------------------------------
  Latent dim   : 384
  Action dim   : 4
  Max steps    : 64  (hardcoded fallback)
  Initial state: first latent of seed=0 episode from buffer
                 → agent at grid position [0, 0]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces


# ── Constants ─────────────────────────────────────────────────────────────────
LATENT_DIM = 384
ACTION_DIM = 4
MAX_STEPS  = 64      # hard step-limit fallback for done


# ── Random model stub ─────────────────────────────────────────────────────────

class _RandomWorldModel(nn.Module):
    """
    Drop-in stub for DinoWorldModel.
    Produces plausible-shaped outputs without any trained weights,
    allowing WorldModelEnv to be developed and tested independently.
    """
    def __init__(self):
        super().__init__()
        # Tiny learned layers so .parameters() is non-empty (optimiser won't crash)
        self._z   = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
        self._rew = nn.Linear(LATENT_DIM, 1,          bias=False)
        self._don = nn.Linear(LATENT_DIM, 1,          bias=False)

    def forward(
        self,
        z_in: torch.Tensor,   # (B, T, 384)
        a_in: torch.Tensor,   # (B, T)
    ):
        pred_next = self._z(z_in)                    # (B, T, 384)
        pred_rew  = self._rew(z_in)                  # (B, T, 1)
        pred_done = self._don(z_in)                  # (B, T, 1) — logits
        return pred_next, pred_rew, pred_done


# ── Environment ───────────────────────────────────────────────────────────────

class WorldModelEnv(gym.Env):
    """
    Gymnasium environment backed by a Latent World Model.

    Parameters
    ----------
    model   : DinoWorldModel (or None → uses _RandomWorldModel stub)
    buffer  : LatentReplayBuffer (or None → reset() returns a zero latent)
    device  : torch.device or str
    max_steps : int, default MAX_STEPS=64
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model=None,
        buffer=None,
        device: torch.device | str = "cpu",
        max_steps: int = MAX_STEPS,
    ):
        super().__init__()

        self.device    = torch.device(device)
        self.max_steps = max_steps
        self.buffer    = buffer

        # Model — fall back to random stub if none supplied
        if model is None:
            self.model = _RandomWorldModel().to(self.device)
        else:
            self.model = model.to(self.device)
        self.model.eval()

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(LATENT_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        # Internal state (set properly in reset())
        self._z:        torch.Tensor | None = None   # (1, 1, 384)
        self._step_count: int               = 0

        # Cache seed=0 initial latent from buffer at construction time
        self._z_init: torch.Tensor = self._fetch_z_init()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_z_init(self) -> torch.Tensor:
        """
        Return the first latent of the seed=0 episode from the buffer.
        This corresponds to the agent at grid position [0, 0] — the fixed,
        reproducible starting point of every episode.

        Falls back to a zero vector if no buffer is available.
        """
        if self.buffer is None or len(self.buffer.episodes) == 0:
            return torch.zeros(1, 1, LATENT_DIM, dtype=torch.float32,
                               device=self.device)

        # seed=0 episode is defined as the first episode added to the buffer
        first_episode = self.buffer.episodes[0]           # np.ndarray (T, 384)
        z0 = torch.tensor(
            first_episode[0],                             # shape (384,)
            dtype=torch.float32,
            device=self.device,
        )
        return z0.unsqueeze(0).unsqueeze(0)               # (1, 1, 384)

    def _z_to_obs(self, z: torch.Tensor) -> np.ndarray:
        """Convert internal (1, 1, 384) latent to (384,) numpy observation."""
        return z.squeeze().cpu().numpy().astype(np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        z_init: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to the fixed starting latent.

        Parameters
        ----------
        z_init : optional override (1, 1, 384) or (384,) tensor.
                 If omitted, uses the first latent of the seed=0 buffer episode
                 → agent at position [0, 0].

        Returns
        -------
        observation : np.ndarray (384,)
        info        : dict
        """
        super().reset(seed=seed)

        if z_init is not None:
            # Accept (384,), (1, 384), or (1, 1, 384)
            z = torch.as_tensor(z_init, dtype=torch.float32, device=self.device)
            if z.ndim == 1:
                z = z.unsqueeze(0).unsqueeze(0)
            elif z.ndim == 2:
                z = z.unsqueeze(0)
            self._z = z
        else:
            self._z = self._z_init.clone()

        self._step_count = 0

        info = {
            "step":          self._step_count,
            "done_logit":    None,
            "source":        "z_init_override" if z_init is not None
                             else "buffer_seed0",
        }
        return self._z_to_obs(self._z), info

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance the world model by one step.

        Parameters
        ----------
        action : int in [0, ACTION_DIM)

        Returns  (Gymnasium 5-tuple)
        -------
        observation : np.ndarray (384,)    — z_next
        reward      : float
        terminated  : bool                 — model predicted done > 0.5
        truncated   : bool                 — step limit reached
        info        : dict  {step, done_logit}
        """
        if self._z is None:
            raise gym.error.ResetNeeded("Call reset() before step().")

        # ── Build single-step tensors ─────────────────────────────────────────
        a_in = torch.tensor(
            [[int(action)]], dtype=torch.long, device=self.device,
        )   # (1, 1)

        # ── World model forward pass ──────────────────────────────────────────
        with torch.no_grad():
            pred_next, pred_rew, pred_done = self.model(self._z, a_in)
            # pred_next  (1, 1, 384)
            # pred_rew   (1, 1, 1)
            # pred_done  (1, 1, 1)  — logit

        # ── Extract scalars ───────────────────────────────────────────────────
        z_next     = pred_next                          # (1, 1, 384)
        reward     = pred_rew.squeeze().item()          # float
        done_logit = pred_done.squeeze().item()         # float (raw logit)
        done_prob  = torch.sigmoid(pred_done).squeeze().item()

        # ── Termination conditions ────────────────────────────────────────────
        self._step_count += 1
        terminated = done_prob > 0.5                    # model says episode over
        truncated  = self._step_count >= self.max_steps # hard step-limit fallback

        # ── Advance internal state ────────────────────────────────────────────
        self._z = z_next

        info = {
            "step":       self._step_count,
            "done_logit": done_logit,       # raw logit — useful for debugging
            "done_prob":  done_prob,        # sigmoid of logit
        }

        return self._z_to_obs(z_next), reward, terminated, truncated, info

    def render(self):
        """Not implemented — latent space has no pixel representation."""
        pass

    def close(self):
        pass

    def __repr__(self) -> str:
        model_name = type(self.model).__name__
        return (
            f"WorldModelEnv("
            f"model={model_name}, "
            f"max_steps={self.max_steps}, "
            f"step={self._step_count})"
        )