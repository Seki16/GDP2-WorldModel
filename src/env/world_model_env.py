"""
src/env/world_model_env.py
===========================
A gymnasium.Env wrapper around the trained Latent World Model.

Phase 3 fixes applied (CDR sprint):
  3a — Latent normalisation alignment: affine correction to WM outputs
  3b — Latent re-anchoring: reset z_t to a real buffer latent every N steps
  3c — Shaped reward: supplement WM reward with -||z_t - z_goal||₂ distance

Provides a standard RL environment interface where:
  - State space  : latent vectors z ∈ R^384  (DINOv2 ViT-S/14)
  - Action space : Discrete(4)               (integer class ids [0..3])
  - Dynamics     : DinoWorldModel forward pass (or random stub)
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
MAX_STEPS  = 64


# ── Random model stub ─────────────────────────────────────────────────────────

class _RandomWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._z   = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
        self._rew = nn.Linear(LATENT_DIM, 1,          bias=False)
        self._don = nn.Linear(LATENT_DIM, 1,          bias=False)

    def forward(self, z_in, a_in):
        return self._z(z_in), self._rew(z_in), self._don(z_in)


# ── Environment ───────────────────────────────────────────────────────────────

class WorldModelEnv(gym.Env):
    """
    Gymnasium environment backed by a Latent World Model.

    Parameters
    ----------
    model           : DinoWorldModel (or None → random stub)
    buffer          : LatentReplayBuffer (or None → zero latent fallback)
    device          : torch.device or str
    max_steps       : int, default 64
    re_anchor_every : int or None — re-anchor z_t to a real buffer latent
                      every N steps. None disables re-anchoring.
    gap_stats_path  : str or None — path to latent_gap_stats.npz for
                      normalisation alignment (step 3a). None disables.
    shaped_reward_w : float — weight for shaped reward term (step 3c).
                      0.0 disables shaped reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model=None,
        buffer=None,
        device: torch.device | str = "cpu",
        max_steps: int = MAX_STEPS,
        re_anchor_every: int | None = None,
        gap_stats_path: str | None = None,
        shaped_reward_w: float = 0.0,
    ):
        super().__init__()

        self.device          = torch.device(device)
        self.max_steps       = max_steps
        self.buffer          = buffer
        self.re_anchor_every = re_anchor_every
        self.shaped_reward_w = shaped_reward_w

        # Model
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

        # Internal state
        self._z:          torch.Tensor | None = None
        self._step_count: int                 = 0

        # FIX 3a: DISABLED by default (gap_stats_path=None).
        # Affine correction caused catastrophic explosion — max variance ratio 11.8
        # amplified near-zero σ_wm dimensions to infinity (avg_ret → -75 trillion).
        # To enable: pass gap_stats_path="evaluation/latent_gap_stats.npz" to __init__
        # WARNING: requires evaluation/latent_gap_stats.npz (gitignored, not in repo).
        # Generate it first by running: python -m src.scripts.latent_distribution_gap
        # ── 3a: Normalisation alignment parameters ────────────────────────────
        self._norm_mu_real:  torch.Tensor | None = None
        self._norm_mu_wm:    torch.Tensor | None = None
        self._norm_std_real: torch.Tensor | None = None
        self._norm_std_wm:   torch.Tensor | None = None
        self._norm_enabled = False

        if gap_stats_path is not None:
            try:
                stats = np.load(gap_stats_path)
                self._norm_mu_real  = torch.tensor(
                    stats["mu_real"],  dtype=torch.float32, device=self.device)
                self._norm_mu_wm    = torch.tensor(
                    stats["mu_wm"],   dtype=torch.float32, device=self.device)
                self._norm_std_real = torch.tensor(
                    np.sqrt(stats["var_real"].clip(1e-8)),
                    dtype=torch.float32, device=self.device)
                self._norm_std_wm   = torch.tensor(
                    np.sqrt(stats["var_wm"].clip(1e-8)),
                    dtype=torch.float32, device=self.device)
                self._norm_enabled  = True
                print(f"[WorldModelEnv] 3a: Normalisation alignment loaded "
                      f"from {gap_stats_path}")
            except FileNotFoundError:
                print(f"[WorldModelEnv] 3a: {gap_stats_path} not found — "
                      f"normalisation alignment disabled")

        # FIX 3b: DISABLED by default (re_anchor_every=None).
        # Resetting z_t to real buffer latents every N steps caused done_head to fire
        # at anchor points → fake 96-100% success. At every step, breaks sequential
        # structure entirely — agent cannot learn navigation when teleported each step.
        # To enable: pass re_anchor_every=4 or 8 to __init__ (experiment carefully).
        # ── 3b: Cache seed=0 initial latent ──────────────────────────────────
        self._z_init: torch.Tensor = self._fetch_z_init()


        # FIX 3c: DISABLED by default (shaped_reward_w=0.0).
        # L2 distances in 384-dim space (~2.5 typical) buried the +1 goal reward
        # at any non-trivial weight. weight=0.1 gave avg_ret=-60; weight=0.001
        # inflated cumulative return above 0.9, making success metric meaningless.
        # To enable: pass shaped_reward_w=0.001 to __init__ and validate carefully.
        # ── 3c: Goal latent (mean of episodes that reached the goal) ──────────
        self._z_goal: torch.Tensor | None = self._compute_goal_latent()
        if self._z_goal is not None:
            print(f"[WorldModelEnv] 3c: Goal latent computed from buffer")
        else:
            print(f"[WorldModelEnv] 3c: No goal transitions found — "
                  f"shaped reward disabled")
            self.shaped_reward_w = 0.0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_z_init(self) -> torch.Tensor:
        """Return first latent of seed=0 episode (agent at [0,0])."""
        if self.buffer is None or len(self.buffer.episodes) == 0:
            return torch.zeros(1, 1, LATENT_DIM, dtype=torch.float32,
                               device=self.device)
        z0 = torch.tensor(
            self.buffer.episodes[0].latents[0],
            dtype=torch.float32, device=self.device,
        )
        return z0.unsqueeze(0).unsqueeze(0)  # (1, 1, 384)

    def _compute_goal_latent(self) -> torch.Tensor | None:
        """
        3c: Compute mean latent of goal states from the real buffer.
        A goal state is the last latent of any episode where reward > 0.5.
        """
        if self.buffer is None or len(self.buffer.episodes) == 0:
            return None

        goal_latents = []
        for ep in self.buffer.episodes:
            if ep.rewards.max() > 0.5:
                # Last latent before done=True is the goal state
                goal_idx = int(np.argmax(ep.rewards > 0.5))
                goal_latents.append(ep.latents[goal_idx])

        if not goal_latents:
            return None

        z_goal = np.stack(goal_latents, axis=0).mean(axis=0)  # (384,)
        return torch.tensor(z_goal, dtype=torch.float32,
                            device=self.device)

    def _sample_real_latent(self) -> torch.Tensor:
        """3b: Sample a random real latent from the buffer."""
        ep  = self.buffer.episodes[
            np.random.randint(len(self.buffer.episodes))]
        idx = np.random.randint(len(ep.latents))
        z   = torch.tensor(ep.latents[idx], dtype=torch.float32,
                           device=self.device)
        return z.unsqueeze(0).unsqueeze(0)  # (1, 1, 384)

    def _apply_norm_alignment(self, z: torch.Tensor) -> torch.Tensor:
        """
        3a: Apply affine correction to align WM latent distribution
        with real DINOv2 latent distribution.
        z_corrected = (z - μ_wm) / σ_wm * σ_real + μ_real
        """
        if not self._norm_enabled:
            return z
        z_flat = z.squeeze()  # (384,)
        z_corr = ((z_flat - self._norm_mu_wm) / self._norm_std_wm
                  * self._norm_std_real + self._norm_mu_real)
        return z_corr.unsqueeze(0).unsqueeze(0)

    def _z_to_obs(self, z: torch.Tensor) -> np.ndarray:
        return z.squeeze().cpu().numpy().astype(np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        z_init: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if z_init is not None:
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
            "step":       self._step_count,
            "done_logit": None,
            "source":     "z_init_override" if z_init is not None
                          else "buffer_seed0",
        }
        return self._z_to_obs(self._z), info

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:

        if self._z is None:
            raise gym.error.ResetNeeded("Call reset() before step().")

        if not self.action_space.contains(np.int64(action)):
            raise ValueError(
                f"Invalid action {action!r}. "
                f"Expected an integer in [0, {ACTION_DIM - 1}]."
            )

        self._step_count += 1

        # ── Forward pass ──────────────────────────────────────────────────────
        a_tensor = torch.tensor([[action]], dtype=torch.long,
                                device=self.device)
        with torch.no_grad():
            pred_next, pred_rew, pred_done = self.model(self._z, a_tensor)

        # ── 3a: Apply normalisation alignment to predicted next latent ────────
        z_next = self._apply_norm_alignment(pred_next[:, -1:, :])

        # ── 3b: Periodic re-anchoring ─────────────────────────────────────────
        if (self.re_anchor_every is not None
                and self.buffer is not None
                and len(self.buffer.episodes) > 0
                and self._step_count % self.re_anchor_every == 0):
            z_next = self._sample_real_latent()

        self._z = z_next

        # ── Reward ────────────────────────────────────────────────────────────
        wm_reward = float(pred_rew[:, -1, 0].item())

        # ── 3c: Shaped reward ─────────────────────────────────────────────────
        shaped = 0.0
        if self.shaped_reward_w > 0.0 and self._z_goal is not None:
            dist    = torch.norm(self._z.squeeze() - self._z_goal).item()
            shaped  = -dist * self.shaped_reward_w

        reward = wm_reward + shaped

        # ── Termination ───────────────────────────────────────────────────────
        done_logit  = float(pred_done[:, -1, 0].item())
        terminated  = False #done_logit > 0.0
        truncated   = (not terminated) and (self._step_count >= self.max_steps)

        info = {
            "step":       self._step_count,
            "done_logit": done_logit,
            "wm_reward":  wm_reward,
            "shaped":     shaped,
        }
        return self._z_to_obs(self._z), reward, terminated, truncated, info

    def __repr__(self) -> str:
        return (
            f"WorldModelEnv("
            f"max_steps={self.max_steps}, "
            f"re_anchor_every={self.re_anchor_every}, "
            f"norm_alignment={self._norm_enabled}, "
            f"shaped_reward_w={self.shaped_reward_w})"
        )