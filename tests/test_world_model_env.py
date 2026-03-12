"""
tests/test_world_model_env.py
==============================
Smoke-tests for WorldModelEnv (GDP Plan integration check).

Verifies:
  1. 1000 random steps run without crash
  2. done triggers at least once across 100 episodes
  3. Reward is non-constant (has variance)
  4. truncated fires after max_steps with never-done model
  5. terminated and truncated are mutually exclusive
  6. reset() with z_init override produces correct first obs
  7. Invalid actions raise ValueError

Run:
    python -m pytest tests/test_world_model_env.py -v
    # or directly:
    python tests/test_world_model_env.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.env.world_model_env import WorldModelEnv, LATENT_DIM, ACTION_DIM

PASS = "✅"
FAIL = "❌"


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {name}{suffix}")
    assert condition, f"SMOKE TEST VIOLATION: {name}{suffix}"


# ── Shared stubs ──────────────────────────────────────────────────────────────

class _ForcedDoneModel(nn.Module):
    """Always outputs done_prob > 0.5 (logit=+5 → prob≈0.993)."""
    def forward(self, z_in, a_in):
        return (
            z_in,
            torch.zeros(*z_in.shape[:-1], 1),
            torch.ones(*z_in.shape[:-1], 1) * 5.0,
        )


class _NeverDoneModel(nn.Module):
    """Always outputs done_prob < 0.5 (logit=−5 → prob≈0.007)."""
    def forward(self, z_in, a_in):
        return (
            z_in,
            torch.zeros(*z_in.shape[:-1], 1),
            torch.ones(*z_in.shape[:-1], 1) * -5.0,
        )


class _VaryingRewardModel(nn.Module):
    """Returns fresh randn reward each step, never terminates."""
    def forward(self, z_in, a_in):
        return (
            z_in,
            torch.randn(*z_in.shape[:-1], 1),
            torch.ones(*z_in.shape[:-1], 1) * -5.0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# TEST 1 — 1000 random steps without crash
# ──────────────────────────────────────────────────────────────────────────────

def test_1000_steps_no_crash():
    print("\n[TEST 1] 1000 random steps without crash")

    env   = WorldModelEnv(model=None, buffer=None, device="cpu")
    steps = 0
    obs, info = env.reset()

    while steps < 1000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (384,),         f"Bad obs shape at step {steps}: {obs.shape}"
        assert isinstance(reward, float),   f"Reward not float at step {steps}"
        assert isinstance(terminated, bool),f"terminated not bool at step {steps}"
        assert isinstance(truncated, bool), f"truncated not bool at step {steps}"
        assert "step"       in info,        "info missing 'step'"
        assert "done_logit" in info,        "info missing 'done_logit'"
        assert "done_prob"  in info,        "info missing 'done_prob'"

        steps += 1
        if terminated or truncated:
            obs, info = env.reset()

    check("1000 steps completed without crash", True, f"total_steps={steps}")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 2 — done triggers at least once in 100 episodes
# ──────────────────────────────────────────────────────────────────────────────

def test_done_triggers():
    print("\n[TEST 2] done triggers at least once in 100 episodes")

    env = WorldModelEnv(model=_ForcedDoneModel(), buffer=None, device="cpu")

    terminated_count = 0
    truncated_count  = 0
    EPISODES         = 100

    for ep in range(EPISODES):
        obs, info = env.reset()
        for _ in range(env.max_steps + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                terminated_count += 1
                break
            if truncated:
                truncated_count += 1
                break

    print(f"     terminated={terminated_count}  truncated={truncated_count}"
          f"  total_episodes={EPISODES}")

    check(
        "done (terminated) triggered at least once in 100 episodes",
        terminated_count >= 1,
        f"terminated={terminated_count}/100",
    )
    check(
        "every episode ends (terminated or truncated)",
        terminated_count + truncated_count == EPISODES,
        f"unended={(EPISODES - terminated_count - truncated_count)}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# TEST 3 — reward is non-constant
# ──────────────────────────────────────────────────────────────────────────────

def test_reward_non_constant():
    print("\n[TEST 3] Reward is non-constant across steps")

    env     = WorldModelEnv(model=_VaryingRewardModel(), buffer=None, device="cpu")
    rewards = []

    obs, info = env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            obs, info = env.reset()

    rewards  = np.array(rewards)
    variance = float(np.var(rewards))
    r_min    = float(rewards.min())
    r_max    = float(rewards.max())

    print(f"     reward  min={r_min:.4f}  max={r_max:.4f}  var={variance:.6f}")

    check(
        "reward has non-zero variance across 200 steps",
        variance > 1e-8,
        f"var={variance:.6f}",
    )
    check(
        "reward range is non-trivial (max - min > 1e-4)",
        (r_max - r_min) > 1e-4,
        f"range={r_max - r_min:.6f}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# TEST 4 — truncated fires after max_steps (never-done model)
# ──────────────────────────────────────────────────────────────────────────────

def test_truncated_fires_at_max_steps():
    print("\n[TEST 4] truncated fires after max_steps")

    max_steps = 5
    env = WorldModelEnv(model=_NeverDoneModel(), buffer=None,
                        device="cpu", max_steps=max_steps)
    env.reset()

    terminated_ever = False
    truncated_ever  = False

    for _ in range(max_steps):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated:
            terminated_ever = True
        if truncated:
            truncated_ever = True

    check("truncated fires at max_steps",      truncated_ever,
          f"max_steps={max_steps}")
    check("terminated never fires (logit=−5)", not terminated_ever)
    check("step_count == max_steps",           env._step_count == max_steps,
          f"step_count={env._step_count}")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 5 — terminated and truncated are mutually exclusive
# ──────────────────────────────────────────────────────────────────────────────

def test_terminated_truncated_mutually_exclusive():
    print("\n[TEST 5] terminated and truncated are mutually exclusive")

    env        = WorldModelEnv(model=None, buffer=None, device="cpu")
    violations = 0

    for _ in range(50):
        env.reset()
        for _ in range(env.max_steps + 1):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated and truncated:
                violations += 1
            if terminated or truncated:
                break

    check("terminated and truncated never both True", violations == 0,
          f"violations={violations}")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 6 — reset() with z_init override produces correct first obs
# ──────────────────────────────────────────────────────────────────────────────

def test_reset_z_init_override():
    print("\n[TEST 6] reset() z_init override produces correct first obs")

    env = WorldModelEnv(model=_NeverDoneModel(), buffer=None, device="cpu")

    # All three accepted input shapes should work
    for shape, label in [
        ((LATENT_DIM,),      "(384,)"),
        ((1, LATENT_DIM),    "(1, 384)"),
        ((1, 1, LATENT_DIM), "(1, 1, 384)"),
    ]:
        z = torch.ones(shape)
        obs, info = env.reset(z_init=z)
        check(f"z_init shape {label} accepted — obs shape correct",
              obs.shape == (LATENT_DIM,), str(obs.shape))
        check(f"info source == z_init_override for {label}",
              info["source"] == "z_init_override")

    # Verify obs actually reflects the override values
    z_known      = torch.zeros(LATENT_DIM)
    z_known[0]   = 99.0
    obs, _       = env.reset(z_init=z_known)
    check("obs[0] reflects z_init value",
          abs(obs[0] - 99.0) < 1e-5, f"obs[0]={obs[0]:.4f}")

    # Verify reproducibility — same z_init after stepping gives same obs
    obs_a, _ = env.reset(z_init=z_known)
    env.step(0)
    obs_b, _ = env.reset(z_init=z_known)
    check("same z_init after step → identical obs",
          np.allclose(obs_a, obs_b, atol=1e-6),
          f"max_diff={np.abs(obs_a - obs_b).max():.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 7 — invalid actions raise ValueError
# ──────────────────────────────────────────────────────────────────────────────

def test_invalid_action_raises():
    print("\n[TEST 7] Invalid actions raise ValueError")

    env = WorldModelEnv(model=None, buffer=None, device="cpu")
    env.reset()

    for bad_action in [-1, ACTION_DIM, 99, -100]:
        raised = False
        try:
            env.step(bad_action)
        except ValueError:
            raised = True
        check(f"action={bad_action} raises ValueError", raised)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  WorldModelEnv Smoke Tests")
    print("=" * 60)

    test_1000_steps_no_crash()
    test_done_triggers()
    test_reward_non_constant()
    test_truncated_fires_at_max_steps()
    test_terminated_truncated_mutually_exclusive()
    test_reset_z_init_override()
    test_invalid_action_raises()

    print("\n" + "=" * 60)
    print("  ✅  ALL ENV SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()