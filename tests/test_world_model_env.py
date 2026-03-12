"""
tests/test_world_model_env.py
==============================
Smoke-tests for WorldModelEnv (GDP Plan integration check).

Verifies:
  1. 1000 random steps run without crash
  2. done triggers at least once across 100 episodes
  3. Reward is non-constant (has variance)

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

from src.env.world_model_env import WorldModelEnv

PASS = "✅"
FAIL = "❌"


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {name}{suffix}")
    assert condition, f"SMOKE TEST VIOLATION: {name}{suffix}"


# ──────────────────────────────────────────────────────────────────────────────
# TEST 1 — 1000 random steps without crash
# ──────────────────────────────────────────────────────────────────────────────

def test_1000_steps_no_crash():
    print("\n[TEST 1] 1000 random steps without crash")

    env   = WorldModelEnv(model=None, buffer=None, device="cpu")
    steps = 0

    obs, info = env.reset()

    while steps < 1000:
        action                          = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Shape contract holds every single step
        assert obs.shape == (384,), \
            f"Bad obs shape at step {steps}: {obs.shape}"
        assert isinstance(reward, float), \
            f"Reward not float at step {steps}: {type(reward)}"
        assert isinstance(terminated, bool), \
            f"terminated not bool at step {steps}"
        assert isinstance(truncated, bool), \
            f"truncated not bool at step {steps}"
        assert "step"       in info, "info missing 'step'"
        assert "done_logit" in info, "info missing 'done_logit'"
        assert "done_prob"  in info, "info missing 'done_prob'"

        steps += 1

        if terminated or truncated:
            obs, info = env.reset()

    check("1000 steps completed without crash", True,
          f"total_steps={steps}")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 2 — done triggers at least once in 100 episodes
# ──────────────────────────────────────────────────────────────────────────────

def test_done_triggers():
    print("\n[TEST 2] done triggers at least once in 100 episodes")

    import torch.nn as nn

    class _ForcedDoneModel(nn.Module):
        """Stub that always outputs done_prob > 0.5 (logit=5 → prob≈0.993)."""
        def forward(self, z_in, a_in):
            pred_next = z_in
            pred_rew  = torch.zeros(*z_in.shape[:-1], 1)
            pred_done = torch.ones(*z_in.shape[:-1], 1) * 5.0   # always done
            return pred_next, pred_rew, pred_done

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

    import torch.nn as nn

    class _VaryingRewardModel(nn.Module):
        """Stub that returns a different reward each step via randn."""
        def forward(self, z_in, a_in):
            pred_next = z_in
            pred_rew  = torch.randn(*z_in.shape[:-1], 1)  # random each step
            pred_done = torch.ones(*z_in.shape[:-1], 1) * -5.0  # never done
            return pred_next, pred_rew, pred_done

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
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  WorldModelEnv Smoke Tests")
    print("=" * 60)

    test_1000_steps_no_crash()
    test_done_triggers()
    test_reward_non_constant()

    print("\n" + "=" * 60)
    print("  ✅  ALL ENV SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()