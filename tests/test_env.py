"""
tests/test_env.py
==================
Verifies that the MazeEnv (Member A's deliverable) satisfies the environment
contract defined in the GDP Plan.

This is the automated version of Integration Checkpoint 1 (Feb 12):
"The Data Handshake" — it checks that the maze produces valid image observations
before we run collect_data.py and generate thousands of episodes.

What is tested:
  - env.reset()        → returns obs of shape (64, 64, 3), dtype uint8
  - env.step(action)   → returns (obs, reward, done, ...) with correct types
  - obs pixel range    → values in [0, 255]
  - action_space       → Discrete(4)  — Up/Down/Left/Right
  - episode terminates → env reaches done=True within max_steps
  - full episode loop  → no crash over a complete episode

How to run:
  python -m pytest tests/test_env.py -v
  # or directly:
  python tests/test_env.py
"""

import sys
from pathlib import Path

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants — must match GDP Plan Interface Contract ─────────────────────────
EXPECTED_OBS_SHAPE = (64, 64, 3)
EXPECTED_N_ACTIONS = 4
MAX_STEPS          = 200   # generous upper bound for termination test

PASS = "✅"
FAIL = "❌"


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {name}{suffix}")
    assert condition, f"ENV CONTRACT VIOLATION: {name}{suffix}"


# ──────────────────────────────────────────────────────────────────────────────
# TEST 1 — Import and instantiation
# ──────────────────────────────────────────────────────────────────────────────

def test_env_imports():
    """MazeEnv and MazeConfig can be imported and instantiated without error."""
    print("\n[TEST 1] Import and instantiation")

    from src.env.maze_env import MazeEnv, MazeConfig

    cfg = MazeConfig(grid_size=10, max_steps=64, obs_size=64, wall_prob=0.20, seed=0)
    env = MazeEnv(cfg)

    check("MazeEnv instantiates",       env is not None)
    check("has action_space attribute", hasattr(env, "action_space"))
    check("has observation_space",      hasattr(env, "observation_space"))

    return env


# ──────────────────────────────────────────────────────────────────────────────
# TEST 2 — Action space contract
# Contract: action_space is Discrete(4)
# ──────────────────────────────────────────────────────────────────────────────

def test_action_space(env):
    """Action space must be Discrete(4) — Up/Down/Left/Right."""
    print("\n[TEST 2] Action space  (contract: Discrete(4))")

    n = int(env.action_space.n)
    check(f"action_space.n == {EXPECTED_N_ACTIONS}", n == EXPECTED_N_ACTIONS,
          f"got n={n}")
    check("action_space.sample() works", env.action_space.sample() in range(n))


# ──────────────────────────────────────────────────────────────────────────────
# TEST 3 — reset() observation shape and type
# Contract: reset() → obs shape (64, 64, 3), dtype uint8
# ──────────────────────────────────────────────────────────────────────────────

def test_reset_obs(env):
    """reset() must return a (64, 64, 3) uint8 RGB array."""
    print("\n[TEST 3] reset() observation  (A.1 contract)")

    result = env.reset()
    # gymnasium returns (obs, info), gym returns obs directly
    obs = result[0] if isinstance(result, tuple) else result

    check("obs is a numpy array",         isinstance(obs, np.ndarray),
          type(obs).__name__)
    check(f"obs shape == {EXPECTED_OBS_SHAPE}", obs.shape == EXPECTED_OBS_SHAPE,
          f"got {obs.shape}")
    check("obs dtype == uint8",           obs.dtype == np.uint8,
          f"got {obs.dtype}")
    check("obs min >= 0",                 obs.min() >= 0,    f"min={obs.min()}")
    check("obs max <= 255",               obs.max() <= 255,  f"max={obs.max()}")
    check("obs has non-zero variance",    obs.var() > 0,
          "all pixels identical — rendering may be broken")

    return obs


# ──────────────────────────────────────────────────────────────────────────────
# TEST 4 — step() output shape and types
# Contract: step(action) → (obs (64,64,3) uint8, reward float, done bool, ...)
# ──────────────────────────────────────────────────────────────────────────────

def test_step_output(env):
    """step() must return obs with correct shape, a float reward, and a bool done."""
    print("\n[TEST 4] step() output  (A.1 contract)")

    env.reset()
    action    = int(env.action_space.sample())
    step_out  = env.step(action)

    # Handle both gymnasium (5-tuple) and gym (4-tuple) APIs
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = step_out

    check("obs is a numpy array",         isinstance(obs, np.ndarray),
          type(obs).__name__)
    check(f"obs shape == {EXPECTED_OBS_SHAPE}", obs.shape == EXPECTED_OBS_SHAPE,
          f"got {obs.shape}")
    check("obs dtype == uint8",           obs.dtype == np.uint8)
    check("reward is a number",           isinstance(reward, (int, float)))
    check("done is a bool",               isinstance(done, bool))
    check("obs values in [0, 255]",       obs.min() >= 0 and obs.max() <= 255)


# ──────────────────────────────────────────────────────────────────────────────
# TEST 5 — Full episode loop runs without crash
# ──────────────────────────────────────────────────────────────────────────────

def test_full_episode(env):
    """A full episode (reset → step loop → done) runs without any exception."""
    print("\n[TEST 5] Full episode loop  (no crash check)")

    result = env.reset()
    obs    = result[0] if isinstance(result, tuple) else result

    steps     = 0
    done      = False
    obs_shapes_ok = True

    while not done and steps < MAX_STEPS:
        action   = int(env.action_space.sample())
        step_out = env.step(action)

        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out

        if obs.shape != EXPECTED_OBS_SHAPE:
            obs_shapes_ok = False
        steps += 1

    check("episode ran at least 1 step",  steps > 0,              f"steps={steps}")
    check("all step obs have correct shape", obs_shapes_ok)
    check(f"episode ended within {MAX_STEPS} steps",
          done or steps < MAX_STEPS,
          f"steps={steps}, done={done}")

    print(f"       Episode length: {steps} steps, terminated: {done}")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 6 — Multiple resets produce different mazes (seeding works)
# ──────────────────────────────────────────────────────────────────────────────

def test_reset_is_stochastic():
    """
    Two envs created with different seeds should produce different observations.
    (Guards against the maze always rendering the same layout.)
    """
    print("\n[TEST 6] Maze stochasticity  (different seeds → different mazes)")

    from src.env.maze_env import MazeEnv, MazeConfig

    cfg_a = MazeConfig(grid_size=10, max_steps=64, obs_size=64, seed=0)
    cfg_b = MazeConfig(grid_size=10, max_steps=64, obs_size=64, seed=999)

    env_a = MazeEnv(cfg_a)
    env_b = MazeEnv(cfg_b)

    result_a = env_a.reset()
    result_b = env_b.reset()

    obs_a = result_a[0] if isinstance(result_a, tuple) else result_a
    obs_b = result_b[0] if isinstance(result_b, tuple) else result_b

    import numpy as np
    are_different = not np.array_equal(obs_a, obs_b)
    check("different seeds → different observations", are_different,
          "both mazes look identical — check seed logic")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MazeEnv Tests  (GDP Plan §3 — Member A, A.1 contract)")
    print("=" * 60)

    try:
        env = test_env_imports()
        test_action_space(env)
        test_reset_obs(env)
        test_step_output(env)
        test_full_episode(env)
        test_reset_is_stochastic()

        print("\n" + "=" * 60)
        print("  ✅  ALL ENV TESTS PASSED — Maze is ready for data collection.")
        print("=" * 60)

    except ImportError as e:
        print(f"\n  ❌  Cannot import MazeEnv: {e}")
        print("     Make sure src/env/maze_env.py exists and src/env/__init__.py is present.")
        sys.exit(1)


if __name__ == "__main__":
    main()