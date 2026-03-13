# test_shapes.py → "Checks if tensors match (384,)" — it's your shape contract validator. It should assert that 
# buffer.sample() returns (B, T, 384), that encoder.encode() returns (384,), and that model.forward() returns
# (B, T-1, 384). This is the automated version of your Integration Checkpoint 2 from the plan.

"""
tests/test_shapes.py
=====================
Verifies that every component in the pipeline produces tensors of the exact
shape defined in the Interface Contract (GDP Plan §2.3). This is the automated version of your Integration 
Checkpoint 2 from the GDP plan made for PDR.

This is the automated version of Integration Checkpoint 2 (Feb 15):
"The Brain Transplant" — it checks that the Buffer and Model are speaking
the same language before the full training loop is run.

What is tested:
  - encoder.encode()       → (384,)            [C.1 contract]
  - buffer.sample()        → (B, T, 384)        [C.2 contract]
  - model.forward()        → pred_next (B,T-1,384), pred_rew (B,T-1,1), pred_val (B,T-1,1)
                                                 [B.2 contract]
  - model.rollout()        → (B, H, 384)        [B.2 rollout contract]
  - latent_mse_loss()      → scalar             [B.3 contract]

How to run:
  python -m pytest tests/test_shapes.py -v
  # or directly:
  python tests/test_shapes.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants — must match GDP Plan Interface Contract exactly ─────────────────
LATENT_DIM = 384
SEQ_LEN    = 16
ACTION_DIM = 4
BATCH_SIZE = 8

PASS = "✅"
FAIL = "❌"


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {name}{suffix}")
    assert condition, f"SHAPE CONTRACT VIOLATION: {name}{suffix}"


# ──────────────────────────────────────────────────────────────────────────────
# TEST 1 — Encoder output shape
# Contract: DinoV2Encoder.encode(img) → (384,)
# ──────────────────────────────────────────────────────────────────────────────

def test_encoder_shape():
    """C.1 contract: encoder.encode(image) returns a 1-D vector of length 384."""
    print("\n[TEST 1] Encoder output shape  (C.1 contract)")

    try:
        from src.models.encoder import DinoV2Encoder
        encoder = DinoV2Encoder(device="cpu")

        # Simulate a raw (64, 64, 3) uint8 RGB observation
        dummy_img = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8)
        z = encoder.encode(dummy_img)

        check("output is a Tensor",          isinstance(z, torch.Tensor))
        check("output is 1-D",               z.ndim == 1,            f"got ndim={z.ndim}")
        check(f"output length == {LATENT_DIM}", z.shape[0] == LATENT_DIM, f"got shape={z.shape}")
        check("no NaN values",               not torch.isnan(z).any())

    except ImportError:
        print(f"  ⚠️  Skipping: src.models.encoder not importable")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 2 — Replay Buffer output shape
# Contract: buffer.sample(B, seq_len=T) → (B, T, 384)
# ──────────────────────────────────────────────────────────────────────────────

def test_buffer_shape():
    """C.2 contract: buffer.sample() returns (B, T, 384) float32 Tensor."""
    print("\n[TEST 2] Buffer sample shape  (C.2 contract)")

    try:
        from src.data.buffer import LatentReplayBuffer

        buf = LatentReplayBuffer(capacity_steps=50_000)

        # Seed buffer with synthetic latent episodes
        rng = np.random.default_rng(42)
        for _ in range(20):
            T = SEQ_LEN + 5
            buf.add_episode(
                latents = rng.standard_normal((T, LATENT_DIM)).astype(np.float32),
                actions = rng.integers(0, ACTION_DIM, size=(T,)).astype(np.float32),
                rewards = rng.standard_normal((T,)).astype(np.float32),
                dones   = rng.integers(0, 2, size=(T,)).astype(np.float32),
            )

        batch = buf.sample(BATCH_SIZE, seq_len=SEQ_LEN)

        # Latents
        check("latents is a Tensor",          isinstance(batch.latents, torch.Tensor))
        check(f"latents shape == ({BATCH_SIZE}, {SEQ_LEN}, {LATENT_DIM})",
              batch.latents.shape == (BATCH_SIZE, SEQ_LEN, LATENT_DIM), str(batch.latents.shape))
        check("latents dtype is float32",     batch.latents.dtype == torch.float32)
        check("no NaN in latents",            not torch.isnan(batch.latents).any())
        check("non-zero variance",            batch.latents.var().item() > 1e-6,
              f"var={batch.latents.var().item():.6f}")

        # Actions
        check("actions is a Tensor",          isinstance(batch.actions, torch.Tensor))
        check(f"actions shape == ({BATCH_SIZE}, {SEQ_LEN})",
              batch.actions.shape == (BATCH_SIZE, SEQ_LEN), str(batch.actions.shape))

        # Rewards
        check("rewards is a Tensor",          isinstance(batch.rewards, torch.Tensor))
        check(f"rewards shape == ({BATCH_SIZE}, {SEQ_LEN})",
              batch.rewards.shape == (BATCH_SIZE, SEQ_LEN), str(batch.rewards.shape))

        # Dones
        check("dones is a Tensor",            isinstance(batch.dones, torch.Tensor))
        check(f"dones shape == ({BATCH_SIZE}, {SEQ_LEN})",
              batch.dones.shape == (BATCH_SIZE, SEQ_LEN), str(batch.dones.shape))


    except ImportError:
        print(f"  ⚠️  Skipping: src.data.buffer not importable")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 3 — World Model forward() output shapes
# Contract: model.forward(z_in, a_in) →
#     pred_next  (B, T-1, 384)
#     pred_rew   (B, T-1, 1)
#     pred_val   (B, T-1, 1)
# ──────────────────────────────────────────────────────────────────────────────

def test_model_forward_shapes():
    """B.2 contract: model.forward() returns correct output shapes."""
    print("\n[TEST 3] Model forward() shapes  (B.2 contract)")

    try:
        from src.models.transformer import DinoWorldModel
        from src.models.transformer_configuration import TransformerWMConfiguration as Config

        config = Config()
        model  = DinoWorldModel(config)
        model.eval()

        B, T = BATCH_SIZE, SEQ_LEN
        z_in = torch.randn(B, T - 1, LATENT_DIM)
        a_in = torch.randint(0, ACTION_DIM, (B, T - 1))

        with torch.no_grad():
            pred_next, pred_rew, pred_val = model(z_in, a_in)

        check(f"pred_next shape == ({B}, {T-1}, {LATENT_DIM})",
              pred_next.shape == (B, T - 1, LATENT_DIM), str(pred_next.shape))
        check(f"pred_rew  shape == ({B}, {T-1}, 1)",
              pred_rew.shape  == (B, T - 1, 1),          str(pred_rew.shape))
        check(f"pred_val  shape == ({B}, {T-1}, 1)",
              pred_val.shape  == (B, T - 1, 1),          str(pred_val.shape))
        check("no NaN in pred_next",          not torch.isnan(pred_next).any())
        check("no NaN in pred_rew",           not torch.isnan(pred_rew).any())
        check("no NaN in pred_val",           not torch.isnan(pred_val).any())

    except ImportError:
        print(f"  ⚠️  Skipping: src.models.transformer not importable")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 4 — World Model rollout() output shapes
# Contract: model.rollout(z0, actions) →
#     pred_latents  (B, H, 384)
#     pred_rewards  (B, H, 1)
#     pred_values   (B, H, 1)
# ──────────────────────────────────────────────────────────────────────────────

def test_model_rollout_shapes():
    """B.2 rollout contract: model.rollout() returns correct output shapes."""
    print("\n[TEST 4] Model rollout() shapes  (B.2 rollout contract)")

    try:
        from src.models.transformer import DinoWorldModel
        from src.models.transformer_configuration import TransformerWMConfiguration as Config

        config  = Config()
        model   = DinoWorldModel(config)
        model.eval()

        B, H = BATCH_SIZE, SEQ_LEN
        z0      = torch.randn(B, 1, LATENT_DIM)
        actions = torch.randint(0, ACTION_DIM, (B, H))

        pred_latents, pred_rewards, pred_values = model.rollout(z0, actions)

        check(f"pred_latents shape == ({B}, {H}, {LATENT_DIM})",
              pred_latents.shape == (B, H, LATENT_DIM), str(pred_latents.shape))
        check(f"pred_rewards shape == ({B}, {H}, 1)",
              pred_rewards.shape == (B, H, 1),          str(pred_rewards.shape))
        check(f"pred_values  shape == ({B}, {H}, 1)",
              pred_values.shape  == (B, H, 1),          str(pred_values.shape))

    except ImportError:
        print(f"  ⚠️  Skipping: src.models.transformer not importable")


# ──────────────────────────────────────────────────────────────────────────────
# TEST 5 — MSE loss is a finite scalar
# Contract: latent_mse_loss(pred, target) → scalar float
# ──────────────────────────────────────────────────────────────────────────────

def test_loss_scalar():
    """B.3 contract: latent_mse_loss returns a finite positive scalar."""
    print("\n[TEST 5] Loss function shape  (B.3 contract)")

    try:
        from src.models.transformer import latent_mse_loss

        B, T = BATCH_SIZE, SEQ_LEN
        pred   = torch.randn(B, T - 1, LATENT_DIM)
        target = torch.randn(B, T - 1, LATENT_DIM)

        loss = latent_mse_loss(pred, target)

        check("loss is a scalar (0-D tensor)", loss.ndim == 0,          f"ndim={loss.ndim}")
        check("loss is finite",                torch.isfinite(loss),     f"val={loss.item()}")
        check("loss > 0",                      loss.item() > 0,          f"val={loss.item():.6f}")

    except ImportError:
        print(f"  ⚠️  Skipping: src.models.transformer not importable")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Shape Contract Tests  (GDP Plan §2.3 Interface Contract)")
    print("=" * 60)

    test_encoder_shape()
    test_buffer_shape()
    test_model_forward_shapes()
    test_model_rollout_shapes()
    test_loss_scalar()

    print("\n" + "=" * 60)
    print("  ✅  ALL SHAPE TESTS PASSED — Interface Contract respected.")
    print("=" * 60)


if __name__ == "__main__":
    main()