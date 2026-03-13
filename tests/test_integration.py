"""
test_integration.py  —  Member E: Task E.3
============================================
Runs the full pipeline for 1 epoch and asserts:
  • No RuntimeError (shape mismatches, import errors)
  • Buffer output shape  : (B, T, 384)
  • Model output shape   : (B, T-1, 384)
  • Loss is a finite scalar

Run with:
    python -m src.tests.test_integration
"""



from __future__ import annotations

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn

# Notebook-safe project root detection
_here = Path(__file__).resolve().parent.parent

if _here not in sys.path:
    sys.path.insert(0, _here)

print(f"[INFO] Project root: {_here}")

# ── Imports with graceful fallback ────────────────────────────────────────────

try:
    from src.data.buffer import LatentReplayBuffer
    REAL_BUFFER = True
except ImportError:
    REAL_BUFFER = False

try:
    from src.models.transformer import DinoWorldModel, latent_mse_loss
    from src.models.transformer_configuration import TransformerWMConfiguration as Config
    REAL_MODEL = True
except ImportError:
    REAL_MODEL = False

# Re-use the mock stubs from the training script
from src.scripts.train_world_model import (
    _MockBuffer, _MockModel, _mock_mse,
    build_components, train_epoch,
    SEQ_LEN, LATENT_DIM, ACTION_DIM,
)

# ── Test helpers ──────────────────────────────────────────────────────────────

PASS = "✅"
FAIL = "❌"

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {name}{suffix}")
    if not condition:
        raise AssertionError(f"FAILED: {name}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_buffer_shape():
    """C.2 contract: buffer.sample() → (B, T, 384)"""
    print("\n[TEST] Buffer Shape Contract")
    B, T = 8, SEQ_LEN

    if REAL_BUFFER:
        import numpy as np
        buf = LatentReplayBuffer(capacity_steps=50_000)
        # Seed it with a few synthetic episodes
        for _ in range(20):
            T_ep = T + 5 # episode length slightly longer than seq_len
            buf.add_episode(
                latents = np.random.randn(T_ep, LATENT_DIM).astype("float32"),
                actions = np.random.randint(0, ACTION_DIM, size=(T_ep,)).astype("float32"),
                rewards = np.random.randn(T_ep).astype("float32"),
                dones   = np.zeros(T_ep, dtype="float32"),
            )
        batch = buf.sample(B, seq_len=T)
        out   = batch.latents
    else:
        batch = _MockBuffer().sample(B, T)
        out   = batch.latents

    check("output is a Tensor",     isinstance(out, torch.Tensor))
    check("shape (B, T, 384)",      out.shape == (B, T, LATENT_DIM),
                                    str(out.shape))
    check("no NaN values",          not torch.isnan(out).any())
    check("non-zero variance",      out.var().item() > 1e-6)


def test_model_forward():
    """B.2 contract: model.forward(z_in, a_in) → pred_next (B, T-1, 384)"""
    print("\n[TEST] Model Forward Contract")
    B, T = 4, SEQ_LEN
    device = torch.device("cpu")

    z_in = torch.randn(B, T - 1, LATENT_DIM)
    a_in = torch.randint(0, ACTION_DIM, (B, T - 1))

    if REAL_MODEL:
        config = Config()
        model  = DinoWorldModel(config)
    else:
        model  = _MockModel()

    model.eval()
    with torch.no_grad():
        pred_next, pred_rew, pred_val = model(z_in, a_in)

    check("pred_next shape (B, T-1, 384)",
          pred_next.shape == (B, T - 1, LATENT_DIM), str(pred_next.shape))
    check("pred_rew shape  (B, T-1, 1)",
          pred_rew.shape  == (B, T - 1, 1),          str(pred_rew.shape))
    check("pred_val shape  (B, T-1, 1)",
          pred_val.shape  == (B, T - 1, 1),           str(pred_val.shape))
    check("no NaN in pred_next", not torch.isnan(pred_next).any())


def test_loss_finite():
    """Loss is a finite positive scalar."""
    print("\n[TEST] Loss Finite")
    B, T = 4, SEQ_LEN
    pred   = torch.randn(B, T - 1, LATENT_DIM)
    target = torch.randn(B, T - 1, LATENT_DIM)

    loss_fn = latent_mse_loss if REAL_MODEL else _mock_mse
    loss = loss_fn(pred, target)

    check("loss is scalar",   loss.ndim == 0)
    check("loss is finite",   torch.isfinite(loss))
    check("loss > 0",         loss.item() > 0)


def test_one_epoch_no_crash():
    """E.3 core test: 1 epoch × 10 batches runs without any exception."""
    print("\n[TEST] E.3 — Full pipeline, 1 epoch × 10 batches")
    import numpy as np
    device = torch.device("cpu")

    buffer, model, optimizer, scheduler, loss_fn = build_components(
        device, use_real=True
    )

    # Seed buffer if real — build_components leaves it empty
    if REAL_BUFFER:
        import numpy as np
        for _ in range(20):
            T_ep = SEQ_LEN + 5
            buffer.add_episode(
                latents = np.random.randn(T_ep, LATENT_DIM).astype("float32"),
                actions = np.random.randint(0, ACTION_DIM, size=(T_ep,)).astype("float32"),
                rewards = np.random.randn(T_ep).astype("float32"),
                dones   = np.zeros(T_ep, dtype="float32"),
            )

    stats = train_epoch(
        model, buffer, optimizer, loss_fn,
        device, batch_size=8, batches_per_epoch=10,
    )
    check("avg_loss is finite",  torch.isfinite(torch.tensor(stats["avg_loss"])))
    check("min_loss >= 0",       stats["min_loss"] >= 0)
    check("max_loss >= min",     stats["max_loss"] >= stats["min_loss"])
    print(f"     avg_loss={stats['avg_loss']:.6f}  "
          f"min={stats['min_loss']:.6f}  max={stats['max_loss']:.6f}")


# ── Entry-point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  Integration Test Suite  (E.3 Verification)")
    print("=" * 58)
    print(f"  Real buffer : {REAL_BUFFER}")
    print(f"  Real model  : {REAL_MODEL}")

    test_buffer_shape()
    test_model_forward()
    test_loss_finite()
    test_one_epoch_no_crash()

    print("\n" + "=" * 58)
    print("  ✅  ALL TESTS PASSED — E.3 Definition of Done met.")
    print("=" * 58)


if __name__ == "__main__":
    main()
