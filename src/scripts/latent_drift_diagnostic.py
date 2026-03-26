"""
src/scripts/latent_drift_diagnostic.py
========================================
Latent Drift Diagnostic — autoregressive rollout quality check.

Runs a 50-step autoregressive rollout from 10 different real starting
latents, computes MSE and DDA between predicted and ground-truth latent
at each step, and reports the horizon at which prediction quality degrades.

MSE measures magnitude of error — how far the predicted latent is from
the real one. DDA (Directional Dynamics Accuracy) measures direction
quality — does the WM predict the correct *direction* of change even
when the absolute position has drifted?

    DDA(t) = 1 if cos(z_hat_{t+1} - z_hat_t, z_{t+1} - z_t) > 0
             0 otherwise

A model with DDA > 0.5 is predicting transitions better than random,
even if MSE is high. This matters for navigation: the agent needs the
WM to point toward the goal, not to be at exactly the right position.

Using both metrics together avoids the failure mode where MSE improves
but DDA degrades (e.g. predictions cluster near the prior mean — low
MSE variance, but transitions are meaningless).

Usage
-----
    python -m src.scripts.latent_drift_diagnostic \\
        --checkpoint checkpoints/wm_ss/world_model_best.pt \\
        --data_dir   data/processed \\
        --out_dir    evaluation/wm_ss

Outputs
-------
    evaluation/latent_drift.png   — MSE and DDA vs step horizon (2 subplots)
    evaluation/latent_drift.npz   — raw mse_matrix and dda_matrix arrays
    stdout                        — step thresholds and per-step summary

Interface Contract (GDP Plan §2.3)
------------------------------------
  Latent dim  : 384
  Rollout len : 50 steps
  Seeds       : 10 different real starting latents from buffer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Imports ───────────────────────────────────────────────────────────────────

try:
    from src.models.transformer import DinoWorldModel
    from src.models.transformer_configuration import TransformerWMConfiguration as Config
    _REAL_MODEL = True
except ImportError:
    _REAL_MODEL = False
    print("[WARN] src.models.transformer not found — cannot run diagnostic")

try:
    from src.data.buffer import LatentReplayBuffer
    _REAL_BUFFER = True
except ImportError:
    _REAL_BUFFER = False
    print("[WARN] src.data.buffer not found — cannot run diagnostic")

# ── Constants ─────────────────────────────────────────────────────────────────

LATENT_DIM    = 384
ACTION_DIM    = 4
ROLLOUT_STEPS = 50
N_SEEDS       = 10


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> "DinoWorldModel":
    """Load DinoWorldModel weights from a checkpoint saved by train_world_model.py."""
    if not _REAL_MODEL:
        raise RuntimeError("src.models.transformer is required for this diagnostic.")

    config = Config()
    model  = DinoWorldModel(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "world_model_state" in ckpt:
        state_dict = ckpt["world_model_state"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("metrics", {}).get("avg_loss", float("nan"))
    print(f"[INFO] Loaded checkpoint  epoch={epoch}  avg_loss={loss:.6f}")
    return model


# ── Buffer loading ────────────────────────────────────────────────────────────

def load_buffer(data_dir: Path) -> "LatentReplayBuffer":
    """Populate buffer from .npz files in data_dir."""
    if not _REAL_BUFFER:
        raise RuntimeError("src.data.buffer is required for this diagnostic.")

    buf   = LatentReplayBuffer(capacity_steps=200_000)
    files = sorted(data_dir.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    loaded = 0
    for fpath in files:
        with np.load(fpath) as data:
            missing = [k for k in ("latents", "actions", "rewards", "dones")
                       if k not in data]
            if missing:
                print(f"[SKIP] {fpath.name} — missing keys: {missing}")
                continue
            buf.add_episode(
                latents=data["latents"],
                actions=data["actions"],
                rewards=data["rewards"],
                dones=data["dones"],
            )
            loaded += 1

    print(f"[INFO] Loaded {loaded}/{len(files)} episodes  "
          f"total_steps={buf.total_steps}")
    return buf


# ── Starting latent extraction ────────────────────────────────────────────────

def get_seed_episodes(buffer: "LatentReplayBuffer", n_seeds: int, rollout_steps: int):
    """
    Return n_seeds episodes long enough for a full rollout.
    Spaced evenly across the buffer for diversity.
    """
    candidates = [ep for ep in buffer.episodes
                  if ep.latents.shape[0] >= rollout_steps + 1]

    if len(candidates) < n_seeds:
        raise ValueError(
            f"Need {n_seeds} episodes of length >= {rollout_steps + 1}, "
            f"but only {len(candidates)} qualify."
        )

    indices = np.linspace(0, len(candidates) - 1, n_seeds, dtype=int)
    return [candidates[i] for i in indices]


# ── DDA helper ────────────────────────────────────────────────────────────────

def compute_dda_step(
    pred_delta: torch.Tensor,   # (1, 1, 384) — predicted transition vector
    real_delta: torch.Tensor,   # (1, 1, 384) — real transition vector
) -> float:
    """
    Directional Dynamics Accuracy for a single step.

    Measures whether the predicted transition vector points in the same
    direction as the real transition vector, via cosine similarity sign:

        DDA = 1  if  cos(pred_delta, real_delta) > 0
              0  otherwise

    Why cosine and not dot product?
        Dot product is scale-dependent — a large predicted delta always
        agrees with a tiny real delta in sign. Cosine normalises both
        vectors so only direction matters, not magnitude.

    Why > 0 threshold and not > some positive value?
        DDA > 0 means "better than perpendicular" — the prediction is
        in the correct half-space. A stricter threshold (e.g. > 0.5)
        would be interesting but harder to interpret as a binary metric.
        Random direction gives DDA = 0.5 in expectation (half the time
        a random vector lands in the correct half-space).

    Parameters
    ----------
    pred_delta : (1, 1, 384) — z_hat_{t+1} - z_hat_t  (predicted delta)
    real_delta : (1, 1, 384) — z_{t+1} - z_t          (real delta)

    Returns
    -------
    1.0 if cosine similarity > 0, else 0.0
    """
    # Flatten to (384,) for cosine computation
    p = pred_delta.reshape(-1)   # (384,)
    r = real_delta.reshape(-1)   # (384,)

    # Avoid division by zero — if either delta is near-zero (no movement),
    # cosine is undefined. Return 0.5 (neutral, no signal) in that case.
    p_norm = p.norm()
    r_norm = r.norm()
    if p_norm < 1e-8 or r_norm < 1e-8:
        return 0.5

    cos_sim = torch.dot(p / p_norm, r / r_norm).item()
    return 1.0 if cos_sim > 0.0 else 0.0


# ── Core diagnostic ───────────────────────────────────────────────────────────

def run_drift_diagnostic(
    model:         "DinoWorldModel",
    seed_episodes: list,
    device:        torch.device,
    rollout_steps: int = ROLLOUT_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each seed episode, run an autoregressive rollout for rollout_steps,
    replaying the episode's recorded actions.

    Computes at each step t:
    - MSE between predicted latent z_hat_{t+1} and real z_{t+1}
    - DDA: whether pred delta direction matches real delta direction

    Returns
    -------
    mse_matrix : np.ndarray (n_seeds, rollout_steps)
    dda_matrix : np.ndarray (n_seeds, rollout_steps)
        DDA values are 0.0, 0.5 (undefined), or 1.0 per step per seed.
        Mean over seeds gives DDA ∈ [0, 1]. Random baseline = 0.5.
    """
    n_seeds    = len(seed_episodes)
    mse_matrix = np.zeros((n_seeds, rollout_steps), dtype=np.float32)
    dda_matrix = np.zeros((n_seeds, rollout_steps), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for s, ep in enumerate(seed_episodes):
            gt = torch.tensor(
                ep.latents[:rollout_steps + 1],
                dtype=torch.float32,
                device=device,
            )   # (rollout_steps+1, 384)

            recorded_actions = torch.tensor(
                ep.actions[:rollout_steps],
                dtype=torch.long,
                device=device,
            )   # (rollout_steps,)

            # Seed context with real z_0
            z_history = gt[0].unsqueeze(0).unsqueeze(0)   # (1, 1, 384)
            a_history = None
            prev_pred = gt[0].unsqueeze(0).unsqueeze(0)   # (1, 1, 384) — tracks z_hat_t

            for h in range(rollout_steps):
                a = recorded_actions[h].unsqueeze(0).unsqueeze(0)   # (1, 1)
                if a_history is None:
                    a_history = a
                else:
                    a_history = torch.cat([a_history, a], dim=1)

                pred_seq, _, _ = model(z_history, a_history)
                pred_next = pred_seq[:, -1:, :]   # (1, 1, 384) — z_hat_{h+1}
                gt_next   = gt[h + 1].unsqueeze(0).unsqueeze(0)   # (1, 1, 384)

                # ── MSE ───────────────────────────────────────────────────────
                mse = torch.mean((pred_next - gt_next) ** 2).item()
                mse_matrix[s, h] = mse

                # ── DDA ───────────────────────────────────────────────────────
                # Predicted delta: z_hat_{h+1} - z_hat_h
                # Real delta:      z_{h+1}     - z_h
                # prev_pred holds z_hat_h (the prediction from the previous step,
                # or z_0 at h=0 since the first delta is from the real seed).
                pred_delta = pred_next - prev_pred               # (1, 1, 384)
                real_delta = gt_next   - gt[h].unsqueeze(0).unsqueeze(0)  # (1, 1, 384)

                dda_matrix[s, h] = compute_dda_step(pred_delta, real_delta)

                # Advance history
                z_history = torch.cat([z_history, pred_next], dim=1)
                prev_pred = pred_next

    return mse_matrix, dda_matrix


# ── Threshold detection ───────────────────────────────────────────────────────

def find_drift_threshold(mean_mse: np.ndarray) -> int | None:
    """First step h where mean_mse[h] > 2 × mean_mse[0]. None if never."""
    baseline = mean_mse[0]
    for h, mse in enumerate(mean_mse):
        if mse > 2.0 * baseline:
            return h
    return None


def find_dda_threshold(mean_dda: np.ndarray) -> int | None:
    """
    First step h where mean_dda[h] drops below 0.6.
    0.6 chosen as a conservative margin above random (0.5) — below this
    the WM is barely better than random at predicting transition direction.
    Returns None if DDA stays above 0.6 throughout.
    """
    for h, dda in enumerate(mean_dda):
        if dda < 0.6:
            return h
    return None


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_drift(
    mse_matrix:     np.ndarray,
    dda_matrix:     np.ndarray,
    threshold_step: int | None,
    dda_threshold:  int | None,
    out_dir:        Path,
):
    """
    Two-subplot figure:
      Top:    MSE vs step horizon (unchanged from original)
      Bottom: DDA vs step horizon with random baseline at 0.5

    Saves to out_dir/latent_drift.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds, rollout_steps = mse_matrix.shape
    steps    = np.arange(1, rollout_steps + 1)
    mean_mse = mse_matrix.mean(axis=0)
    std_mse  = mse_matrix.std(axis=0)
    mean_dda = dda_matrix.mean(axis=0)
    std_dda  = dda_matrix.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    # ── Top: MSE ──────────────────────────────────────────────────────────────
    for s in range(n_seeds):
        ax1.plot(steps, mse_matrix[s], color="steelblue", alpha=0.25,
                 linewidth=0.8, label="_nolegend_")

    ax1.plot(steps, mean_mse, color="steelblue", linewidth=2.5,
             label=f"Mean MSE ({n_seeds} seeds)")
    ax1.fill_between(steps, mean_mse - std_mse, mean_mse + std_mse,
                     color="steelblue", alpha=0.15, label="±1 std")

    baseline = mean_mse[0]
    ax1.axhline(2.0 * baseline, color="tomato", linestyle="--", linewidth=1.5,
                label=f"2× step-1 MSE ({2.0 * baseline:.4f})")

    if threshold_step is not None:
        ax1.axvline(threshold_step + 1, color="tomato", linestyle=":",
                    linewidth=2.0,
                    label=f"Drift threshold: step {threshold_step + 1}")
        ax1.scatter([threshold_step + 1], [mean_mse[threshold_step]],
                    color="tomato", zorder=5, s=80)

    ax1.set_ylabel("Latent MSE", fontsize=12)
    ax1.set_title("Latent Drift Diagnostic — MSE and DDA vs Rollout Horizon",
                  fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)

    # ── Bottom: DDA ───────────────────────────────────────────────────────────
    for s in range(n_seeds):
        ax2.plot(steps, dda_matrix[s], color="seagreen", alpha=0.2,
                 linewidth=0.8, label="_nolegend_")

    ax2.plot(steps, mean_dda, color="seagreen", linewidth=2.5,
             label=f"Mean DDA ({n_seeds} seeds)")
    ax2.fill_between(steps, mean_dda - std_dda, mean_dda + std_dda,
                     color="seagreen", alpha=0.15, label="±1 std")

    # Random baseline — a random direction agrees with real delta 50% of the time
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1.5,
                label="Random baseline (0.5)")

    # DDA degradation threshold
    ax2.axhline(0.6, color="orange", linestyle="--", linewidth=1.2,
                label="DDA threshold (0.6)")

    if dda_threshold is not None:
        ax2.axvline(dda_threshold + 1, color="orange", linestyle=":",
                    linewidth=2.0,
                    label=f"DDA < 0.6 at step {dda_threshold + 1}")
        ax2.scatter([dda_threshold + 1], [mean_dda[dda_threshold]],
                    color="orange", zorder=5, s=80)

    ax2.set_xlabel("Rollout Step (horizon)", fontsize=12)
    ax2.set_ylabel("DDA (directional accuracy)", fontsize=12)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(fontsize=10)
    ax2.set_xlim(1, rollout_steps)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "latent_drift.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Latent Drift Diagnostic — MSE and DDA vs rollout horizon"
    )
    p.add_argument("--checkpoint",    type=str, default="checkpoints/dqn_dream.pt")
    p.add_argument("--data_dir",      type=str, default="data/processed")
    p.add_argument("--out_dir",       type=str, default="evaluation")
    p.add_argument("--rollout_steps", type=int, default=ROLLOUT_STEPS)
    p.add_argument("--n_seeds",       type=int, default=N_SEEDS)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 58)
    print("  Latent Drift Diagnostic (MSE + DDA)")
    print("=" * 58)
    print(f"[INFO] Checkpoint    : {args.checkpoint}")
    print(f"[INFO] Data dir      : {args.data_dir}")
    print(f"[INFO] Rollout steps : {args.rollout_steps}")
    print(f"[INFO] Seeds         : {args.n_seeds}")
    print(f"[INFO] Device        : {device}")

    # 1. Load model and buffer
    model  = load_model(Path(args.checkpoint), device)
    buffer = load_buffer(Path(args.data_dir))

    # 2. Extract diverse seed episodes
    seed_episodes = get_seed_episodes(buffer, args.n_seeds, args.rollout_steps)
    print(f"[INFO] Using {len(seed_episodes)} seed episodes")

    # 3. Run autoregressive rollout — collect MSE and DDA
    print(f"\n[INFO] Running {args.rollout_steps}-step autoregressive rollout "
          f"from {args.n_seeds} seeds...")
    mse_matrix, dda_matrix = run_drift_diagnostic(
        model, seed_episodes, device,
        rollout_steps=args.rollout_steps,
    )

    # 4. Compute means and find thresholds
    mean_mse       = mse_matrix.mean(axis=0)
    mean_dda       = dda_matrix.mean(axis=0)
    threshold_step = find_drift_threshold(mean_mse)
    dda_threshold  = find_dda_threshold(mean_dda)

    # 5. Report
    print("\n─── Drift Report " + "─" * 41)
    print(f"  {'Step':<8}  {'MSE':>10}  {'DDA':>8}")
    print(f"  {'----':<8}  {'---':>10}  {'---':>8}")
    for step in (1, 10, 25, 50):
        idx = step - 1
        if idx < len(mean_mse):
            print(f"  {step:<8}  {mean_mse[idx]:>10.6f}  {mean_dda[idx]:>8.3f}")

    print()
    if threshold_step is not None:
        print(f"  ⚠️  MSE exceeds 2× baseline at step {threshold_step + 1}  "
              f"(MSE={mean_mse[threshold_step]:.6f}  "
              f"threshold={2.0 * mean_mse[0]:.6f})")
    else:
        print(f"  ✅  MSE never exceeded 2× baseline across {args.rollout_steps} steps.")

    if dda_threshold is not None:
        print(f"  ⚠️  DDA drops below 0.6 at step {dda_threshold + 1}  "
              f"(DDA={mean_dda[dda_threshold]:.3f})")
    else:
        print(f"  ✅  DDA stayed above 0.6 across all {args.rollout_steps} steps.")

    # 6. Save raw matrices for later analysis / report figures
    npz_path = Path(args.out_dir) / "latent_drift.npz"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, mse_matrix=mse_matrix, dda_matrix=dda_matrix)
    print(f"\n[INFO] Raw data saved → {npz_path}")

    # 7. Plot
    plot_drift(mse_matrix, dda_matrix, threshold_step, dda_threshold,
               Path(args.out_dir))

    print("\n" + "=" * 58)
    print("  Latent Drift Diagnostic complete.")
    print("=" * 58)


if __name__ == "__main__":
    main()