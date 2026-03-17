# GDP2-WorldModel — Latent World Model Proof-of-Concept

**Cranfield University | AAI Group Design Project 2 | Feb 2026**

A 5-person team project implementing a Latent World Model pipeline for a custom maze environment. The agent learns to *dream* — planning entirely in compressed latent space without touching the real simulator — using a DINOv2 vision encoder and a causal Transformer as the world model core.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Interface Contract](#interface-contract)
- [Repository Structure](#repository-structure)
- [Team Roles](#team-roles)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Running Tests](#running-tests)
- [Evaluation & Diagnostics](#evaluation--diagnostics)
- [File Reference](#file-reference)

---

## Architecture Overview

```
Raw Maze Image (64×64 RGB)
        │
        ▼
  ┌─────────────┐
  │  DINOv2     │  → 384-dim latent vector z_t
  │  Encoder    │     (Member C — encoder.py)
  └─────────────┘
        │
        ▼
  ┌──────────────────────────┐
  │  Latent Replay Buffer    │  stores sequences (T, 384)
  │                          │     (Member C — buffer.py)
  └──────────────────────────┘
        │  (B, T, 384) batches
        ▼
  ┌──────────────────────────┐
  │  World Model Transformer │  predicts z_{t+1}, reward, value
  │  (Causal, RoPE, 4L/4H)  │     (Member B — transformer.py)
  └──────────────────────────┘
        │
        ▼
  ┌─────────────┐
  │ CEM Planner │  plans action sequences in latent space
  │ (MPC-style) │     (Member B — transformer.py)
  └─────────────┘
```

The **Dreaming Loop**: the agent never touches the real environment during planning. It samples action candidates, rolls them out through the World Model, scores predicted returns, and picks the best action — all in imagination.

---

## Interface Contract

These values are **immutable**. Changing any of them breaks everyone else's code.

| Contract | Value |
|---|---|
| Latent dimension | `384` — DINOv2 ViT-S/14 output |
| Sequence length T | `24` steps |
| Input image size | `(64, 64, 3)` RGB uint8 |
| Buffer output shape | `(Batch_Size, 16, 384)` float32 |
| Action space | `Discrete(4)` — 0:Up 1:Down 2:Left 3:Right |
| Model forward output | `pred_next (B,T,384)`, `pred_rew (B,T,1)`, `pred_val (B,T,1)` |

---

## Repository Structure

```
GDP2-WorldModel/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                        # ⚠️ gitignored — local only
│   ├── raw/                     # Raw .npz episodes from collect_data.py
│   └── processed/               # Encoded latent .npz files from encode_dataset.py
│
├── checkpoints/                 # ⚠️ gitignored — local only
│   ├── dqn_baseline.pt
│   ├── world_model_best.pt
│   ├── world_model_epoch0010.pt
│   ├── world_model_epoch0020.pt
│   ├── world_model_epoch0030.pt
│   ├── world_model_epoch0040.pt
│   ├── world_model_epoch0050.pt
│   └── world_model_final.pt
│
├── logs/                        # ⚠️ gitignored — local only
│   └── training_log.json
│
├── evaluation/                  # ⚠️ gitignored — local only
│   └── metrics_summary.json
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── buffer.py            # Latent replay buffer (Member C — C.2)
│   ├── env/
│   │   ├── __init__.py
│   │   └── maze_env.py          # Custom Gymnasium maze (Member A — A.1)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py           # DINOv2 wrapper (Member C — C.1)
│   │   ├── transformer.py       # World Model + CEM Planner (Member B — B.2)
│   │   └── transformer_configuration.py  # Hyperparameters (Member B — B.1)
│   ├── scripts/
│   │   ├── collect_data.py      # Pipeline step 1: run maze, save raw episodes
│   │   ├── encode_dataset.py    # Pipeline step 2: encode images → latents
│   │   ├── train_world_model.py # Pipeline step 3: train world model (Member E — E.2)
│   │   ├── train_baseline.py    # Train DQN baseline agent (Member A — A.3)
│   │   ├── evaluate_world_model.py  # Generate evaluation plots (Member D — D.3)
│   │   ├── evaluate_transfer.py     # Head-to-head: DQN vs CEM agent
│   │   ├── check_deltas.py          # Diagnostic: is model learning real dynamics?
│   │   └── plot_baseline.py         # Generate DQN reward curves
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py           # MSE & cosine similarity (Member D — D.2)
│       └── visualizer.py        # All evaluation plots (Member D — D.3)
│
└── tests/
    ├── test_shapes.py           # Verifies tensor shapes match interface contract
    ├── test_env.py              # Verifies maze env renders correct observations
    └── test_integration.py      # Full pipeline smoke test (E.3 Definition of Done)
```

---

## Team Roles

| Member | Role | Core Deliverables |
|---|---|---|
| A | Environment & Baseline | `maze_env.py`, `collect_data.py`, `train_baseline.py` |
| B | World Model Architect | `transformer.py`, `transformer_configuration.py` |
| C | Data & Vision Engineer | `encoder.py`, `buffer.py`, `encode_dataset.py` |
| D | Validation & Theory | `metrics.py`, `visualizer.py`, `evaluate_world_model.py` |
| E | Integration Lead | `train_world_model.py`, repo structure, final training run |

---

## Setup

```bash
git clone https://github.com/<your-org>/GDP2-WorldModel.git
cd GDP2-WorldModel
pip install -r requirements.txt
```

> **Note:** DINOv2 is loaded from `torch.hub` on first use — requires internet access and ~350 MB download.

---

## Pipeline

Run these steps in order from the repo root.

### Step 1 — Collect raw episodes
```bash
python -m src.scripts.collect_data --episodes 1000
# Output: data/raw/ep_000000.npz ... ep_000999.npz
```

### Step 2 — Encode images to latents
```bash
python -m src.scripts.encode_dataset --raw_dir data/raw --out_dir data/processed
# Output: data/processed/ep_000000.npz  (latents key, shape (T, 384))
# Prints: Latent variance > 0.01 ✅
```

### Step 3 — Integration smoke test
```bash
python -m src.scripts.train_world_model --smoke_test
# Runs 1 epoch × 10 batches — must complete without crash (E.3 Definition of Done)
```

### Step 4 — Full training run
```bash
python -m src.scripts.train_world_model --epochs 50 --batches_per_epoch 200
# Checkpoints → checkpoints/world_model_best.pt
# Logs        → logs/training_log.json
```

### Step 5 — Evaluate and generate plots
```bash
python -m src.scripts.evaluate_world_model \
    --checkpoint checkpoints/world_model_best.pt \
    --data_dir data/processed \
    --log_file logs/training_log.json \
    --out_dir evaluation
# Output: evaluation/training_loss.png
#         evaluation/predicted_vs_actual.png
#         evaluation/cosine_over_horizon.png
#         evaluation/per_step_mse.png
#         evaluation/metrics_summary.json
```

### Step 6 — Head-to-head: DQN baseline vs CEM agent
```bash
python -m src.scripts.evaluate_transfer \
    --dqn_weights checkpoints/dqn_baseline.pt \
    --wm_weights  checkpoints/world_model_best.pt \
    --episodes 50
```

### DQN Baseline
```bash
python -m src.scripts.train_baseline --steps 200000
# Checkpoint → checkpoints/dqn_baseline.pt

python -m src.scripts.plot_baseline
# Output: evaluation/baseline_return_curve.png
```

---

## Running Tests

```bash
# Verify tensor shapes match the interface contract (Integration Checkpoint 2)
python -m pytest tests/test_shapes.py -v

# Verify maze env renders correct observations (Integration Checkpoint 1)
python -m pytest tests/test_env.py -v

# Full pipeline smoke test (E.3 Definition of Done)
python -m pytest tests/test_integration.py -v

# Run all tests at once
python -m pytest tests/ -v
```

---

## Evaluation & Diagnostics

### Check the model is learning real dynamics
```bash
python -m src.scripts.check_deltas
# Compares delta magnitude vs input magnitude.
# delta/input ratio < 0.05 → model is copying input (bad)
# delta cosine > 0.7       → model is learning real dynamics (good)
```

> **Why cosine similarity alone is misleading:** a model that simply copies its input
> will score near 1.0 on cosine similarity without learning anything. Always use
> `check_deltas.py` alongside cosine similarity to confirm the model is genuinely
> predicting state changes.

---

## File Reference

| File | What it does | Used by |
|---|---|---|
| `src/env/maze_env.py` | Custom Gymnasium maze — generates (64,64,3) RGB obs, +1 reward on goal | `collect_data.py`, `train_baseline.py`, `evaluate_transfer.py` |
| `src/models/encoder.py` | Wraps DINOv2 ViT-S/14, converts image → 384-dim latent | `encode_dataset.py`, `evaluate_transfer.py` |
| `src/models/transformer.py` | Causal Transformer: predicts next latent, reward, value. Contains RoPE attention, CEM planner, MPC controller | `train_world_model.py`, `evaluate_world_model.py`, `evaluate_transfer.py`, `check_deltas.py` |
| `src/models/transformer_configuration.py` | Single source of truth for all hyperparameters | all model-related files |
| `src/data/buffer.py` | Stores latent episodes, samples (B, T, 384) batches | `train_world_model.py`, `evaluate_world_model.py`, `check_deltas.py` |
| `src/scripts/collect_data.py` | Runs maze with random actions, saves raw obs to `data/raw/` | Pipeline step 1 |
| `src/scripts/encode_dataset.py` | Runs DINOv2 on raw obs, saves latents to `data/processed/` | Pipeline step 2 |
| `src/scripts/train_world_model.py` | Main training loop: loads buffer, trains transformer, saves checkpoints | Pipeline step 3–4 |
| `src/scripts/train_baseline.py` | Trains DQN agent directly on maze pixels | Baseline comparison |
| `src/scripts/evaluate_world_model.py` | Loads checkpoint, computes MSE/cosine, generates all plots | Pipeline step 5 |
| `src/scripts/evaluate_transfer.py` | Head-to-head: DQN vs CEM agent on fixed maze episodes | Presentation proof |
| `src/scripts/check_deltas.py` | Checks whether model predicts genuine state changes vs copying input | Diagnostic |
| `src/scripts/plot_baseline.py` | Reads DQN training log CSV, generates reward curve plot | Presentation |
| `src/utils/metrics.py` | MSE and cosine similarity functions | `evaluate_world_model.py`, `check_deltas.py` |
| `src/utils/visualizer.py` | All plot generation functions | `evaluate_world_model.py` |
| `tests/test_shapes.py` | Asserts all tensor shapes match the interface contract | CI / Checkpoint 2 |
| `tests/test_env.py` | Asserts maze env returns (64,64,3) obs and steps without crash | CI / Checkpoint 1 |
| `tests/test_integration.py` | Full E.3 pipeline: buffer → model → loss → 1 epoch no crash | CI / E.3 DoD |
