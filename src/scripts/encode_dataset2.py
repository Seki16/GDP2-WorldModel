"""
encode_dataset.py  —  Member C: Task C.3
==========================================
Reads raw episodes from data/raw/, encodes each frame with DINOv2,
and saves latent sequences to data/processed/.

Usage (standard — frozen DINOv2):
    python -m src.scripts.encode_dataset --raw_dir data/raw --out_dir data/processed

Usage (after joint training — fine-tuned encoder):
    python -m src.scripts.encode_dataset \\
        --raw_dir data/raw \\
        --out_dir data/processed \\
        --encoder_checkpoint checkpoints/encoder_finetuned.pt

Output per episode: ep_XXXXXX.npz with keys:
    latents  : (T, 384)  float32
    actions  : (T,)      int64
    rewards  : (T,)      float32
    dones    : (T,)      bool

─────────────────────────────────────────────────────────────
MODIFICATION — Member B (Fine-tuned Encoder Support)
─────────────────────────────────────────────────────────────
Added --encoder_checkpoint flag to load the fine-tuned DINOv2
encoder produced by train_world_model_joint.py.

After joint training, the latents in data/processed/ were
produced by the frozen DINOv2 and are no longer compatible
with the fine-tuned world model. Re-encoding with the
fine-tuned encoder produces latents in the reshaped latent
space that the jointly trained world model expects.

If --encoder_checkpoint is not provided, behaviour is
identical to the original script (frozen DINOv2).
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import DinoV2Encoder

# ImageNet normalization constants (same as encoder.py)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_backbone(
    device:               torch.device,
    encoder_checkpoint:   str | None = None,
) -> nn.Module:
    """
    Load DINOv2 backbone — either frozen (standard) or fine-tuned (joint).

    Args:
        device             : torch device
        encoder_checkpoint : path to encoder_finetuned.pt from joint training.
                             If None, loads the standard frozen DINOv2.

    Returns:
        backbone : DINOv2 backbone in eval mode on device
    """
    if encoder_checkpoint is None:
        # Standard frozen DINOv2 — original behaviour
        print("[encode_dataset] Loading frozen DINOv2 backbone...")
        encoder  = DinoV2Encoder(device=device)
        backbone = encoder.backbone
        print("[encode_dataset] Using frozen DINOv2 (standard pipeline)")
        return backbone

    # Fine-tuned encoder from train_world_model_joint.py
    ckpt_path = Path(encoder_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {ckpt_path}\n"
            f"Run train_world_model_joint.py first to produce it."
        )

    print(f"[encode_dataset] Loading fine-tuned encoder from {ckpt_path}...")

    # Load the FineTunableDinoV2 state dict
    # We only need the backbone for encoding — reconstruct it via torch hub
    # then load the fine-tuned weights on top
    print("[encode_dataset] Loading DINOv2 base architecture...")
    encoder  = DinoV2Encoder(device=device)
    backbone = encoder.backbone

    ckpt = torch.load(ckpt_path, map_location=device)

    # encoder_finetuned.pt stores the full FineTunableDinoV2 state dict
    # which includes backbone.* keys and mean/std buffers
    # Extract only backbone weights
    encoder_state = ckpt["encoder_state"]
    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in encoder_state.items()
        if k.startswith("backbone.")
    }

    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys in backbone: {len(missing)} "
              f"(frozen layers not saved — this is expected)")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    epoch = ckpt.get("epoch", "?")
    print(f"[encode_dataset] Fine-tuned encoder loaded (epoch={epoch})")
    print(f"[encode_dataset] ✅ Latents will be in fine-tuned latent space")

    backbone.eval().to(device)
    return backbone


def encode_episode_batched(
    backbone,
    obs: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encodes all frames of one episode in batches.

    Args:
        backbone  : DINOv2 backbone (frozen or fine-tuned)
        obs       : (T, 64, 64, 3) uint8 numpy array
        device    : cuda or cpu
        batch_size: frames per batch (reduce if OOM)

    Returns:
        latents: (T, 384) float32 numpy array
    """
    mean = IMAGENET_MEAN.to(device)
    std  = IMAGENET_STD.to(device)

    # Convert full episode to tensor: (T, 64, 64, 3) -> (T, 3, 224, 224)
    x = torch.from_numpy(obs).to(device)           # (T, 64, 64, 3) uint8
    x = x.permute(0, 3, 1, 2).float() / 255.0      # (T, 3, 64, 64)
    x = (x - mean) / std
    x = F.interpolate(x, size=(224, 224),
                      mode="bilinear", align_corners=False)

    T       = x.shape[0]
    latents = []

    with torch.no_grad():
        for start in range(0, T, batch_size):
            end   = min(start + batch_size, T)
            batch = x[start:end]                   # (B, 3, 224, 224)
            z     = backbone(batch)                # (B, 384)
            latents.append(z.cpu())

    latents_tensor = torch.cat(latents, dim=0)     # (T, 384)
    return latents_tensor.numpy().astype(np.float32)


def main():
    print("[encode_dataset] Starting...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",    type=str, default="data/raw")
    parser.add_argument("--out_dir",    type=str, default="data/processed")
    parser.add_argument("--max_eps",    type=int, default=None,
                        help="Limit number of episodes (for testing)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Frames per batch (reduce if OOM)")
    parser.add_argument(
        "--encoder_checkpoint", type=str, default=None,
        help=(
            "Path to fine-tuned encoder checkpoint from train_world_model_joint.py "
            "(checkpoints/encoder_finetuned.pt). "
            "If not provided, uses standard frozen DINOv2."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[encode_dataset] Device   : {device}")
    if device.type == "cuda":
        print(f"[encode_dataset] GPU      : {torch.cuda.get_device_name(0)}")
        print(f"[encode_dataset] VRAM     : "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("ep_*.npz"))
    print(f"[encode_dataset] raw_dir  : {raw_dir.resolve()}")
    print(f"[encode_dataset] out_dir  : {out_dir.resolve()}")
    print(f"[encode_dataset] Found    : {len(files)} raw episodes")

    if len(files) == 0:
        print("[encode_dataset] Nothing to encode. Did you run collect_data_dqn.py first?")
        return

    if args.max_eps:
        files = files[:args.max_eps]
        print(f"[encode_dataset] Limiting : {len(files)} episodes (--max_eps)")

    # Load backbone — frozen or fine-tuned
    backbone = load_backbone(device, args.encoder_checkpoint)

    all_latents = []

    for i, fpath in enumerate(files):
        data    = np.load(fpath)
        obs     = data["obs"]        # (T, 64, 64, 3) uint8
        actions = data["actions"]
        rewards = data["rewards"]
        dones   = data["dones"]

        latents = encode_episode_batched(backbone, obs, device, args.batch_size)
        all_latents.append(latents)

        np.savez_compressed(
            out_dir / fpath.name,
            latents = latents,
            actions = actions,
            rewards = rewards,
            dones   = dones,
        )

        if (i + 1) % 50 == 0:
            print(f"  Encoded {i+1}/{len(files)} episodes")

    # Variance check
    all_latents_np = np.concatenate(all_latents, axis=0)   # (N*T, 384)
    variance = float(np.var(all_latents_np))
    encoder_type = "fine-tuned" if args.encoder_checkpoint else "frozen"
    print(f"\n[encode_dataset] DONE")
    print(f"  Encoder type         : {encoder_type}")
    print(f"  Total frames encoded : {all_latents_np.shape[0]}")
    print(f"  Latent variance      : {variance:.6f}  "
          f"({'✅ PASS' if variance > 0.01 else '❌ FAIL — check encoder'})")

    if args.encoder_checkpoint:
        print(f"\n  ✅ data/processed/ now contains fine-tuned latents.")
        print(f"     Use checkpoints/world_model_joint_best.pt for WM training.")
        print(f"     Do NOT mix fine-tuned latents with standard WM checkpoints.")


if __name__ == "__main__":
    main()