"""
encode_dataset.py  —  Member C: Task C.3
==========================================
Reads raw episodes from data/raw/, encodes each frame with DINOv2,
and saves latent sequences to data/processed/.

Usage:
    python -m src.scripts.encode_dataset --raw_dir data/raw --out_dir data/processed

Output per episode: ep_XXXXXX.npz with keys:
    latents  : (T, 384)  float32
    actions  : (T,)      int64
    rewards  : (T,)      float32
    dones    : (T,)      bool
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.encoder import DinoV2Encoder

# ImageNet normalization constants (same as encoder.py)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def encode_episode_batched(
    backbone,
    obs: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encodes all frames of one episode in batches on GPU.

    Args:
        backbone  : the raw DINOv2 backbone (encoder.backbone)
        obs       : (T, 64, 64, 3) uint8 numpy array
        device    : cuda or cpu
        batch_size: how many frames to encode at once (lower if VRAM is tight)

    Returns:
        latents: (T, 384) float32 numpy array
    """
    mean = IMAGENET_MEAN.to(device)
    std  = IMAGENET_STD.to(device)

    # Convert full episode to tensor in one shot: (T, 64, 64, 3) -> (T, 3, 64, 64)
    x = torch.from_numpy(obs).to(device)          # (T, 64, 64, 3) uint8
    x = x.permute(0, 3, 1, 2).float() / 255.0     # (T, 3, 64, 64) float [0,1]
    x = (x - mean) / std                           # ImageNet normalisation
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    # (T, 3, 224, 224) ready for DINOv2

    T = x.shape[0]
    latents = []

    # Process in mini-batches to avoid OOM on long episodes
    with torch.no_grad():
        for start in range(0, T, batch_size):
            end   = min(start + batch_size, T)
            batch = x[start:end]                   # (B, 3, 224, 224)
            z     = backbone(batch)                # (B, 384)
            latents.append(z.cpu())                # move back to CPU for numpy

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
                        help="Frames per GPU batch (reduce if VRAM error)")
    args = parser.parse_args()

    # Device setup
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
    print(f"[encode_dataset] Found    : {len(files)} raw episodes")

    if len(files) == 0:
        print("[encode_dataset] Nothing to encode. Did you run collect_data.py first?")
        return

    if args.max_eps:
        files = files[:args.max_eps]
        print(f"[encode_dataset] Limiting : {len(files)} episodes (--max_eps)")

    # Load encoder and extract just the backbone (we handle pre-processing ourselves)
    print("[encode_dataset] Loading DINOv2 backbone...")
    encoder  = DinoV2Encoder(device=device)
    backbone = encoder.backbone   # already on device, already in eval mode

    all_latents = []   # for variance check at the end

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
            latents=latents,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

        if (i + 1) % 50 == 0:
            print(f"  Encoded {i+1}/{len(files)} episodes")

    # C.3 critical check: latent variance must be > 0.01
    all_latents_np = np.concatenate(all_latents, axis=0)   # (N*T, 384)
    variance = float(np.var(all_latents_np))
    print(f"\n[encode_dataset] DONE")
    print(f"  Total frames encoded : {all_latents_np.shape[0]}")
    print(f"  Latent variance      : {variance:.6f}  "
          f"({'✅ PASS' if variance > 0.01 else '❌ FAIL — check encoder'})")


if __name__ == "__main__":
    main()