"""
done_head_diagnostic.py  —  Done head output distribution analysis
===================================================================
Samples real transitions from the buffer, runs them through the WM,
and reports done_logit distributions for goal vs non-goal steps.

Usage:
    python -m src.scripts.done_head_diagnostic \
        --checkpoint checkpoints/world_model_best.pt \
        --data_dir data/processed
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import torch

from src.models.transformer import DinoWorldModel
from src.models.transformer_configuration import TransformerWMConfiguration as Config
from src.data.buffer import LatentReplayBuffer


def load_model(checkpoint_path: Path, device: torch.device) -> DinoWorldModel:
    config = Config()
    model  = DinoWorldModel(config).to(device)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_buffer(data_dir: Path) -> LatentReplayBuffer:
    buf   = LatentReplayBuffer(capacity_steps=200_000)
    files = sorted(data_dir.glob("*.npz"))
    for f in files:
        d = np.load(f)
        missing = [k for k in ("latents","actions","rewards","dones") if k not in d]
        if missing:
            continue
        buf.add_episode(
            latents=d["latents"], actions=d["actions"],
            rewards=d["rewards"], dones=d["dones"],
        )
        d.close()
    print(f"[INFO] Buffer: {len(buf.episodes)} episodes, {buf.total_steps} steps")
    return buf


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/world_model_best.pt")
    parser.add_argument("--data_dir",   type=str, default="data/processed")
    parser.add_argument("--n_samples",  type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(Path(args.checkpoint), device)
    buffer = load_buffer(Path(args.data_dir))

    goal_logits    = []
    nongoal_logits = []
    sampled        = 0

    for ep in buffer.episodes:
        T = ep.latents.shape[0]
        if T < 2:
            continue

        for t in range(T - 1):
            z_in = torch.tensor(
                ep.latents[t], dtype=torch.float32, device=device
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, 384)

            a_in = torch.tensor(
                [[int(ep.actions[t])]], dtype=torch.long, device=device
            )

            _, _, pred_done = model(z_in, a_in)
            logit = float(pred_done[0, 0, 0].item())

            is_goal = bool(ep.rewards[t] > 0.5)
            if is_goal:
                goal_logits.append(logit)
            else:
                nongoal_logits.append(logit)

            sampled += 1
            if sampled >= args.n_samples:
                break
        if sampled >= args.n_samples:
            break

    goal_logits    = np.array(goal_logits)
    nongoal_logits = np.array(nongoal_logits)

    print(f"\n{'='*58}")
    print(f"  DONE HEAD DIAGNOSTIC REPORT")
    print(f"{'='*58}")
    print(f"\n  NON-GOAL steps ({len(nongoal_logits)} samples):")
    print(f"    mean logit : {nongoal_logits.mean():.4f}")
    print(f"    std  logit : {nongoal_logits.std():.4f}")
    print(f"    min  logit : {nongoal_logits.min():.4f}")
    print(f"    max  logit : {nongoal_logits.max():.4f}")
    print(f"    % firing at threshold 0.0 : "
          f"{100*(nongoal_logits > 0.0).mean():.1f}%")
    print(f"    % firing at threshold 2.0 : "
          f"{100*(nongoal_logits > 2.0).mean():.1f}%")
    print(f"    % firing at threshold 3.0 : "
          f"{100*(nongoal_logits > 3.0).mean():.1f}%")

    print(f"\n  GOAL steps ({len(goal_logits)} samples):")
    if len(goal_logits) > 0:
        print(f"    mean logit : {goal_logits.mean():.4f}")
        print(f"    std  logit : {goal_logits.std():.4f}")
        print(f"    min  logit : {goal_logits.min():.4f}")
        print(f"    max  logit : {goal_logits.max():.4f}")
        print(f"    % firing at threshold 0.0 : "
              f"{100*(goal_logits > 0.0).mean():.1f}%")
        print(f"    % firing at threshold 2.0 : "
              f"{100*(goal_logits > 2.0).mean():.1f}%")
        print(f"    % firing at threshold 3.0 : "
              f"{100*(goal_logits > 3.0).mean():.1f}%")
    else:
        print("    No goal steps found in sampled transitions.")

    if len(goal_logits) > 0:
        separation = goal_logits.mean() - nongoal_logits.mean()
        print(f"\n  Mean separation (goal - nongoal): {separation:.4f}")
        if separation > 2.0:
            midpoint = (goal_logits.mean() + nongoal_logits.mean()) / 2
            print(f"  ✅ Clear separation — recommended threshold: {midpoint:.2f}")
        elif separation > 0.5:
            print(f"  ⚠️  Partial separation — use high threshold (3.0+) "
                  f"or rely on reward head only")
        else:
            print(f"  ❌ No separation — done head is not reliable. "
                  f"Disable it, rely on reward head only.")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()