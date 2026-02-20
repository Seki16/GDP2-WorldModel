"""
check_deltas.py — Diagnostic for delta prediction
===================================================
Checks if the model is learning meaningful state changes (deltas)
or just copying the input.

Usage:
    python -m src.scripts.check_deltas
"""

import torch
import numpy as np
from pathlib import Path

from src.models.transformer import DinoWorldModel
from src.models.transformer_configuration import TransformerWMConfiguration as Config
from src.data.buffer import LatentReplayBuffer


def load_buffer(data_dir: str):
    """Load real latent data from processed episodes"""
    buf = LatentReplayBuffer(capacity_steps=200_000)
    files = sorted(Path(data_dir).glob("*.npz"))
    for f in files[:100]:  # Load 100 episodes
        d = np.load(f)
        if "latents" in d:
            buf.add_episode(d["latents"])
    return buf


def analyze_model(model, name: str, latents, actions):
    """Analyze delta magnitudes for a given model"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    with torch.no_grad():
        z_in = latents[:, :-1]       # (B, T-1, 384)
        a_in = actions[:, :-1]       # (B, T-1)
        z_target = latents[:, 1:]    # (B, T-1, 384)
        
        # Model prediction
        pred_next, _, _ = model(z_in, a_in)
        
        # Calculate deltas
        pred_delta = pred_next - z_in       # Model's predicted change
        true_delta = z_target - z_in        # Actual change in data
        
        # Magnitudes
        input_mag = z_in.abs().mean().item()
        pred_delta_mag = pred_delta.abs().mean().item()
        true_delta_mag = true_delta.abs().mean().item()
        output_mag = pred_next.abs().mean().item()
        
        # Delta accuracy
        delta_mse = torch.nn.functional.mse_loss(pred_delta, true_delta).item()
        delta_cos = torch.nn.functional.cosine_similarity(
            pred_delta.reshape(-1, 384),
            true_delta.reshape(-1, 384),
            dim=-1
        ).mean().item()
        
        # Output accuracy (your original metric)
        output_cos = torch.nn.functional.cosine_similarity(
            pred_next.reshape(-1, 384),
            z_target.reshape(-1, 384),
            dim=-1
        ).mean().item()
        
        # Input-target similarity (baseline)
        baseline_cos = torch.nn.functional.cosine_similarity(
            z_in.reshape(-1, 384),
            z_target.reshape(-1, 384),
            dim=-1
        ).mean().item()
    
    print(f"\n[Magnitudes]")
    print(f"  Input latents         : {input_mag:.4f}")
    print(f"  True delta (data)     : {true_delta_mag:.4f}")
    print(f"  Predicted delta       : {pred_delta_mag:.4f}")
    print(f"  Output latents        : {output_mag:.4f}")
    print(f"  Delta/Input ratio     : {pred_delta_mag/input_mag:.4f}")
    
    print(f"\n[Delta Prediction Quality]")
    print(f"  Delta MSE             : {delta_mse:.6f}")
    print(f"  Delta cosine sim      : {delta_cos:.4f}")
    
    print(f"\n[Output Quality]")
    print(f"  Output cosine sim     : {output_cos:.4f}")
    print(f"  Baseline (input→target): {baseline_cos:.4f}")
    print(f"  Improvement over baseline: {(output_cos - baseline_cos)*100:.2f}%")
    
    print(f"\n[Interpretation]")
    if pred_delta_mag / input_mag < 0.05:
        print("  ⚠️  Delta is very small — model may be copying input")
    elif pred_delta_mag / input_mag > 0.5:
        print("  ⚠️  Delta is very large — residual connection not working")
    else:
        print("  ✅  Delta magnitude looks reasonable")
    
    if delta_cos < 0.3:
        print("  ⚠️  Delta prediction is poor — model not learning changes")
    elif delta_cos < 0.7:
        print("  ⚠️  Delta prediction is mediocre")
    else:
        print("  ✅  Delta prediction is good")
    
    if output_cos - baseline_cos < 0.01:
        print("  ⚠️  Barely better than baseline — task may be too easy")
    else:
        print(f"  ✅  Meaningful improvement over baseline")
    
    return {
        "delta_cos": delta_cos,
        "output_cos": output_cos,
        "baseline_cos": baseline_cos,
        "delta_mag": pred_delta_mag,
        "input_mag": input_mag
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    
    print("="*60)
    print("  Delta Prediction Analysis")
    print("="*60)
    print(f"Device: {device}")
    
    # Load real data
    print("\nLoading real latent data...")
    buffer = load_buffer("data/processed")
    latents = buffer.sample(32, seq_len=16).to(device)
    actions = torch.randint(0, 4, (32, 16), device=device)
    
    # Test 1: Random untrained model
    print("\n" + "="*60)
    print("TEST 1: Random Untrained Model")
    print("="*60)
    random_model = DinoWorldModel(config).to(device)
    random_model.eval()
    random_results = analyze_model(random_model, "Random Model", latents, actions)
    
    # Test 2: Trained model
    print("\n" + "="*60)
    print("TEST 2: Your Trained Model")
    print("="*60)
    trained_model = DinoWorldModel(config).to(device)
    
    try:
        ckpt = torch.load("checkpoints/world_model_best.pt", map_location=device)
        trained_model.load_state_dict(ckpt["model_state"])
        trained_model.eval()
        trained_results = analyze_model(trained_model, "Trained Model", latents, actions)
        
        # Comparison
        print("\n" + "="*60)
        print("  COMPARISON: Trained vs Random")
        print("="*60)
        print(f"\nDelta Cosine Similarity:")
        print(f"  Random  : {random_results['delta_cos']:.4f}")
        print(f"  Trained : {trained_results['delta_cos']:.4f}")
        print(f"  Improvement: {(trained_results['delta_cos'] - random_results['delta_cos'])*100:.1f}%")
        
        print(f"\nOutput Cosine Similarity:")
        print(f"  Baseline (input≈target): {trained_results['baseline_cos']:.4f}")
        print(f"  Random model           : {random_results['output_cos']:.4f}")
        print(f"  Trained model          : {trained_results['output_cos']:.4f}")
        
        print(f"\n[Final Verdict]")
        if trained_results['delta_cos'] > 0.7 and trained_results['delta_cos'] > random_results['delta_cos'] + 0.2:
            print("  ✅  Model learned meaningful dynamics")
        elif trained_results['output_cos'] - trained_results['baseline_cos'] > 0.02:
            print("  ✅  Model provides meaningful improvement over baseline")
        else:
            print("  ⚠️  Model improvement is marginal — task may be too easy")
            print("      or evaluation metric is saturated")
        
    except FileNotFoundError:
        print("\n⚠️  Trained model checkpoint not found at checkpoints/world_model_best.pt")
        print("Run training first: python -m src.scripts.train_world_model --smoke_test")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()