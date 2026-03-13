"""
src/tuning/aggregate_results.py
=================================
Member E — Sweep Results Aggregator

Queries the MLflow tracking server for all runs in the sweep experiment
and prints a ranked summary table sorted by DDA (primary metric).

Use this after running the sweep to:
  - Identify the best hyperparameter configuration
  - Share a clean table with Member B for review
  - Extract the optimal config to update transformer_configuration.py

═══════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════

After running the sweep:
    python -m src.tuning.aggregate_results

To also print the recommended config update for transformer_configuration.py:
    python -m src.tuning.aggregate_results --show-best-config

To export results to CSV:
    python -m src.tuning.aggregate_results --csv evaluation/sweep_results.csv

View live in browser (run in separate terminal):
    mlflow ui  →  http://localhost:5000
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mlflow


EXPERIMENT_NAME = "gdp2_wm_hp_sweep"

PARAM_COLS = [
    "num_heads", "num_layers", "mlp_ratio",
    "learning_rate", "batch_size", "sequence_length", "head_dim",
]

METRIC_COLS = [
    "final/DDA", "final/latent_mse", "final/cosine_sim",
    "final/best_dda", "final/training_time_s",
]


def fetch_runs() -> list[dict]:
    """Fetch all completed runs from MLflow and return as list of dicts."""
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    except Exception:
        experiment = None

    if experiment is None:
        print(f"\n[ERROR] MLflow experiment '{EXPERIMENT_NAME}' not found.")
        print("Have you run the sweep yet?")
        print("  python -m src.tuning.train_sweep --multirun --config-name sweep/optuna")
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.`final/DDA` DESC"],
    )

    results = []
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        row = {"run_id": run.info.run_id[:8]}
        for p in PARAM_COLS:
            row[p] = run.data.params.get(p, "—")
        for m in METRIC_COLS:
            val = run.data.metrics.get(m)
            row[m] = f"{val:.4f}" if val is not None else "—"
        results.append(row)

    return results


def print_table(results: list[dict]):
    """Print ranked results table to terminal."""
    if not results:
        return

    print(f"\n{'='*90}")
    print(f"  SWEEP RESULTS — {EXPERIMENT_NAME}  ({len(results)} runs, ranked by DDA ↑)")
    print(f"{'='*90}")

    header = (
        f"  {'#':<4} {'Heads':<6} {'Layers':<7} {'MLP':<5} {'LR':<9} "
        f"{'BS':<5} {'SeqLen':<7} {'head_dim':<9} "
        f"{'DDA ↑':<8} {'MSE ↓':<9} {'CosSim':<8} {'Time':>7}"
    )
    print(header)
    print(f"  {'-'*86}")

    for i, r in enumerate(results, 1):
        marker = "  ◄ BEST" if i == 1 else ""
        print(
            f"  {i:<4} "
            f"{r['num_heads']:<6} "
            f"{r['num_layers']:<7} "
            f"{r['mlp_ratio']:<5} "
            f"{r['learning_rate']:<9} "
            f"{r['batch_size']:<5} "
            f"{r['sequence_length']:<7} "
            f"{r['head_dim']:<9} "
            f"{r['final/DDA']:<8} "
            f"{r['final/latent_mse']:<9} "
            f"{r['final/cosine_sim']:<8} "
            f"{r['final/training_time_s']:>6}s"
            f"{marker}"
        )

    print(f"{'='*90}")


def print_best_config(results: list[dict]):
    """Print the recommended update for transformer_configuration.py."""
    if not results:
        return

    best = results[0]
    print(f"\n{'='*60}")
    print(f"  RECOMMENDED CONFIG UPDATE")
    print(f"  Update src/models/transformer_configuration.py with:")
    print(f"{'='*60}")
    print(f"""
class TransformerWMConfiguration:
    LATENT_DIM      = 384    # fixed
    ACTION_DIM      = 4      # fixed
    SEQUENCE_LENGTH = {best['sequence_length']}
    NUM_LAYERS      = {best['num_layers']}
    NUM_HEADS       = {best['num_heads']}     # head_dim = 384/{best['num_heads']} = {best['head_dim']}
    MLP_RATIO       = {best['mlp_ratio']}
    LEARNING_RATE   = {best['learning_rate']}
""")
    print(f"  Final DDA with this config: {best['final/DDA']}")
    print(f"{'='*60}\n")


def export_csv(results: list[dict], path: str):
    """Export results to CSV."""
    if not results:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults exported to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate GDP2 WM sweep results from MLflow")
    parser.add_argument("--show-best-config", action="store_true",
                        help="Print recommended transformer_configuration.py update")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export results to CSV at this path")
    args = parser.parse_args()

    results = fetch_runs()
    print_table(results)

    if args.show_best_config:
        print_best_config(results)

    if args.csv:
        export_csv(results, args.csv)

    if results:
        print(f"\nView full results with curves in browser:")
        print(f"  mlflow ui  →  http://localhost:5000\n")