"""
R2-2: Architecture Comparison

Evaluates multiple lightweight 1D CNN variants suitable for edge deployment,
all using the standard PTB-XL fold protocol (1-8/9/10) with Mondrian CP.

Usage:
    python -m experiments.run_architecture_comparison --base_path <PTB-XL-DIR>
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

from experiments.experiment_runner import run_experiment, save_results, FoldConfig
from experiments.architectures import MODEL_REGISTRY


STANDARD_FOLDS = FoldConfig(
    train_folds=list(range(1, 9)),
    val_fold=9,
    test_fold=10,
)


def format_comparison_table(all_results: list) -> str:
    """Format architecture comparison as a text table for the paper."""
    lines = []
    lines.append("=" * 110)
    lines.append("ARCHITECTURE COMPARISON — Standard PTB-XL Protocol (Folds 1-8/9/10)")
    lines.append("=" * 110)
    lines.append("")

    header = (f"{'Model':<22} {'Params':>8} {'ROC-AUC':>18} {'PR-AUC':>8} {'Bal.Acc':>8} "
              f"{'Miscov':>8} {'SetSize':>8} {'Sing%':>8}")
    lines.append(header)
    lines.append("-" * 110)

    for r in all_results:
        clf = r['classification']
        cp = r['conformal']
        auc_str = f"{clf['roc_auc']:.3f} ({clf['roc_auc_ci_lo']:.3f}-{clf['roc_auc_ci_hi']:.3f})"
        lines.append(
            f"{r['model_name']:<22} {r['n_params']:>8,} {auc_str:>18} "
            f"{clf['pr_auc']:>8.3f} {clf['balanced_accuracy']:>8.3f} "
            f"{cp['miscoverage_overall']:>8.4f} {cp['avg_set_size_overall']:>8.4f} {cp['singleton_rate_overall']:>8.4f}"
        )

    lines.append("=" * 110)
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Architecture Comparison (R2-2)')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Path to PTB-XL dataset root')
    parser.add_argument('--models', nargs='+',
                        default=['LightweightCNN', 'ResNet1D', 'DSC_CNN', 'TCN', 'MiniResNet'],
                        help='Models to evaluate (keys from MODEL_REGISTRY)')
    parser.add_argument('--output_dir', type=str, default='experiments/results/architecture_comparison',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            print(f"WARNING: {model_name} not in MODEL_REGISTRY, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        result = run_experiment(
            base_path=args.base_path,
            fold_config=STANDARD_FOLDS,
            model_builder=MODEL_REGISTRY[model_name],
            model_name=model_name,
            seed=42,
            verbose=True
        )
        all_results.append(result)
        path = save_results(result, args.output_dir)
        print(f"  Saved: {path}")

    # Save comparison table
    table = format_comparison_table(all_results)
    with open(os.path.join(args.output_dir, 'architecture_comparison_table.txt'), 'w') as f:
        f.write(table)
    print(f"\n{table}")

    # Save as CSV
    rows = []
    for r in all_results:
        row = {'model': r['model_name'], 'n_params': r['n_params']}
        row.update({f'clf_{k}': v for k, v in r['classification'].items()})
        row.update({f'cp_{k}': v for k, v in r['conformal'].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'architecture_comparison.csv'), index=False)

    print(f"\nAll results saved to {args.output_dir}/")
