"""
R4-3: Fold Stability Analysis

Runs the full pipeline with multiple random fold assignments to demonstrate
that results are stable and not artifacts of the specific fold 1-8/9/10 split.

Strategy:
- The standard PTB-XL protocol (folds 1-8 train, 9 val, 10 test) is kept as
  the primary configuration, consistent with the PTB-XL benchmark literature.
- We additionally run N_CONFIGS alternative random fold assignments and report
  mean ± std across all configurations to demonstrate robustness.

Usage:
    python -m experiments.run_fold_stability --base_path <PTB-XL-DIR> [--n_configs 5]
"""

import argparse
import json
import os
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

from experiments.experiment_runner import run_experiment, save_results, FoldConfig
from experiments.architectures import create_lightweight_cnn


def generate_fold_configs(n_configs: int, seed: int = 2026) -> list:
    """
    Generate random fold assignments.

    Each config randomly picks 1 fold for test, 1 for val, rest for training.
    PTB-XL has 10 folds (1-10).
    """
    rng = np.random.default_rng(seed)
    all_folds = list(range(1, 11))
    configs = []

    # Always include the standard protocol as config 0
    configs.append(FoldConfig(
        train_folds=list(range(1, 9)),
        val_fold=9,
        test_fold=10,
    ))

    # Generate unique random assignments
    used = {(9, 10)}  # (val, test) already used
    attempts = 0
    while len(configs) < n_configs + 1 and attempts < 200:
        shuffled = rng.permutation(all_folds).tolist()
        test_fold = shuffled[0]
        val_fold = shuffled[1]
        train_folds = sorted(shuffled[2:])

        key = (val_fold, test_fold)
        if key not in used:
            used.add(key)
            configs.append(FoldConfig(
                train_folds=train_folds,
                val_fold=val_fold,
                test_fold=test_fold,
            ))
        attempts += 1

    return configs


def summarize_results(all_results: list) -> dict:
    """Compute mean ± std across fold configurations."""
    keys_clf = ['roc_auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'sensitivity_at_95spec']
    keys_cp = ['miscoverage_overall', 'miscoverage_normal', 'miscoverage_mi',
               'avg_set_size_overall', 'singleton_rate_overall']

    summary = {'n_configs': len(all_results)}

    for key in keys_clf:
        vals = [r['classification'][key] for r in all_results if not np.isnan(r['classification'].get(key, np.nan))]
        summary[f'clf_{key}_mean'] = float(np.mean(vals))
        summary[f'clf_{key}_std'] = float(np.std(vals))
        summary[f'clf_{key}_min'] = float(np.min(vals))
        summary[f'clf_{key}_max'] = float(np.max(vals))

    for key in keys_cp:
        vals = [r['conformal'][key] for r in all_results if not np.isnan(r['conformal'].get(key, np.nan))]
        summary[f'cp_{key}_mean'] = float(np.mean(vals))
        summary[f'cp_{key}_std'] = float(np.std(vals))
        summary[f'cp_{key}_min'] = float(np.min(vals))
        summary[f'cp_{key}_max'] = float(np.max(vals))

    return summary


def format_table(all_results: list, summary: dict) -> str:
    """Format results as a clean text table for the paper."""
    lines = []
    lines.append("=" * 90)
    lines.append("FOLD STABILITY ANALYSIS — Lightweight CNN")
    lines.append("=" * 90)
    lines.append("")

    # Per-config table
    header = f"{'Config':<12} {'Train Folds':<20} {'Val':>4} {'Test':>4} " \
             f"{'ROC-AUC':>9} {'PR-AUC':>8} {'Bal.Acc':>8} " \
             f"{'Miscov':>8} {'SetSize':>8} {'Sing%':>8}"
    lines.append(header)
    lines.append("-" * 90)

    for i, r in enumerate(all_results):
        fc = r['fold_config']
        tag = "Standard" if i == 0 else f"Random-{i}"
        train_str = ','.join(map(str, fc['train_folds']))
        clf = r['classification']
        cp = r['conformal']
        lines.append(
            f"{tag:<12} {train_str:<20} {fc['val_fold']:>4} {fc['test_fold']:>4} "
            f"{clf['roc_auc']:>9.4f} {clf['pr_auc']:>8.4f} {clf['balanced_accuracy']:>8.4f} "
            f"{cp['miscoverage_overall']:>8.4f} {cp['avg_set_size_overall']:>8.4f} {cp['singleton_rate_overall']:>8.4f}"
        )

    lines.append("-" * 90)
    lines.append(
        f"{'Mean±Std':<12} {'':<20} {'':>4} {'':>4} "
        f"{summary['clf_roc_auc_mean']:>5.4f}±{summary['clf_roc_auc_std']:.4f} "
        f"{summary['clf_pr_auc_mean']:>4.4f}±{summary['clf_pr_auc_std']:.4f} "
        f"{summary['clf_balanced_accuracy_mean']:>4.4f}±{summary['clf_balanced_accuracy_std']:.4f} "
        f"{summary['cp_miscoverage_overall_mean']:>4.4f}±{summary['cp_miscoverage_overall_std']:.4f} "
        f"{summary['cp_avg_set_size_overall_mean']:>4.4f}±{summary['cp_avg_set_size_overall_std']:.4f} "
        f"{summary['cp_singleton_rate_overall_mean']:>4.4f}±{summary['cp_singleton_rate_overall_std']:.4f}"
    )
    lines.append("=" * 90)

    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fold Stability Analysis (R4-3)')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Path to PTB-XL dataset root')
    parser.add_argument('--n_configs', type=int, default=4,
                        help='Number of ADDITIONAL random fold configs (beyond standard)')
    parser.add_argument('--output_dir', type=str, default='experiments/results/fold_stability',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Seed for fold config generation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate fold configs
    configs = generate_fold_configs(args.n_configs, seed=args.seed)
    print(f"Running {len(configs)} fold configurations (1 standard + {args.n_configs} random)")

    all_results = []
    for i, config in enumerate(configs):
        tag = "standard" if i == 0 else f"random_{i}"
        print(f"\n{'='*60}")
        print(f"  Config {i+1}/{len(configs)}: {tag}")
        print(f"  Train: {config.train_folds} | Val: {config.val_fold} | Test: {config.test_fold}")
        print(f"{'='*60}")

        result = run_experiment(
            base_path=args.base_path,
            fold_config=config,
            model_builder=create_lightweight_cnn,
            model_name='LightweightCNN',
            seed=42,  # same training seed across configs
            verbose=True
        )
        all_results.append(result)

        # Save individual result
        path = save_results(result, args.output_dir, tag=tag)
        print(f"  Saved: {path}")

    # Summary statistics
    summary = summarize_results(all_results)

    # Save summary
    with open(os.path.join(args.output_dir, 'fold_stability_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save formatted table
    table = format_table(all_results, summary)
    with open(os.path.join(args.output_dir, 'fold_stability_table.txt'), 'w') as f:
        f.write(table)
    print(f"\n{table}")

    # Save all results as single CSV for easy plotting
    rows = []
    for i, r in enumerate(all_results):
        row = {
            'config': 'standard' if i == 0 else f'random_{i}',
            'val_fold': r['fold_config']['val_fold'],
            'test_fold': r['fold_config']['test_fold'],
        }
        row.update({f'clf_{k}': v for k, v in r['classification'].items()})
        row.update({f'cp_{k}': v for k, v in r['conformal'].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'fold_stability_all.csv'), index=False)

    print(f"\nAll results saved to {args.output_dir}/")
