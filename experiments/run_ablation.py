"""
R2-3: Architectural Sensitivity (Ablation Study)

Varies kernel size and number of convolutional blocks in the Lightweight CNN
to show how architectural choices affect conformal prediction metrics.

Usage:
    python -m experiments.run_ablation --base_path <PTB-XL-DIR>
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from functools import partial

from experiments.experiment_runner import run_experiment, save_results, FoldConfig
from experiments.architectures import create_lightweight_cnn_ablation


STANDARD_FOLDS = FoldConfig(
    train_folds=list(range(1, 9)),
    val_fold=9,
    test_fold=10,
)

# Ablation grid
ABLATION_CONFIGS = [
    # Vary kernel size (keep 2 blocks, 32/64 filters)
    {'kernel_size': 3, 'n_filters_1': 32, 'n_filters_2': 64, 'n_blocks': 2, 'fc_units': 64},
    {'kernel_size': 5, 'n_filters_1': 32, 'n_filters_2': 64, 'n_blocks': 2, 'fc_units': 64},  # baseline
    {'kernel_size': 7, 'n_filters_1': 32, 'n_filters_2': 64, 'n_blocks': 2, 'fc_units': 64},
    {'kernel_size': 9, 'n_filters_1': 32, 'n_filters_2': 64, 'n_blocks': 2, 'fc_units': 64},

    # Vary depth (keep kernel=5, 32/64 filters)
    {'kernel_size': 5, 'n_filters_1': 32, 'n_filters_2': 64, 'n_blocks': 1, 'fc_units': 64},
    # n_blocks=2 already covered above
    {'kernel_size': 5, 'n_filters_1': 32, 'n_filters_2': 64, 'n_blocks': 3, 'fc_units': 64},

    # Vary filter width (keep kernel=5, 2 blocks)
    {'kernel_size': 5, 'n_filters_1': 16, 'n_filters_2': 32, 'n_blocks': 2, 'fc_units': 64},
    # 32/64 already covered above
    {'kernel_size': 5, 'n_filters_1': 64, 'n_filters_2': 128, 'n_blocks': 2, 'fc_units': 64},
]


def config_to_name(cfg: dict) -> str:
    return f"k{cfg['kernel_size']}_b{cfg['n_blocks']}_f{cfg['n_filters_1']}_{cfg['n_filters_2']}"


def format_ablation_table(all_results: list, all_configs: list) -> str:
    lines = []
    lines.append("=" * 120)
    lines.append("ABLATION STUDY — Lightweight CNN Architectural Sensitivity")
    lines.append("=" * 120)
    lines.append("")

    header = (f"{'Variant':<25} {'Kernel':>6} {'Blocks':>6} {'Filters':>10} {'Params':>8} "
              f"{'ROC-AUC':>9} {'PR-AUC':>8} {'Miscov':>8} {'SetSize':>8} {'Sing%':>8}")
    lines.append(header)
    lines.append("-" * 120)

    for cfg, r in zip(all_configs, all_results):
        clf = r['classification']
        cp = r['conformal']
        filt_str = f"{cfg['n_filters_1']}/{cfg['n_filters_2']}"
        is_baseline = (cfg['kernel_size'] == 5 and cfg['n_blocks'] == 2
                       and cfg['n_filters_1'] == 32 and cfg['n_filters_2'] == 64)
        tag = " *" if is_baseline else ""
        lines.append(
            f"{config_to_name(cfg) + tag:<25} {cfg['kernel_size']:>6} {cfg['n_blocks']:>6} "
            f"{filt_str:>10} {r['n_params']:>8,} "
            f"{clf['roc_auc']:>9.4f} {clf['pr_auc']:>8.4f} "
            f"{cp['miscoverage_overall']:>8.4f} {cp['avg_set_size_overall']:>8.4f} {cp['singleton_rate_overall']:>8.4f}"
        )

    lines.append("-" * 120)
    lines.append("* = paper baseline configuration")
    lines.append("=" * 120)
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation Study (R2-3)')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Path to PTB-XL dataset root')
    parser.add_argument('--output_dir', type=str, default='experiments/results/ablation',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Deduplicate configs
    seen = set()
    unique_configs = []
    for cfg in ABLATION_CONFIGS:
        key = config_to_name(cfg)
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)

    print(f"Running {len(unique_configs)} ablation configurations")

    all_results = []
    for i, cfg in enumerate(unique_configs):
        name = config_to_name(cfg)
        print(f"\n{'='*60}")
        print(f"  Ablation {i+1}/{len(unique_configs)}: {name}")
        print(f"  kernel={cfg['kernel_size']}, blocks={cfg['n_blocks']}, "
              f"filters={cfg['n_filters_1']}/{cfg['n_filters_2']}")
        print(f"{'='*60}")

        def model_builder(input_shape, _cfg=cfg):
            return create_lightweight_cnn_ablation(input_shape, **_cfg)

        result = run_experiment(
            base_path=args.base_path,
            fold_config=STANDARD_FOLDS,
            model_builder=model_builder,
            model_name=name,
            seed=42,
            verbose=True
        )
        all_results.append(result)
        path = save_results(result, args.output_dir)
        print(f"  Saved: {path}")

    # Format and save table
    table = format_ablation_table(all_results, unique_configs)
    with open(os.path.join(args.output_dir, 'ablation_table.txt'), 'w') as f:
        f.write(table)
    print(f"\n{table}")

    # Save as CSV
    rows = []
    for cfg, r in zip(unique_configs, all_results):
        row = {**cfg, 'n_params': r['n_params']}
        row.update({f'clf_{k}': v for k, v in r['classification'].items()})
        row.update({f'cp_{k}': v for k, v in r['conformal'].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'ablation_results.csv'), index=False)

    print(f"\nAll results saved to {args.output_dir}/")
