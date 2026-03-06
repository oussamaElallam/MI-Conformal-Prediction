#!/bin/bash
# ──────────────────────────────────────────────────────────
# BSPC Revision: Run All Experiments
# ──────────────────────────────────────────────────────────
#
# Usage:
#   bash experiments/run_all.sh <PTB-XL-DIR>
#
# Example:
#   bash experiments/run_all.sh ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
#
# On Colab, set PTB-XL-DIR to the extracted dataset path.
# Estimated total time: ~2-4 hours on Colab GPU (T4)
# ──────────────────────────────────────────────────────────

set -e

BASE_PATH=${1:?"Usage: bash experiments/run_all.sh <PTB-XL-DIR>"}

echo "============================================"
echo " BSPC Revision Experiments"
echo " Dataset: $BASE_PATH"
echo " Started: $(date)"
echo "============================================"

# ── R4-3: Fold Stability (longest — start first) ──
echo ""
echo ">>> [1/3] Fold Stability Analysis (R4-3)"
echo "    Running standard + 4 random fold configurations..."
python -m experiments.run_fold_stability \
    --base_path "$BASE_PATH" \
    --n_configs 4 \
    --output_dir experiments/results/fold_stability

# ── R2-2: Architecture Comparison ──
echo ""
echo ">>> [2/3] Architecture Comparison (R2-2)"
echo "    Evaluating 5 model architectures..."
python -m experiments.run_architecture_comparison \
    --base_path "$BASE_PATH" \
    --output_dir experiments/results/architecture_comparison

# ── R2-3: Ablation Study ──
echo ""
echo ">>> [3/3] Ablation Study (R2-3)"
echo "    Running kernel size / depth / width sweep..."
python -m experiments.run_ablation \
    --base_path "$BASE_PATH" \
    --output_dir experiments/results/ablation

echo ""
echo "============================================"
echo " All experiments complete!"
echo " Finished: $(date)"
echo ""
echo " Results:"
echo "   experiments/results/fold_stability/"
echo "   experiments/results/architecture_comparison/"
echo "   experiments/results/ablation/"
echo "============================================"
