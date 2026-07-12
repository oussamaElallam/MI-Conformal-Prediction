# Quantization-aware conformal ECG case study

This repository contains the code for the PTB-XL/Chapman-Shaoxing embedded ECG study. The primary contribution is an engineering evaluation of Mondrian split-conformal prediction with a fully INT8 deployment model. The code does not claim per-patient “90% confidence,” guaranteed performance under distribution shift, or a clinically deployable monitoring system.

## Authoritative PTB-XL pipeline

Use one trained model for every primary table and figure. Do not combine values produced by `improved_mi_classification.py` with values from the split-conformal model.

```bash
python -m pip install -r requirements.txt
python train_split_conformal_model.py \
  --base_path <PTB-XL-DIR>
python edge/export_tflite_micro_model.py \
  --base_path <PTB-XL-DIR>
python edge/compute_cp_thresholds.py
python conformal_prediction_evaluation.py \
  --base_path <PTB-XL-DIR>
python -m experiments.run_fp32_int8_calibration_comparison \
  --base_path <PTB-XL-DIR>
```

`train_split_conformal_model.py` is the single source of truth for the primary Lightweight CNN. It uses folds 1–8 as a development pool, fold 9 for early stopping and fold 10 as the untouched test set. Proper training and conformal calibration are split by `patient_id` (with an explicitly logged record fallback only when an identifier is missing). Normalization statistics are calculated from proper training only.

Important outputs:

- `results/split_conformal/metrics.json`
- `results/split_conformal/test_predictions_fp32.csv`
- `results/split_conformal/split_manifest.csv`
- `results/split_conformal/artifact_manifest.json`
- `results/conformal_tradeoff_metrics.csv`
- `results/conformal_tradeoff_figure.png`
- `experiments/results/fp32_int8_calibration/`

The FP32-versus-INT8 experiment compares:

1. FP32 calibration outputs → the deployed INT8 test model.
2. INT8 calibration outputs → the same deployed INT8 test model.

The model, calibration labels, test set, significance level and INT8 test probabilities are held fixed. Empty sets, singleton rate, average set size and class-conditional miscoverage are reported separately.

## Chapman-Shaoxing experiments

External transfer without local retraining:

```bash
python -m validation.validate_on_chapman
```

Domain-matched in-distribution training, calibration and testing:

```bash
python -m experiments.run_chapman_in_distribution \
  --data_dir <CHAPMAN-DIR> \
  --short_record_policy skip
```

The in-distribution script creates group-disjoint train/calibration/validation/test partitions. Explicit patient IDs are used when present; otherwise record IDs are used and the fallback is reported in the JSON and CSV manifests. Short records are excluded and logged by default. Use `--short_record_policy pad` only for a declared sensitivity analysis.

Important outputs:

- `experiments/results/chapman_in_distribution/chapman_split_manifest_used.csv`
- `experiments/results/chapman_in_distribution/chapman_exclusions.csv`
- `experiments/results/chapman_in_distribution/chapman_demographics_by_split.csv`
- `experiments/results/chapman_in_distribution/chapman_in_distribution_metrics.csv`
- `experiments/results/chapman_in_distribution/chapman_in_distribution_results.json`

## Extended experiments

```bash
python -m experiments.run_fold_stability --base_path <PTB-XL-DIR>
python -m experiments.run_architecture_comparison --base_path <PTB-XL-DIR>
python -m experiments.run_ablation --base_path <PTB-XL-DIR>
```

These runners use group-disjoint proper-training/calibration splits and training-only normalization.

## ESP32-S3 deployment

The INT8 export is tied to the authoritative H5 model by SHA-256 manifests. `edge/compute_cp_thresholds.py` consumes the already-normalized calibration array exactly once and exports finite-sample class-specific thresholds. Hardware latency and RAM must be measured on the target board. Power values must be labelled as measured only when obtained with external instrumentation; otherwise label them as estimates.

## Result-use rule

Do not update the manuscript from console screenshots or separate training runs. Update all tables and figures from the generated CSV/JSON files in one completed run, and archive the artifact manifests with the revision.
