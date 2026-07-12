# R4 reviewer-requested experiments

This folder adds the two experiments requested in the BSPC major-revision reports.
The scripts produce machine-readable CSV/JSON outputs and paper-ready summaries;
they do **not** insert guessed numerical values into the manuscript.

## 1. Matched FP32-calibration versus INT8-calibration comparison

This is the central quantization-aware conformal experiment. Both workflows use:

- the same trained Keras model,
- the same calibration samples and labels,
- the same PTB-XL fold-10 test samples,
- the same significance level,
- the same deployed INT8 test probabilities.

Only the calibration precision changes:

1. **Conventional:** FP32 calibration outputs -> INT8 test outputs.
2. **Quantization-aware:** INT8 calibration outputs -> INT8 test outputs.

First generate the fixed split-conformal and INT8 artifacts:

```bash
python train_split_conformal_model.py
python edge/export_tflite_micro_model.py
```

Then run:

```bash
python -m experiments.run_fp32_int8_calibration_comparison \
  --base_path ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
```

Main outputs:

- `experiments/results/fp32_int8_calibration/fp32_int8_calibration_metrics.csv`
- `experiments/results/fp32_int8_calibration/fp32_int8_calibration_results.json`
- `experiments/results/fp32_int8_calibration/fp32_int8_sample_level.csv`
- `experiments/results/fp32_int8_calibration/fp32_int8_paper_table.txt`

Report overall and class-conditional miscoverage, average set size, singleton rate,
empty-set rate, and the paired fraction of samples whose prediction-set membership
changes after INT8 calibration.

## 2. Chapman-Shaoxing in-distribution experiment

This experiment trains, calibrates, and tests inside Chapman-Shaoxing using disjoint
patient groups. The split mirrors the PTB-XL workflow at cohort level:

- 64% proper training,
- 16% conformal calibration,
- 10% validation,
- 10% test.

Run:

```bash
python -m experiments.run_chapman_in_distribution \
  --data_dir chapman_shaoxing
```

The default handling for records shorter than 10 seconds is to exclude and log
them rather than zero-pad them. To reproduce the old padding behavior explicitly:

```bash
python -m experiments.run_chapman_in_distribution \
  --data_dir chapman_shaoxing \
  --short_record_policy pad
```

Main outputs:

- `experiments/results/chapman_in_distribution/chapman_in_distribution_metrics.csv`
- `experiments/results/chapman_in_distribution/chapman_in_distribution_results.json`
- `experiments/results/chapman_in_distribution/chapman_split_manifest_used.csv`
- `experiments/results/chapman_in_distribution/chapman_test_predictions.csv`
- `experiments/results/chapman_in_distribution/chapman_demographics_by_split.csv`
- `experiments/results/chapman_in_distribution/chapman_exclusions.csv` when exclusions occur

The primary manuscript comparison should distinguish:

1. PTB-XL in-distribution performance.
2. PTB-XL model + PTB-XL calibration applied to Chapman without recalibration.
3. Chapman-trained + Chapman-INT8-calibrated in-distribution performance.

The third setting tests whether domain-matched training and labeled local calibration
restore discrimination and conformal coverage. It does not turn miscoverage into an
unlabeled online drift detector.
