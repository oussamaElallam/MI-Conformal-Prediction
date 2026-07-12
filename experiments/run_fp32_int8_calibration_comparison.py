"""Matched FP32-calibration versus INT8-calibration experiment.

Reviewer question
-----------------
Does calibrating Mondrian conformal prediction on the deployed INT8 model change
coverage or efficiency relative to the conventional workflow in which thresholds
are derived from FP32 outputs and then applied to INT8 predictions?

The comparison is deliberately paired:
- one trained model,
- one calibration set,
- one test set,
- one epsilon,
- identical INT8 test probabilities.

Usage
-----
python -m experiments.run_fp32_int8_calibration_comparison \
    --base_path ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from experiments.experiment_runner import load_signals_by_fold, parse_labels
from experiments.reviewer_experiment_utils import (
    classification_metrics,
    dump_json,
    encode_sets,
    evaluate_prediction_sets,
    mondrian_prediction_sets,
    predict_tflite,
    set_global_determinism,
)


def paper_table(rows: list[dict]) -> str:
    columns = [
        ("workflow", "Workflow", 31),
        ("miscoverage_overall", "Miscov.", 9),
        ("miscoverage_normal", "Normal", 9),
        ("miscoverage_mi", "MI", 9),
        ("avg_set_size_overall", "Set size", 10),
        ("singleton_rate_overall", "Singleton", 10),
        ("empty_set_rate_overall", "Empty", 8),
    ]
    header = " ".join(f"{label:>{width}}" for _, label, width in columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        values = []
        for key, _, width in columns:
            if key == "workflow":
                values.append(f"{row[key]:>{width}}")
            else:
                values.append(f"{row[key]:>{width}.4f}")
        lines.append(" ".join(values))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_path", required=True, help="PTB-XL dataset root")
    parser.add_argument(
        "--artifact_dir",
        default=os.path.join("results", "split_conformal"),
        help="Directory containing the fixed model, calibration set, and normalization stats",
    )
    parser.add_argument(
        "--int8_model",
        default=os.path.join("edge", "model", "model_int8.tflite"),
        help="Full-INT8 TFLite model exported from the same fixed Keras model",
    )
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default=os.path.join("experiments", "results", "fp32_int8_calibration"),
    )
    args = parser.parse_args()

    set_global_determinism(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.artifact_dir, "split_conformal_model.h5")
    required = [
        model_path,
        os.path.join(args.artifact_dir, "X_cal.npy"),
        os.path.join(args.artifact_dir, "y_cal.npy"),
        os.path.join(args.artifact_dir, "lead_mean.npy"),
        os.path.join(args.artifact_dir, "lead_std.npy"),
        args.int8_model,
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n- " + "\n- ".join(missing))

    print("Loading fixed FP32 model and fixed calibration set...")
    model = tf.keras.models.load_model(model_path)
    x_cal = np.load(os.path.join(args.artifact_dir, "X_cal.npy"), mmap_mode="r")
    y_cal = np.load(os.path.join(args.artifact_dir, "y_cal.npy"))
    lead_mean = np.load(os.path.join(args.artifact_dir, "lead_mean.npy"))
    lead_std = np.load(os.path.join(args.artifact_dir, "lead_std.npy"))

    print("Reconstructing the unchanged PTB-XL fold-10 test set...")
    metadata = parse_labels(args.base_path)
    x_test, y_test = load_signals_by_fold(args.base_path, metadata, [10])
    x_test = x_test.astype(np.float32, copy=False)
    x_test -= lead_mean.astype(np.float32)
    x_test /= lead_std.astype(np.float32)

    # Export a directly comparable FP32 TFLite file so the manuscript can
    # report the real FP32-to-INT8 file-size ratio rather than the unrelated
    # ResNet-to-lightweight parameter-count ratio.
    fp32_tflite_path = os.path.join(args.output_dir, "fixed_model_fp32.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(fp32_tflite_path, "wb") as handle:
        handle.write(converter.convert())

    print("Computing paired FP32 and INT8 probabilities...")
    fp32_cal_prob = model.predict(x_cal, verbose=0).reshape(-1)
    fp32_test_prob = model.predict(x_test, verbose=0).reshape(-1)
    int8_cal_prob = predict_tflite(args.int8_model, np.asarray(x_cal))
    int8_test_prob = predict_tflite(args.int8_model, x_test)

    # Conventional: FP32 calibration, deployed INT8 test model.
    conventional_sets, conventional_p = mondrian_prediction_sets(
        fp32_cal_prob, y_cal, int8_test_prob, args.epsilon
    )
    # Proposed: INT8 calibration, same deployed INT8 test model.
    qa_sets, qa_p = mondrian_prediction_sets(
        int8_cal_prob, y_cal, int8_test_prob, args.epsilon
    )

    conventional_metrics = evaluate_prediction_sets(conventional_sets, y_test)
    qa_metrics = evaluate_prediction_sets(qa_sets, y_test)
    conventional_metrics["workflow"] = "FP32 calibration -> INT8 test"
    qa_metrics["workflow"] = "INT8 calibration -> INT8 test"
    rows = [conventional_metrics, qa_metrics]

    set_changed = np.any(conventional_sets != qa_sets, axis=1)
    conventional_covered = conventional_sets[np.arange(len(y_test)), y_test]
    qa_covered = qa_sets[np.arange(len(y_test)), y_test]
    paired_summary = {
        "n_test": int(len(y_test)),
        "set_membership_changed_count": int(set_changed.sum()),
        "set_membership_changed_rate": float(set_changed.mean()),
        "set_size_changed_count": int(
            np.sum(conventional_sets.sum(axis=1) != qa_sets.sum(axis=1))
        ),
        "conventional_only_covered": int(np.sum(conventional_covered & ~qa_covered)),
        "qa_only_covered": int(np.sum(~conventional_covered & qa_covered)),
        "both_covered": int(np.sum(conventional_covered & qa_covered)),
        "neither_covered": int(np.sum(~conventional_covered & ~qa_covered)),
        "mean_abs_calibration_probability_shift": float(
            np.mean(np.abs(fp32_cal_prob - int8_cal_prob))
        ),
        "max_abs_calibration_probability_shift": float(
            np.max(np.abs(fp32_cal_prob - int8_cal_prob))
        ),
        "mean_abs_test_probability_shift": float(
            np.mean(np.abs(fp32_test_prob - int8_test_prob))
        ),
        "max_abs_test_probability_shift": float(
            np.max(np.abs(fp32_test_prob - int8_test_prob))
        ),
    }

    # The discriminative score is not workflow-specific because both deployed
    # conformal workflows use the same INT8 test probabilities.
    discriminative = {
        "fp32_test": classification_metrics(fp32_test_prob, y_test),
        "int8_test": classification_metrics(int8_test_prob, y_test),
    }

    metrics_path = os.path.join(args.output_dir, "fp32_int8_calibration_metrics.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    sample_table = pd.DataFrame(
        {
            "y_true": y_test,
            "fp32_probability_mi": fp32_test_prob,
            "int8_probability_mi": int8_test_prob,
            "conventional_p_normal": conventional_p[:, 0],
            "conventional_p_mi": conventional_p[:, 1],
            "qa_p_normal": qa_p[:, 0],
            "qa_p_mi": qa_p[:, 1],
            "conventional_set": encode_sets(conventional_sets),
            "qa_set": encode_sets(qa_sets),
            "conventional_covered": conventional_covered,
            "qa_covered": qa_covered,
            "set_changed": set_changed,
        }
    )
    sample_table.to_csv(
        os.path.join(args.output_dir, "fp32_int8_sample_level.csv"), index=False
    )

    payload = {
        "design": {
            "epsilon": args.epsilon,
            "seed": args.seed,
            "fixed_model": model_path,
            "fixed_calibration_set": os.path.join(args.artifact_dir, "X_cal.npy"),
            "fixed_test_fold": 10,
            "deployed_test_model": args.int8_model,
        },
        "conformal_workflows": rows,
        "paired_summary": paired_summary,
        "classification": discriminative,
        "model_file_sizes": {
            "fp32_tflite_path": fp32_tflite_path,
            "fp32_tflite_bytes": int(os.path.getsize(fp32_tflite_path)),
            "int8_tflite_path": args.int8_model,
            "int8_tflite_bytes": int(os.path.getsize(args.int8_model)),
            "fp32_to_int8_size_ratio": float(
                os.path.getsize(fp32_tflite_path) / os.path.getsize(args.int8_model)
            ),
        },
    }
    dump_json(os.path.join(args.output_dir, "fp32_int8_calibration_results.json"), payload)

    table = paper_table(rows)
    with open(
        os.path.join(args.output_dir, "fp32_int8_paper_table.txt"), "w", encoding="utf-8"
    ) as handle:
        handle.write(table + "\n")

    print("\n" + table)
    print("\nPaired set changes:")
    print(json.dumps(paired_summary, indent=2))
    print(f"\nSaved reviewer-comparison outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
