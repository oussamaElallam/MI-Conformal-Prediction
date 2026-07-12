"""Evaluate Mondrian conformal prediction from the authoritative PTB-XL model.

This script distinguishes empirical coverage (true label contained in the set)
from singleton rate (fraction of unambiguous one-label sets).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from experiments.experiment_runner import load_signals_by_fold, parse_labels
from experiments.reviewer_experiment_utils import (
    encode_sets,
    evaluate_prediction_sets,
    mondrian_prediction_sets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_path",
        default="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    )
    parser.add_argument(
        "--artifact_dir", default=os.path.join("results", "split_conformal")
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10, 0.15, 0.20],
    )
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    required = [
        os.path.join(args.artifact_dir, "split_conformal_model.h5"),
        os.path.join(args.artifact_dir, "X_cal.npy"),
        os.path.join(args.artifact_dir, "y_cal.npy"),
        os.path.join(args.artifact_dir, "lead_mean.npy"),
        os.path.join(args.artifact_dir, "lead_std.npy"),
        os.path.join(args.artifact_dir, "artifact_manifest.json"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing authoritative artifacts:\n- " + "\n- ".join(missing))

    with open(os.path.join(args.artifact_dir, "artifact_manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("x_cal_normalized") is not True:
        raise RuntimeError("Expected normalized X_cal according to artifact_manifest.json")

    model = tf.keras.models.load_model(
        os.path.join(args.artifact_dir, "split_conformal_model.h5")
    )
    x_cal = np.load(os.path.join(args.artifact_dir, "X_cal.npy"))
    y_cal = np.load(os.path.join(args.artifact_dir, "y_cal.npy"))
    mean = np.load(os.path.join(args.artifact_dir, "lead_mean.npy"))
    std = np.load(os.path.join(args.artifact_dir, "lead_std.npy"))

    metadata = parse_labels(args.base_path)
    x_test, y_test = load_signals_by_fold(args.base_path, metadata, [10])
    x_test = x_test.astype(np.float32, copy=False)
    x_test -= mean.astype(np.float32)
    x_test /= std.astype(np.float32)

    calibration_probabilities = model.predict(x_cal, verbose=0).reshape(-1)
    test_probabilities = model.predict(x_test, verbose=0).reshape(-1)

    rows = []
    prediction_sets_at_point_one = None
    p_values_at_point_one = None
    for epsilon in args.epsilons:
        included, p_values = mondrian_prediction_sets(
            calibration_probabilities,
            y_cal,
            test_probabilities,
            epsilon,
        )
        metrics = evaluate_prediction_sets(included, y_test)
        rows.append({"epsilon": epsilon, "nominal_coverage": 1.0 - epsilon, **metrics})
        if np.isclose(epsilon, 0.10):
            prediction_sets_at_point_one = included
            p_values_at_point_one = p_values

    results = pd.DataFrame(rows).sort_values("epsilon")
    os.makedirs(args.output_dir, exist_ok=True)
    results.to_csv(
        os.path.join(args.output_dir, "conformal_tradeoff_metrics.csv"), index=False
    )

    nominal = results["nominal_coverage"].to_numpy()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(nominal, results["avg_set_size_overall"], marker="o")
    axes[0].set_xlabel("Nominal coverage (1 - epsilon)")
    axes[0].set_ylabel("Average prediction-set size")
    axes[0].set_title("A. Set size")

    axes[1].plot(nominal, results["singleton_rate_overall"], marker="o", label="Overall")
    axes[1].plot(nominal, results["singleton_rate_normal"], marker="s", label="Normal")
    axes[1].plot(nominal, results["singleton_rate_mi"], marker="^", label="MI")
    axes[1].set_xlabel("Nominal coverage (1 - epsilon)")
    axes[1].set_ylabel("Singleton rate")
    axes[1].set_title("B. Unambiguous sets")
    axes[1].legend()

    axes[2].plot([nominal.min(), nominal.max()], [nominal.min(), nominal.max()], "--", label="Nominal")
    axes[2].plot(nominal, results["coverage_overall"], marker="o", label="Overall")
    axes[2].plot(nominal, results["coverage_normal"], marker="s", label="Normal")
    axes[2].plot(nominal, results["coverage_mi"], marker="^", label="MI")
    axes[2].set_xlabel("Nominal coverage (1 - epsilon)")
    axes[2].set_ylabel("Empirical label coverage")
    axes[2].set_title("C. Coverage calibration")
    axes[2].legend()

    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        os.path.join(args.output_dir, "conformal_tradeoff_figure.png"), dpi=300
    )
    plt.close(figure)

    if prediction_sets_at_point_one is not None and p_values_at_point_one is not None:
        pd.DataFrame(
            {
                "y_true": y_test,
                "probability_mi": test_probabilities,
                "p_normal": p_values_at_point_one[:, 0],
                "p_mi": p_values_at_point_one[:, 1],
                "prediction_set": encode_sets(prediction_sets_at_point_one),
                "covered": prediction_sets_at_point_one[
                    np.arange(len(y_test)), y_test
                ],
            }
        ).to_csv(
            os.path.join(args.output_dir, "conformal_predictions_epsilon_0_10.csv"),
            index=False,
        )

    print(results.to_string(index=False))
    print(
        "Coverage means true-label inclusion over the test cohort; singleton rate "
        "means the fraction of one-label prediction sets."
    )


if __name__ == "__main__":
    main()
