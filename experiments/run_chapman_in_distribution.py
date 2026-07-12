"""Train, calibrate, quantize, and test inside Chapman-Shaoxing.

The split is group-disjoint.  When the dataset header contains an explicit patient
identifier it is used; otherwise the record identifier is used and the fallback is
reported in the output manifest rather than silently called patient-wise.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from experiments.architectures import create_lightweight_cnn
from experiments.data_splitting import (
    assert_group_disjoint,
    split_summary,
    stratified_group_holdout,
)
from experiments.reviewer_experiment_utils import (
    classification_metrics,
    convert_full_int8,
    dump_json,
    encode_sets,
    evaluate_prediction_sets,
    mondrian_prediction_sets,
    predict_tflite,
    set_global_determinism,
)
from validation.chapman_loader import load_chapman_data


def four_way_group_split(manifest: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    development, test = stratified_group_holdout(
        manifest,
        group_column="group_id",
        label_column="label",
        test_size=0.10,
        random_state=seed,
    )
    pool, validation = stratified_group_holdout(
        development,
        group_column="group_id",
        label_column="label",
        test_size=1.0 / 9.0,
        random_state=seed + 1,
    )
    train, calibration = stratified_group_holdout(
        pool,
        group_column="group_id",
        label_column="label",
        test_size=0.20,
        random_state=seed + 2,
    )
    partitions = {
        "train": train,
        "calibration": calibration,
        "validation": validation,
        "test": test,
    }
    assert_group_disjoint(
        *((name, frame) for name, frame in partitions.items()),
        group_column="group_id",
    )
    return partitions


def arrays_for(frame: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = frame["array_index"].to_numpy(dtype=int)
    labels = y[indices]
    if not np.array_equal(labels, frame["label"].to_numpy(dtype=int)):
        raise AssertionError("Manifest labels and signal-array labels are misaligned")
    return x[indices].copy(), labels.copy()


def demographic_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, frame in manifest.groupby("split", sort=False):
        sex = frame["sex"].astype(str).str.strip().str.lower()
        rows.append(
            {
                "split": split,
                "records": int(len(frame)),
                "groups": int(frame["group_id"].nunique()),
                "normal": int((frame["label"] == 0).sum()),
                "mi": int((frame["label"] == 1).sum()),
                "age_mean": float(frame["age"].mean()),
                "age_std": float(frame["age"].std()),
                "female_count": int(sex.isin(["f", "female"]).sum()),
                "male_count": int(sex.isin(["m", "male"]).sum()),
                "unknown_sex_count": int((~sex.isin(["f", "female", "m", "male"])).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="chapman_shaoxing")
    parser.add_argument(
        "--output_dir",
        default=os.path.join("experiments", "results", "chapman_in_distribution"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_records", type=int)
    parser.add_argument(
        "--short_record_policy",
        choices=["skip", "pad"],
        default="skip",
        help="Exclude and log short records, or zero-pad them before resampling.",
    )
    args = parser.parse_args()

    set_global_determinism(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    x, y, manifest, exclusions = load_chapman_data(
        data_dir=args.data_dir,
        max_records=args.max_records,
        short_record_policy=args.short_record_policy,
        return_metadata=True,
    )
    if len(x) == 0 or len(np.unique(y)) != 2:
        raise RuntimeError("Chapman loader returned no usable two-class cohort")
    if len(manifest) != len(x):
        raise AssertionError("Manifest and signal-array lengths differ")

    partitions = four_way_group_split(manifest, args.seed)
    split_manifest = pd.concat(
        [frame.assign(split=name) for name, frame in partitions.items()],
        ignore_index=True,
    )
    split_manifest.to_csv(
        os.path.join(args.output_dir, "chapman_split_manifest_used.csv"), index=False
    )
    if not exclusions.empty:
        exclusions.to_csv(
            os.path.join(args.output_dir, "chapman_exclusions.csv"), index=False
        )

    summary = split_summary(
        partitions.items(), group_column="group_id", label_column="label"
    )
    summary.to_csv(os.path.join(args.output_dir, "chapman_split_counts.csv"), index=False)
    demographics = demographic_summary(split_manifest)
    demographics.to_csv(
        os.path.join(args.output_dir, "chapman_demographics_by_split.csv"), index=False
    )

    x_train, y_train = arrays_for(partitions["train"], x, y)
    x_cal, y_cal = arrays_for(partitions["calibration"], x, y)
    x_val, y_val = arrays_for(partitions["validation"], x, y)
    x_test, y_test = arrays_for(partitions["test"], x, y)

    mean = x_train.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    for array in (x_train, x_cal, x_val, x_test):
        array -= mean
        array /= std
    np.save(os.path.join(args.output_dir, "lead_mean.npy"), mean)
    np.save(os.path.join(args.output_dir, "lead_std.npy"), std)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(label): float(weight) for label, weight in zip(classes, weights)}

    model = create_lightweight_cnn(x_train.shape[1:])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )
    keras_path = os.path.join(args.output_dir, "chapman_model.keras")
    model.save(keras_path)
    pd.DataFrame(history.history).to_csv(
        os.path.join(args.output_dir, "training_history.csv"), index=False
    )

    fp32_tflite_path = os.path.join(args.output_dir, "chapman_model_fp32.tflite")
    fp32_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(fp32_tflite_path, "wb") as handle:
        handle.write(fp32_converter.convert())

    int8_path = os.path.join(args.output_dir, "chapman_model_int8.tflite")
    convert_full_int8(model, x_train, int8_path, seed=args.seed)

    fp32_test = model.predict(x_test, verbose=0).reshape(-1)
    int8_cal = predict_tflite(int8_path, x_cal)
    int8_test = predict_tflite(int8_path, x_test)
    included, p_values = mondrian_prediction_sets(
        int8_cal, y_cal, int8_test, args.epsilon
    )

    conformal = evaluate_prediction_sets(included, y_test)
    fp32_classification = classification_metrics(fp32_test, y_test)
    int8_classification = classification_metrics(int8_test, y_test)

    counts = {
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_calibration": int(len(y_cal)),
        "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_test_mi": int(y_test.sum()),
        "n_test_normal": int(len(y_test) - y_test.sum()),
    }
    row = {
        "dataset": "Chapman-Shaoxing in-distribution",
        **counts,
        **{f"fp32_{key}": value for key, value in fp32_classification.items()},
        **{f"int8_{key}": value for key, value in int8_classification.items()},
        **conformal,
    }
    pd.DataFrame([row]).to_csv(
        os.path.join(args.output_dir, "chapman_in_distribution_metrics.csv"),
        index=False,
    )

    test_records = partitions["test"].reset_index(drop=True).copy()
    test_records["y_true"] = y_test
    test_records["fp32_probability_mi"] = fp32_test
    test_records["int8_probability_mi"] = int8_test
    test_records["p_normal"] = p_values[:, 0]
    test_records["p_mi"] = p_values[:, 1]
    test_records["prediction_set"] = encode_sets(included)
    test_records["covered"] = included[np.arange(len(y_test)), y_test]
    test_records.to_csv(
        os.path.join(args.output_dir, "chapman_test_predictions.csv"), index=False
    )

    group_source_counts = {
        str(key): int(value)
        for key, value in manifest["group_id_source"].value_counts().items()
    }
    payload = {
        "design": {
            "seed": args.seed,
            "epsilon": args.epsilon,
            "split_fractions_target": {
                "train": 0.64,
                "calibration": 0.16,
                "validation": 0.10,
                "test": 0.10,
            },
            "group_disjoint": True,
            "group_id_source_counts": group_source_counts,
            "short_record_policy": args.short_record_policy,
            "normalization": "proper training only",
            "class_weights": class_weights,
            "epochs_completed": len(history.history.get("loss", [])),
        },
        "counts": counts,
        "classification": {
            "fp32": fp32_classification,
            "int8": int8_classification,
        },
        "conformal_int8_calibrated": conformal,
        "model_files": {
            "keras_path": keras_path,
            "fp32_tflite_path": fp32_tflite_path,
            "fp32_tflite_bytes": os.path.getsize(fp32_tflite_path),
            "int8_tflite_path": int8_path,
            "int8_tflite_bytes": os.path.getsize(int8_path),
            "fp32_to_int8_tflite_size_ratio": os.path.getsize(fp32_tflite_path)
            / os.path.getsize(int8_path),
        },
        "split_summary": summary.to_dict(orient="records"),
        "exclusion_count": int(len(exclusions)),
    }
    dump_json(
        os.path.join(args.output_dir, "chapman_in_distribution_results.json"),
        payload,
    )

    print(summary.to_string(index=False))
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
