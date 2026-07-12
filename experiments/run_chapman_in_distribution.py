"""Reviewer #9: train/calibrate/test the model in-distribution on Chapman-Shaoxing.

Run: python -m experiments.run_chapman_in_distribution --data_dir chapman_shaoxing
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from experiments.architectures import create_lightweight_cnn
from experiments.reviewer_experiment_utils import (
    classification_metrics, convert_full_int8, dump_json, encode_sets,
    evaluate_prediction_sets, mondrian_prediction_sets, predict_tflite,
    set_global_determinism,
)
from validation.chapman_loader import load_chapman_data


def stratified_four_way_split(x, y, seed):
    """64% proper train, 16% calibration, 10% validation, 10% test."""
    x_dev, x_test, y_dev, y_test = train_test_split(
        x, y, test_size=.10, random_state=seed, stratify=y)
    x_pool, x_val, y_pool, y_val = train_test_split(
        x_dev, y_dev, test_size=1/9, random_state=seed+1, stratify=y_dev)
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_pool, y_pool, test_size=.20, random_state=seed+2, stratify=y_pool)
    return x_train, y_train, x_cal, y_cal, x_val, y_val, x_test, y_test


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="chapman_shaoxing")
    p.add_argument("--output_dir", default="experiments/results/chapman_in_distribution")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epsilon", type=float, default=.10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max_records", type=int)
    args = p.parse_args()

    set_global_determinism(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    x, y = load_chapman_data(data_dir=args.data_dir, max_records=args.max_records)
    if len(x) == 0 or len(np.unique(y)) != 2:
        raise RuntimeError("Chapman loader returned no usable two-class cohort")
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    parts = stratified_four_way_split(x, y, args.seed)
    x_train, y_train, x_cal, y_cal, x_val, y_val, x_test, y_test = parts

    # Proper-training statistics only: no calibration/validation/test leakage.
    mean = x_train.mean(axis=(0,1), keepdims=True, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=(0,1), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1., std).astype(np.float32)
    for array in (x_train, x_cal, x_val, x_test):
        array -= mean
        array /= std
    np.save(os.path.join(args.output_dir, "lead_mean.npy"), mean)
    np.save(os.path.join(args.output_dir, "lead_std.npy"), std)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    model = create_lightweight_cnn(x_train.shape[1:])
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience,
                                 restore_best_weights=True)], verbose=1)
    model_path = os.path.join(args.output_dir, "chapman_model.keras")
    model.save(model_path)
    pd.DataFrame(history.history).to_csv(
        os.path.join(args.output_dir, "training_history.csv"), index=False)

    int8_path = os.path.join(args.output_dir, "chapman_model_int8.tflite")
    convert_full_int8(model, x_train, int8_path, seed=args.seed)
    fp32_test = model.predict(x_test, verbose=0).reshape(-1)
    int8_cal = predict_tflite(int8_path, x_cal)
    int8_test = predict_tflite(int8_path, x_test)
    included, p_values = mondrian_prediction_sets(
        int8_cal, y_cal, int8_test, args.epsilon)

    cp = evaluate_prediction_sets(included, y_test)
    fp32_clf = classification_metrics(fp32_test, y_test)
    int8_clf = classification_metrics(int8_test, y_test)
    counts = {
        "n_total": int(len(y)), "n_train": int(len(y_train)),
        "n_calibration": int(len(y_cal)), "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)), "n_test_mi": int(y_test.sum()),
        "n_test_normal": int(len(y_test)-y_test.sum()),
    }
    row = {
        "dataset": "Chapman-Shaoxing in-distribution", **counts,
        **{f"fp32_{k}": v for k,v in fp32_clf.items()},
        **{f"int8_{k}": v for k,v in int8_clf.items()}, **cp,
    }
    pd.DataFrame([row]).to_csv(
        os.path.join(args.output_dir, "chapman_in_distribution_metrics.csv"), index=False)
    pd.DataFrame({
        "y_true": y_test, "fp32_probability_mi": fp32_test,
        "int8_probability_mi": int8_test, "p_normal": p_values[:,0],
        "p_mi": p_values[:,1], "prediction_set": encode_sets(included),
        "covered": included[np.arange(len(y_test)), y_test],
    }).to_csv(os.path.join(args.output_dir, "chapman_test_predictions.csv"), index=False)
    payload = {
        "design": {"seed": args.seed, "epsilon": args.epsilon,
                   "split_fractions": {"train":.64,"calibration":.16,"validation":.10,"test":.10},
                   "normalization": "proper training only", "class_weights": class_weights,
                   "epochs_completed": len(history.history.get("loss", []))},
        "counts": counts, "classification": {"fp32": fp32_clf, "int8": int8_clf},
        "conformal_int8_calibrated": cp,
        "model_files": {"fp32_path": model_path, "fp32_bytes": os.path.getsize(model_path),
                        "int8_path": int8_path, "int8_bytes": os.path.getsize(int8_path),
                        "fp32_to_int8_file_size_ratio": os.path.getsize(model_path)/os.path.getsize(int8_path)},
    }
    dump_json(os.path.join(args.output_dir, "chapman_in_distribution_results.json"), payload)
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
