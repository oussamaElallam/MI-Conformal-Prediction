"""Authoritative PTB-XL training and split-conformal artifact generator."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, Dense, GlobalAveragePooling1D, Input, MaxPooling1D
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential

from experiments.data_splitting import assert_group_disjoint, make_group_id, split_summary, stratified_group_holdout


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_labels(metadata_path: str, scp_statements_path: str) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    statements = pd.read_csv(scp_statements_path, index_col=0)
    statements = statements[statements.diagnostic_class.isin(["MI", "NORM"])]

    def get_class(value: str) -> str | None:
        try:
            codes = ast.literal_eval(value)
        except Exception:
            return None
        scores = {"MI": 0.0, "NORM": 0.0}
        for code, weight in codes.items():
            if code in statements.index:
                cls = statements.loc[code].diagnostic_class
                if cls in scores:
                    scores[cls] += float(weight)
        if scores["MI"] == 0.0 and scores["NORM"] == 0.0:
            return None
        return "MI" if scores["MI"] >= scores["NORM"] else "NORM"

    metadata = metadata.copy()
    metadata["diagnostic_superclass"] = metadata.scp_codes.apply(get_class)
    metadata = metadata.dropna(subset=["diagnostic_superclass"]).copy()
    metadata["label"] = (metadata.diagnostic_superclass == "MI").astype(int)
    metadata = make_group_id(metadata, preferred_columns=("patient_id", "ecg_id"))
    return metadata


def load_signals_for_rows(base_path: str, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    signals, labels = [], []
    for row in rows.itertuples(index=False):
        signal, _ = wfdb.rdsamp(os.path.join(base_path, row.filename_lr))
        signals.append(np.asarray(signal, dtype=np.float32))
        labels.append(int(row.label))
    return np.stack(signals), np.asarray(labels, dtype=np.int64)


def create_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    return Sequential([
        Input(shape=input_shape),
        Conv1D(32, 5, padding="same", use_bias=False),
        BatchNormalization(), Activation("relu"), MaxPooling1D(2),
        Conv1D(64, 5, padding="same", use_bias=False),
        BatchNormalization(), Activation("relu"), MaxPooling1D(2),
        GlobalAveragePooling1D(), Dense(64, activation="relu"), Dense(1, activation="sigmoid"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_path", default="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    parser.add_argument("--output_dir", default=os.path.join("results", "split_conformal"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    metadata = parse_labels(
        os.path.join(args.base_path, "ptbxl_database.csv"),
        os.path.join(args.base_path, "scp_statements.csv"),
    )
    development = metadata[metadata.strat_fold.isin(range(1, 9))].copy()
    validation = metadata[metadata.strat_fold == 9].copy()
    test = metadata[metadata.strat_fold == 10].copy()
    proper, calibration = stratified_group_holdout(
        development,
        group_column="group_id",
        label_column="label",
        test_size=0.20,
        random_state=args.seed,
    )
    assert_group_disjoint(
        ("train", proper), ("calibration", calibration), ("validation", validation), ("test", test),
        group_column="group_id",
    )

    x_train, y_train = load_signals_for_rows(args.base_path, proper)
    x_cal, y_cal = load_signals_for_rows(args.base_path, calibration)
    x_val, y_val = load_signals_for_rows(args.base_path, validation)
    x_test, y_test = load_signals_for_rows(args.base_path, test)

    mean = x_train.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    for array in (x_train, x_cal, x_val, x_test):
        array -= mean
        array /= std

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    model = create_model(x_train.shape[1:])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC(name="auc")])
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)],
        class_weight=class_weights,
        verbose=1,
    )
    test_loss, test_accuracy, test_auc = model.evaluate(x_test, y_test, verbose=0)
    test_prob = model.predict(x_test, verbose=0).reshape(-1)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "split_conformal_model.h5")
    model.save(model_path)
    np.save(os.path.join(args.output_dir, "lead_mean.npy"), mean)
    np.save(os.path.join(args.output_dir, "lead_std.npy"), std)
    np.save(os.path.join(args.output_dir, "X_cal.npy"), x_cal)
    np.save(os.path.join(args.output_dir, "y_cal.npy"), y_cal)
    pd.DataFrame(history.history).to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)
    pd.DataFrame({"y_true": y_test, "probability_mi": test_prob}).to_csv(
        os.path.join(args.output_dir, "test_predictions_fp32.csv"), index=False
    )

    manifest_parts = []
    for name, frame in (("train", proper), ("calibration", calibration), ("validation", validation), ("test", test)):
        part = frame.copy()
        part["split"] = name
        manifest_parts.append(part)
    split_manifest = pd.concat(manifest_parts, ignore_index=True)
    split_manifest_path = os.path.join(args.output_dir, "split_manifest.csv")
    split_manifest.to_csv(split_manifest_path, index=False)
    split_summary(
        (("train", proper), ("calibration", calibration), ("validation", validation), ("test", test)),
        group_column="group_id", label_column="label",
    ).to_csv(os.path.join(args.output_dir, "split_summary.csv"), index=False)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_auc": float(test_auc),
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "seed": args.seed,
        "epochs_completed": len(history.history.get("loss", [])),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    artifact_manifest = {
        "h5_model_path": model_path,
        "h5_model_sha256": sha256_file(model_path),
        "split_manifest_path": split_manifest_path,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "calibration_array_state": "already normalized exactly once with proper-training statistics",
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "artifact_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(artifact_manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
