"""Export the authoritative PTB-XL model as a full-INT8 TFLite model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_representative_signals(
    base_path: str,
    manifest_path: str,
    mean: np.ndarray,
    std: np.ndarray,
    max_examples: int,
    seed: int,
) -> np.ndarray:
    manifest = pd.read_csv(manifest_path)
    if "split" not in manifest or "filename_lr" not in manifest:
        raise ValueError("split_manifest.csv must contain split and filename_lr columns")
    train_rows = manifest[manifest.split == "train"].copy()
    if train_rows.empty:
        raise ValueError("No proper-training rows found in split_manifest.csv")
    if len(train_rows) > max_examples:
        train_rows = train_rows.sample(n=max_examples, random_state=seed)

    signals = []
    for row in train_rows.itertuples(index=False):
        signal, _ = wfdb.rdsamp(os.path.join(base_path, row.filename_lr))
        signals.append(np.asarray(signal, dtype=np.float32))
    x = np.stack(signals)
    x -= mean.astype(np.float32)
    x /= std.astype(np.float32)
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_path",
        default="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    )
    parser.add_argument(
        "--artifact_dir", default=os.path.join("results", "split_conformal")
    )
    parser.add_argument("--output_dir", default=os.path.join("edge", "model"))
    parser.add_argument("--max_representative", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = os.path.join(args.artifact_dir, "split_conformal_model.h5")
    artifact_manifest_path = os.path.join(args.artifact_dir, "artifact_manifest.json")
    split_manifest_path = os.path.join(args.artifact_dir, "split_manifest.csv")
    mean_path = os.path.join(args.artifact_dir, "lead_mean.npy")
    std_path = os.path.join(args.artifact_dir, "lead_std.npy")
    required = [model_path, artifact_manifest_path, split_manifest_path, mean_path, std_path]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing authoritative artifacts:\n- " + "\n- ".join(missing))

    with open(artifact_manifest_path, encoding="utf-8") as handle:
        artifact_manifest = json.load(handle)
    actual_model_hash = sha256_file(model_path)
    expected_model_hash = artifact_manifest.get("h5_model_sha256")
    if expected_model_hash and actual_model_hash != expected_model_hash:
        raise RuntimeError("The H5 model hash does not match artifact_manifest.json")

    mean = np.load(mean_path)
    std = np.load(std_path)
    representative_x = load_representative_signals(
        args.base_path,
        split_manifest_path,
        mean,
        std,
        args.max_representative,
        args.seed,
    )

    model = tf.keras.models.load_model(model_path)
    os.makedirs(args.output_dir, exist_ok=True)
    tflite_path = os.path.join(args.output_dir, "model_int8.tflite")

    def representative_dataset():
        for i in range(len(representative_x)):
            yield [representative_x[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as handle:
        handle.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    export_manifest = {
        "source_model_path": model_path,
        "source_model_sha256": actual_model_hash,
        "source_split_manifest_path": split_manifest_path,
        "source_split_manifest_sha256": sha256_file(split_manifest_path),
        "representative_split": "proper training only",
        "representative_examples": int(len(representative_x)),
        "seed": int(args.seed),
        "tflite_path": tflite_path,
        "tflite_sha256": sha256_file(tflite_path),
        "tflite_bytes": int(os.path.getsize(tflite_path)),
        "input_dtype": str(input_details["dtype"]),
        "input_shape": [int(value) for value in input_details["shape"]],
        "input_scale": float(input_details["quantization"][0]),
        "input_zero_point": int(input_details["quantization"][1]),
        "output_dtype": str(output_details["dtype"]),
        "output_shape": [int(value) for value in output_details["shape"]],
        "output_scale": float(output_details["quantization"][0]),
        "output_zero_point": int(output_details["quantization"][1]),
    }
    with open(
        os.path.join(args.output_dir, "model_int8_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(export_manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(export_manifest, indent=2))


if __name__ == "__main__":
    main()
