"""Compute finite-sample Mondrian thresholds from INT8 calibration outputs.

``X_cal.npy`` is already normalized by ``train_split_conformal_model.py``.  This
script intentionally does not normalize it again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import tensorflow as tf


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_sample_quantile(scores: np.ndarray, epsilon: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    if len(scores) == 0:
        raise ValueError("Calibration scores are empty")
    rank = int(np.ceil((len(scores) + 1) * (1.0 - epsilon)))
    index = int(np.clip(rank - 1, 0, len(scores) - 1))
    return float(scores[index])


def predict_int8(model_path: str, x: np.ndarray) -> tuple[np.ndarray, dict]:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero = input_details["quantization"]
    output_scale, output_zero = output_details["quantization"]
    if input_scale <= 0 or output_scale <= 0:
        raise ValueError("The model does not expose valid integer quantization parameters")

    probabilities = np.empty(len(x), dtype=np.float64)
    for i in range(len(x)):
        quantized = np.rint(x[i : i + 1] / input_scale + input_zero)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quantized)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details["index"]).reshape(-1)[0]
        probabilities[i] = np.clip(
            (float(raw) - float(output_zero)) * float(output_scale), 0.0, 1.0
        )

    quantization = {
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero),
    }
    return probabilities, quantization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact_dir", default=os.path.join("results", "split_conformal")
    )
    parser.add_argument("--model_dir", default=os.path.join("edge", "model"))
    parser.add_argument("--epsilon", type=float, default=0.10)
    args = parser.parse_args()

    x_cal_path = os.path.join(args.artifact_dir, "X_cal.npy")
    y_cal_path = os.path.join(args.artifact_dir, "y_cal.npy")
    mean_path = os.path.join(args.artifact_dir, "lead_mean.npy")
    std_path = os.path.join(args.artifact_dir, "lead_std.npy")
    artifact_manifest_path = os.path.join(args.artifact_dir, "artifact_manifest.json")
    tflite_path = os.path.join(args.model_dir, "model_int8.tflite")
    tflite_manifest_path = os.path.join(args.model_dir, "model_int8_manifest.json")
    required = [
        x_cal_path,
        y_cal_path,
        mean_path,
        std_path,
        artifact_manifest_path,
        tflite_path,
        tflite_manifest_path,
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n- " + "\n- ".join(missing))

    with open(artifact_manifest_path, encoding="utf-8") as handle:
        artifact_manifest = json.load(handle)
    if artifact_manifest.get("x_cal_normalized") is not True:
        raise RuntimeError("artifact_manifest.json must declare x_cal_normalized=true")

    with open(tflite_manifest_path, encoding="utf-8") as handle:
        tflite_manifest = json.load(handle)
    if sha256_file(tflite_path) != tflite_manifest.get("tflite_sha256"):
        raise RuntimeError("INT8 TFLite hash does not match model_int8_manifest.json")
    if (
        artifact_manifest.get("h5_model_sha256")
        and tflite_manifest.get("source_model_sha256")
        != artifact_manifest.get("h5_model_sha256")
    ):
        raise RuntimeError("INT8 model was not exported from the authoritative H5 model")

    x_cal = np.load(x_cal_path).astype(np.float32, copy=False)
    y_cal = np.load(y_cal_path).astype(np.int64, copy=False)
    mean = np.load(mean_path)
    std = np.load(std_path)
    probabilities, quantization = predict_int8(tflite_path, x_cal)

    normal_scores = probabilities[y_cal == 0]
    mi_scores = (1.0 - probabilities)[y_cal == 1]
    tau_normal = finite_sample_quantile(normal_scores, args.epsilon)
    tau_mi = finite_sample_quantile(mi_scores, args.epsilon)

    params = {
        "epsilon": float(args.epsilon),
        "tau_norm": tau_normal,
        "tau_mi": tau_mi,
        "n_cal_normal": int(len(normal_scores)),
        "n_cal_mi": int(len(mi_scores)),
        "calibration_precision": "INT8 TFLite",
        "x_cal_normalized_once": True,
        "source_model_sha256": tflite_manifest.get("source_model_sha256"),
        "tflite_sha256": tflite_manifest.get("tflite_sha256"),
        "lead_mean": mean.reshape(-1).tolist(),
        "lead_std": std.reshape(-1).tolist(),
        **quantization,
    }
    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "cp_params.json"), "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)

    header = f"""#ifndef CP_PARAMS_H
#define CP_PARAMS_H

#include <stdint.h>

#define CP_EPSILON {args.epsilon:.8f}f
#define CP_TAU_NORM {tau_normal:.8f}f
#define CP_TAU_MI {tau_mi:.8f}f
#define CP_IN_SCALE {quantization['input_scale']:.10f}f
#define CP_IN_ZP {quantization['input_zero_point']}
#define CP_OUT_SCALE {quantization['output_scale']:.10f}f
#define CP_OUT_ZP {quantization['output_zero_point']}

static const float CP_LEAD_MEAN[12] = {{ {', '.join(f'{value:.8f}f' for value in mean.reshape(-1))} }};
static const float CP_LEAD_STD[12] = {{ {', '.join(f'{value:.8f}f' for value in std.reshape(-1))} }};

#endif  // CP_PARAMS_H
"""
    with open(os.path.join(args.model_dir, "cp_params.h"), "w", encoding="utf-8") as handle:
        handle.write(header)

    print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
