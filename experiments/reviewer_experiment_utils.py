"""Shared utilities for the R4 reviewer-requested experiments.

The functions in this module deliberately separate model probabilities from
conformal calibration so that FP32-calibrated and INT8-calibrated workflows can
be compared with the same test-time INT8 predictor.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def set_global_determinism(seed: int) -> None:
    """Best-effort deterministic setup for NumPy, Python, and TensorFlow."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        # Older TensorFlow builds may not expose deterministic ops.
        pass


def _quantize_array(x: np.ndarray, scale: float, zero_point: int, dtype: np.dtype) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"Invalid TFLite quantization scale: {scale}")
    info = np.iinfo(dtype)
    q = np.rint(x / scale + zero_point)
    return np.clip(q, info.min, info.max).astype(dtype)


def predict_tflite(model_path: str, x: np.ndarray) -> np.ndarray:
    """Run a TFLite model and return dequantized scalar probabilities.

    The exported embedded model uses batch size one. This routine therefore
    invokes the interpreter sample-by-sample while preserving the exact input
    and output quantization parameters stored in the TFLite file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"TFLite model not found: {model_path}. Run "
            "`python edge/export_tflite_micro_model.py` first."
        )
    if x.ndim != 3:
        raise ValueError(f"Expected X with shape (n, time, leads); got {x.shape}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    expected = tuple(int(v) for v in input_details["shape"][1:])
    if tuple(x.shape[1:]) != expected:
        raise ValueError(
            f"Input shape mismatch: TFLite expects {expected}, received {x.shape[1:]}"
        )

    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]
    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    probabilities = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        sample = x[i : i + 1].astype(np.float32, copy=False)
        if np.issubdtype(input_dtype, np.integer):
            sample = _quantize_array(sample, float(in_scale), int(in_zero), input_dtype)
        else:
            sample = sample.astype(input_dtype, copy=False)

        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details["index"]).reshape(-1)[0]

        if np.issubdtype(output_dtype, np.integer):
            if out_scale <= 0:
                raise ValueError(f"Invalid output quantization scale: {out_scale}")
            value = (float(raw) - float(out_zero)) * float(out_scale)
        else:
            value = float(raw)
        probabilities[i] = np.clip(value, 0.0, 1.0)

    return probabilities


def convert_full_int8(
    model: tf.keras.Model,
    representative_x: np.ndarray,
    output_path: str,
    max_representative: int = 2000,
    seed: int = 42,
) -> str:
    """Convert a Keras model to a fully integer TFLite model."""
    if representative_x.ndim != 3 or len(representative_x) == 0:
        raise ValueError("representative_x must be a non-empty 3-D array")

    rng = np.random.default_rng(seed)
    if len(representative_x) > max_representative:
        indices = rng.choice(len(representative_x), size=max_representative, replace=False)
        representative_x = representative_x[indices]

    def representative_dataset() -> Iterable[list[np.ndarray]]:
        for i in range(len(representative_x)):
            yield [representative_x[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as handle:
        handle.write(tflite_model)
    return output_path


def mondrian_prediction_sets(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    test_probabilities: np.ndarray,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct binary Mondrian split-conformal prediction sets.

    Returns
    -------
    included : ndarray of shape (n_test, 2)
        Boolean membership matrix. Column 0 is Normal and column 1 is MI.
    p_values : ndarray of shape (n_test, 2)
        Class-conditional conformal p-values.
    """
    calibration_probabilities = np.asarray(calibration_probabilities, dtype=float).reshape(-1)
    calibration_labels = np.asarray(calibration_labels, dtype=int).reshape(-1)
    test_probabilities = np.asarray(test_probabilities, dtype=float).reshape(-1)

    if len(calibration_probabilities) != len(calibration_labels):
        raise ValueError("Calibration probability and label lengths differ")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between 0 and 1")

    normal_scores = calibration_probabilities[calibration_labels == 0]
    mi_scores = (1.0 - calibration_probabilities)[calibration_labels == 1]
    if len(normal_scores) == 0 or len(mi_scores) == 0:
        raise ValueError("Both classes must be represented in the calibration set")

    p_values = np.empty((len(test_probabilities), 2), dtype=np.float64)
    for i, probability_mi in enumerate(test_probabilities):
        score_normal = probability_mi
        score_mi = 1.0 - probability_mi
        p_values[i, 0] = (np.count_nonzero(normal_scores >= score_normal) + 1) / (
            len(normal_scores) + 1
        )
        p_values[i, 1] = (np.count_nonzero(mi_scores >= score_mi) + 1) / (
            len(mi_scores) + 1
        )

    included = p_values > epsilon
    return included, p_values


def evaluate_prediction_sets(included: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute overall and class-conditional conformal metrics."""
    included = np.asarray(included, dtype=bool)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    if included.shape != (len(y_true), 2):
        raise ValueError(f"Expected membership shape {(len(y_true), 2)}, got {included.shape}")

    set_sizes = included.sum(axis=1)
    row_index = np.arange(len(y_true))
    covered = included[row_index, y_true]

    metrics: Dict[str, float] = {
        "n_test": int(len(y_true)),
        "miscoverage_overall": float(1.0 - covered.mean()),
        "coverage_overall": float(covered.mean()),
        "avg_set_size_overall": float(set_sizes.mean()),
        "singleton_rate_overall": float(np.mean(set_sizes == 1)),
        "empty_set_rate_overall": float(np.mean(set_sizes == 0)),
        "doubleton_rate_overall": float(np.mean(set_sizes == 2)),
    }

    for label, name in ((0, "normal"), (1, "mi")):
        mask = y_true == label
        if not np.any(mask):
            metrics[f"n_test_{name}"] = 0
            metrics[f"miscoverage_{name}"] = float("nan")
            metrics[f"coverage_{name}"] = float("nan")
            metrics[f"avg_set_size_{name}"] = float("nan")
            metrics[f"singleton_rate_{name}"] = float("nan")
            metrics[f"empty_set_rate_{name}"] = float("nan")
            continue
        metrics[f"n_test_{name}"] = int(mask.sum())
        metrics[f"miscoverage_{name}"] = float(1.0 - covered[mask].mean())
        metrics[f"coverage_{name}"] = float(covered[mask].mean())
        metrics[f"avg_set_size_{name}"] = float(set_sizes[mask].mean())
        metrics[f"singleton_rate_{name}"] = float(np.mean(set_sizes[mask] == 1))
        metrics[f"empty_set_rate_{name}"] = float(np.mean(set_sizes[mask] == 0))

    singleton_mask = set_sizes == 1
    if np.any(singleton_mask):
        singleton_predictions = np.argmax(included[singleton_mask], axis=1)
        metrics["singleton_accuracy"] = float(
            np.mean(singleton_predictions == y_true[singleton_mask])
        )
    else:
        metrics["singleton_accuracy"] = float("nan")
    return metrics


def classification_metrics(probabilities: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def encode_sets(included: np.ndarray) -> list[str]:
    """Encode membership rows as {}, {Normal}, {MI}, or {Normal,MI}."""
    labels = []
    for normal, mi in np.asarray(included, dtype=bool):
        values = []
        if normal:
            values.append("Normal")
        if mi:
            values.append("MI")
        labels.append("{" + ",".join(values) + "}")
    return labels


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy values and non-finite floats for JSON output."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
