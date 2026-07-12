"""Unified, leakage-safe runner for architecture, ablation, and fold experiments."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from experiments.data_splitting import (
    assert_group_disjoint,
    make_group_id,
    split_summary,
    stratified_group_holdout,
)


@dataclass
class FoldConfig:
    train_folds: List[int] = field(default_factory=lambda: list(range(1, 9)))
    val_fold: int = 9
    test_fold: int = 10
    cal_fraction: float = 0.20
    cal_random_state: int = 42

    @property
    def name(self) -> str:
        train_text = "-".join(map(str, self.train_folds))
        return f"train-{train_text}_val-{self.val_fold}_test-{self.test_fold}"


_metadata_cache: Dict[str, pd.DataFrame] = {}
_signal_cache: Dict[tuple[str, str], np.ndarray] = {}


def parse_labels(base_path: str) -> pd.DataFrame:
    if base_path in _metadata_cache:
        return _metadata_cache[base_path].copy()

    metadata = pd.read_csv(os.path.join(base_path, "ptbxl_database.csv"))
    statements = pd.read_csv(os.path.join(base_path, "scp_statements.csv"), index_col=0)
    statements = statements[statements.diagnostic_class.isin(["MI", "NORM"])]

    def get_class(codes_text: str) -> str | None:
        try:
            codes = ast.literal_eval(codes_text)
        except Exception:
            return None
        totals = {"MI": 0.0, "NORM": 0.0}
        for code, weight in codes.items():
            if code in statements.index:
                diagnostic_class = statements.loc[code].diagnostic_class
                if diagnostic_class in totals:
                    totals[diagnostic_class] += float(weight)
        if totals["MI"] == 0.0 and totals["NORM"] == 0.0:
            return None
        return "MI" if totals["MI"] >= totals["NORM"] else "NORM"

    metadata["diagnostic_superclass"] = metadata.scp_codes.apply(get_class)
    metadata = metadata.dropna(subset=["diagnostic_superclass"]).copy()
    metadata["label"] = (metadata.diagnostic_superclass == "MI").astype(int)
    metadata = make_group_id(
        metadata,
        preferred_columns=("patient_id", "ecg_id"),
        output_column="group_id",
    )
    _metadata_cache[base_path] = metadata.copy()
    return metadata


def _load_signal(base_path: str, filename: str) -> np.ndarray:
    key = (base_path, filename)
    if key not in _signal_cache:
        signal, _ = wfdb.rdsamp(os.path.join(base_path, filename))
        _signal_cache[key] = np.asarray(signal, dtype=np.float32)
    return _signal_cache[key]


def load_signals_for_rows(
    base_path: str, rows: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    signals = [_load_signal(base_path, str(row.filename_lr)) for row in rows.itertuples(index=False)]
    labels = rows["label"].to_numpy(dtype=np.int64)
    return np.stack(signals), labels


def load_signals_by_fold(
    base_path: str, metadata: pd.DataFrame, folds: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    rows = metadata[metadata.strat_fold.isin(folds)].copy()
    return load_signals_for_rows(base_path, rows)


def prepare_data(base_path: str, fold_config: FoldConfig):
    metadata = parse_labels(base_path)
    train_pool_rows = metadata[metadata.strat_fold.isin(fold_config.train_folds)].copy()
    validation_rows = metadata[metadata.strat_fold == fold_config.val_fold].copy()
    test_rows = metadata[metadata.strat_fold == fold_config.test_fold].copy()
    train_rows, calibration_rows = stratified_group_holdout(
        train_pool_rows,
        group_column="group_id",
        label_column="label",
        test_size=fold_config.cal_fraction,
        random_state=fold_config.cal_random_state,
    )
    assert_group_disjoint(
        ("train", train_rows),
        ("calibration", calibration_rows),
        ("validation", validation_rows),
        ("test", test_rows),
        group_column="group_id",
    )

    x_train, y_train = load_signals_for_rows(base_path, train_rows)
    x_cal, y_cal = load_signals_for_rows(base_path, calibration_rows)
    x_val, y_val = load_signals_for_rows(base_path, validation_rows)
    x_test, y_test = load_signals_for_rows(base_path, test_rows)

    lead_mean = x_train.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    lead_std = x_train.std(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    lead_std = np.where(lead_std < 1e-8, 1.0, lead_std).astype(np.float32)
    for array in (x_train, x_cal, x_val, x_test):
        array -= lead_mean
        array /= lead_std

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(label): float(weight) for label, weight in zip(classes, weights)}
    summary = split_summary(
        [
            ("train", train_rows),
            ("calibration", calibration_rows),
            ("validation", validation_rows),
            ("test", test_rows),
        ],
        group_column="group_id",
        label_column="label",
    ).to_dict(orient="records")

    return (
        x_train,
        y_train,
        x_cal,
        y_cal,
        x_val,
        y_val,
        x_test,
        y_test,
        class_weights,
        {"lead_mean": lead_mean, "lead_std": lead_std},
        summary,
    )


def train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    class_weights,
    epochs: int = 30,
    batch_size: int = 32,
    patience: int = 5,
):
    callback = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callback],
        class_weight=class_weights,
        verbose=0,
    )


def compute_classification_metrics(model, x_test, y_test) -> Dict[str, Any]:
    scores = model.predict(x_test, verbose=0).reshape(-1)
    predictions = (scores >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, scores)
    specificity = 1.0 - fpr
    mask = specificity >= 0.95
    sensitivity_at_95 = float(np.max(tpr[mask])) if np.any(mask) else float("nan")

    rng = np.random.default_rng(42)
    bootstrapped = []
    for _ in range(1000):
        indices = rng.integers(0, len(y_test), len(y_test))
        if len(np.unique(y_test[indices])) < 2:
            continue
        bootstrapped.append(roc_auc_score(y_test[indices], scores[indices]))
    lower, upper = (
        np.percentile(bootstrapped, [2.5, 97.5])
        if bootstrapped
        else (np.nan, np.nan)
    )

    return {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "roc_auc_ci_lo": float(lower),
        "roc_auc_ci_hi": float(upper),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "sensitivity_at_95spec": sensitivity_at_95,
        "n_test": int(len(y_test)),
        "n_mi_test": int(y_test.sum()),
        "n_normal_test": int(len(y_test) - y_test.sum()),
    }


def compute_conformal_scores(model, x_cal, y_cal):
    probabilities = model.predict(x_cal, verbose=0).reshape(-1)
    scores_normal = probabilities[y_cal == 0]
    scores_mi = (1.0 - probabilities)[y_cal == 1]
    if len(scores_normal) == 0 or len(scores_mi) == 0:
        raise ValueError("Both labels are required in the calibration set")
    return scores_normal, scores_mi


def conformal_predict(
    probability_mi: float,
    scores_normal: np.ndarray,
    scores_mi: np.ndarray,
    epsilon: float,
) -> set[int]:
    p_normal = (np.count_nonzero(scores_normal >= probability_mi) + 1) / (
        len(scores_normal) + 1
    )
    p_mi = (np.count_nonzero(scores_mi >= (1.0 - probability_mi)) + 1) / (
        len(scores_mi) + 1
    )
    prediction_set: set[int] = set()
    if p_normal > epsilon:
        prediction_set.add(0)
    if p_mi > epsilon:
        prediction_set.add(1)
    return prediction_set


def compute_conformal_metrics(
    model,
    x_test,
    y_test,
    scores_normal,
    scores_mi,
    epsilon: float = 0.10,
) -> Dict[str, Any]:
    probabilities = model.predict(x_test, verbose=0).reshape(-1)
    prediction_sets = [
        conformal_predict(probability, scores_normal, scores_mi, epsilon)
        for probability in probabilities
    ]
    sizes = np.asarray([len(value) for value in prediction_sets], dtype=int)
    covered = np.asarray(
        [int(y_test[i]) in prediction_sets[i] for i in range(len(y_test))],
        dtype=bool,
    )
    normal_mask = y_test == 0
    mi_mask = y_test == 1

    return {
        "epsilon": float(epsilon),
        "coverage_overall": float(covered.mean()),
        "miscoverage_overall": float(1.0 - covered.mean()),
        "coverage_normal": float(covered[normal_mask].mean()),
        "coverage_mi": float(covered[mi_mask].mean()),
        "miscoverage_normal": float(1.0 - covered[normal_mask].mean()),
        "miscoverage_mi": float(1.0 - covered[mi_mask].mean()),
        "avg_set_size_overall": float(sizes.mean()),
        "avg_set_size_normal": float(sizes[normal_mask].mean()),
        "avg_set_size_mi": float(sizes[mi_mask].mean()),
        "singleton_rate_overall": float(np.mean(sizes == 1)),
        "singleton_rate_normal": float(np.mean(sizes[normal_mask] == 1)),
        "singleton_rate_mi": float(np.mean(sizes[mi_mask] == 1)),
        "empty_set_rate_overall": float(np.mean(sizes == 0)),
        "doubleton_rate_overall": float(np.mean(sizes == 2)),
        "n_cal_normal": int(len(scores_normal)),
        "n_cal_mi": int(len(scores_mi)),
    }


def run_experiment(
    base_path: str,
    fold_config: FoldConfig,
    model_builder: Callable,
    model_name: str,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 32,
    patience: int = 5,
    epsilon: float = 0.10,
    verbose: bool = True,
) -> Dict[str, Any]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    (
        x_train,
        y_train,
        x_cal,
        y_cal,
        x_val,
        y_val,
        x_test,
        y_test,
        class_weights,
        normalization,
        split_details,
    ) = prepare_data(base_path, fold_config)

    model = model_builder(x_train.shape[1:])
    history = train_model(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        class_weights,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )
    classification = compute_classification_metrics(model, x_test, y_test)
    scores_normal, scores_mi = compute_conformal_scores(model, x_cal, y_cal)
    conformal = compute_conformal_metrics(
        model, x_test, y_test, scores_normal, scores_mi, epsilon=epsilon
    )

    results = {
        "model_name": model_name,
        "fold_config": asdict(fold_config),
        "fold_config_name": fold_config.name,
        "seed": int(seed),
        "n_params": int(model.count_params()),
        "epochs_trained": int(len(history.history.get("loss", []))),
        "classification": classification,
        "conformal": conformal,
        "class_weights": class_weights,
        "split_summary": split_details,
        "normalization": {
            "source": "proper training only",
            "lead_mean": normalization["lead_mean"].reshape(-1).tolist(),
            "lead_std": normalization["lead_std"].reshape(-1).tolist(),
        },
    }
    if verbose:
        print(json.dumps(results, indent=2))
    return results


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_results(results: Dict[str, Any], output_dir: str, tag: str = "") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{results['model_name']}_{results['fold_config_name']}"
    if tag:
        filename += f"_{tag}"
    path = os.path.join(output_dir, filename + ".json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(results), handle, indent=2, sort_keys=True)
    return path
