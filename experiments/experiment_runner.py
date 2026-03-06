"""
Unified experiment runner for BSPC experiments.

Usage:
    from experiments.experiment_runner import run_experiment, FoldConfig

    config = FoldConfig(train_folds=[1,2,3,4,5,6,7,8], val_fold=9, test_fold=10)
    results = run_experiment(
        base_path='ptb-xl-...',
        fold_config=config,
        model_builder=create_lightweight_cnn,
        model_name='LightweightCNN',
        seed=42
    )
"""

import os
import ast
import json
import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf
from dataclasses import dataclass, field, asdict
from typing import List, Callable, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    accuracy_score, classification_report, confusion_matrix
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class FoldConfig:
    """Defines which PTB-XL folds map to train/val/test."""
    train_folds: List[int] = field(default_factory=lambda: list(range(1, 9)))
    val_fold: int = 9
    test_fold: int = 10
    cal_fraction: float = 0.2  # fraction of train pool held out for calibration
    cal_random_state: int = 42

    @property
    def name(self) -> str:
        return f"train{''.join(map(str, self.train_folds))}_val{self.val_fold}_test{self.test_fold}"


# ─────────────────────────────────────────────
# Data Loading (shared across all experiments)
# ─────────────────────────────────────────────

_metadata_cache: Dict[str, pd.DataFrame] = {}
_signals_cache: Dict[str, np.ndarray] = {}


def parse_labels(base_path: str) -> pd.DataFrame:
    """Parse PTB-XL metadata and assign binary MI/Normal labels.
    Cached so repeated calls with same base_path are free."""
    if base_path in _metadata_cache:
        return _metadata_cache[base_path].copy()

    metadata = pd.read_csv(os.path.join(base_path, 'ptbxl_database.csv'))
    scp_statements = pd.read_csv(os.path.join(base_path, 'scp_statements.csv'), index_col=0)
    scp_statements = scp_statements[scp_statements.diagnostic_class.isin(['MI', 'NORM'])]

    def get_class(scp_codes_str):
        try:
            scp_codes = ast.literal_eval(scp_codes_str)
        except Exception:
            return None
        agg = {'MI': 0.0, 'NORM': 0.0}
        for code, weight in scp_codes.items():
            if code in scp_statements.index:
                cls = scp_statements.loc[code].diagnostic_class
                if cls in agg:
                    agg[cls] += float(weight)
        if agg['MI'] == 0.0 and agg['NORM'] == 0.0:
            return None
        return 'MI' if agg['MI'] >= agg['NORM'] else 'NORM'

    metadata['diagnostic_superclass'] = metadata.scp_codes.apply(get_class)
    metadata.dropna(subset=['diagnostic_superclass'], inplace=True)
    metadata['label'] = metadata.diagnostic_superclass.apply(lambda x: 1 if x == 'MI' else 0)
    _metadata_cache[base_path] = metadata
    return metadata.copy()


def load_signals_by_fold(base_path: str, metadata: pd.DataFrame,
                         folds: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Load ECG signals for records belonging to specified folds."""
    rows = metadata[metadata['strat_fold'].isin(folds)]
    X, y = [], []
    for _, row in rows.iterrows():
        sig, _ = wfdb.rdsamp(os.path.join(base_path, row['filename_lr']))
        X.append(sig)
        y.append(row['label'])
    return np.array(X), np.array(y)


def prepare_data(base_path: str, fold_config: FoldConfig):
    """
    Full data preparation pipeline.
    Returns: X_train, y_train, X_cal, y_cal, X_val, y_val, X_test, y_test,
             class_weights, normalization_stats
    """
    metadata = parse_labels(base_path)

    # Load by fold
    X_train_pool, y_train_pool = load_signals_by_fold(base_path, metadata, fold_config.train_folds)
    X_val, y_val = load_signals_by_fold(base_path, metadata, [fold_config.val_fold])
    X_test, y_test = load_signals_by_fold(base_path, metadata, [fold_config.test_fold])

    # Split train pool → train_proper + calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_pool, y_train_pool,
        test_size=fold_config.cal_fraction,
        random_state=fold_config.cal_random_state,
        stratify=y_train_pool
    )

    # Per-lead z-score normalization (from train_proper only)
    lead_mean = X_train.mean(axis=(0, 1), keepdims=True)
    lead_std = X_train.std(axis=(0, 1), keepdims=True)
    lead_std = np.where(lead_std < 1e-8, 1.0, lead_std)

    X_train = (X_train - lead_mean) / lead_std
    X_cal = (X_cal - lead_mean) / lead_std
    X_val = (X_val - lead_mean) / lead_std
    X_test = (X_test - lead_mean) / lead_std

    # Class weights
    classes = np.unique(y_train)
    cw_arr = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw_arr)}

    norm_stats = {'lead_mean': lead_mean, 'lead_std': lead_std}

    return X_train, y_train, X_cal, y_cal, X_val, y_val, X_test, y_test, class_weights, norm_stats


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_model(model, X_train, y_train, X_val, y_val,
                class_weights, epochs=30, batch_size=32, patience=5):
    """Train a compiled Keras model with early stopping."""
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        class_weight=class_weights,
        verbose=0  # silent for batch runs
    )
    return history


# ─────────────────────────────────────────────
# Classification Metrics
# ─────────────────────────────────────────────

def compute_classification_metrics(model, X_test, y_test) -> Dict[str, Any]:
    """Compute standard discriminative metrics."""
    y_score = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_score > 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_score))
    pr_auc = float(average_precision_score(y_test, y_score))
    accuracy = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    # Sensitivity at 95% specificity
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    spec = 1 - fpr
    if np.any(spec >= 0.95):
        mask = spec >= 0.95
        sens_at_95spec = float(tpr[mask][np.argmax(tpr[mask])])
    else:
        sens_at_95spec = float('nan')

    # Bootstrap CI for ROC-AUC
    rng = np.random.default_rng(42)
    boot_aucs = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_test), len(y_test))
        try:
            boot_aucs.append(roc_auc_score(y_test[idx], y_score[idx]))
        except ValueError:
            pass
    auc_lo, auc_hi = np.percentile(boot_aucs, [2.5, 97.5]) if boot_aucs else (np.nan, np.nan)

    return {
        'roc_auc': roc_auc,
        'roc_auc_ci_lo': float(auc_lo),
        'roc_auc_ci_hi': float(auc_hi),
        'pr_auc': pr_auc,
        'accuracy': accuracy,
        'balanced_accuracy': bal_acc,
        'sensitivity_at_95spec': sens_at_95spec,
        'n_test': len(y_test),
        'n_mi_test': int(y_test.sum()),
        'n_normal_test': int(len(y_test) - y_test.sum()),
    }


# ─────────────────────────────────────────────
# Mondrian Conformal Prediction
# ─────────────────────────────────────────────

def compute_conformal_scores(model, X_cal, y_cal):
    """Compute class-conditional (Mondrian) nonconformity scores on calibration set."""
    cal_probs = model.predict(X_cal, verbose=0).flatten()
    # For Normal hypothesis: score = p(MI)
    scores_norm = cal_probs[y_cal == 0]
    # For MI hypothesis: score = 1 - p(MI)
    scores_mi = (1.0 - cal_probs)[y_cal == 1]
    return scores_norm, scores_mi


def conformal_predict(prob_mi: float, scores_norm: np.ndarray,
                      scores_mi: np.ndarray, epsilon: float) -> set:
    """Mondrian conformal prediction for a single sample."""
    # p-value for Normal hypothesis
    s_norm = prob_mi
    p_val_norm = (np.sum(scores_norm >= s_norm) + 1) / (len(scores_norm) + 1)

    # p-value for MI hypothesis
    s_mi = 1.0 - prob_mi
    p_val_mi = (np.sum(scores_mi >= s_mi) + 1) / (len(scores_mi) + 1)

    pred_set = set()
    if p_val_norm > epsilon:
        pred_set.add(0)  # Normal
    if p_val_mi > epsilon:
        pred_set.add(1)  # MI
    return pred_set


def compute_conformal_metrics(model, X_test, y_test,
                              scores_norm, scores_mi,
                              epsilon=0.1) -> Dict[str, Any]:
    """Compute conformal prediction metrics at a given epsilon."""
    probs = model.predict(X_test, verbose=0).flatten()

    pred_sets = [conformal_predict(p, scores_norm, scores_mi, epsilon) for p in probs]
    set_sizes = np.array([len(s) for s in pred_sets], dtype=float)

    # Overall metrics
    covered = [y_test[i] in pred_sets[i] for i in range(len(y_test))]
    miscoverage = 1.0 - np.mean(covered)
    avg_set_size = float(set_sizes.mean())
    singleton_rate = float(np.mean(set_sizes == 1))

    # Per-class
    idx_norm = np.where(y_test == 0)[0]
    idx_mi = np.where(y_test == 1)[0]

    miscov_norm = 1.0 - np.mean([0 in pred_sets[i] for i in idx_norm]) if len(idx_norm) > 0 else np.nan
    miscov_mi = 1.0 - np.mean([1 in pred_sets[i] for i in idx_mi]) if len(idx_mi) > 0 else np.nan

    singleton_norm = float(np.mean([len(pred_sets[i]) == 1 for i in idx_norm])) if len(idx_norm) > 0 else np.nan
    singleton_mi = float(np.mean([len(pred_sets[i]) == 1 for i in idx_mi])) if len(idx_mi) > 0 else np.nan

    avg_size_norm = float(np.mean([len(pred_sets[i]) for i in idx_norm])) if len(idx_norm) > 0 else np.nan
    avg_size_mi = float(np.mean([len(pred_sets[i]) for i in idx_mi])) if len(idx_mi) > 0 else np.nan

    return {
        'epsilon': epsilon,
        'miscoverage_overall': float(miscoverage),
        'miscoverage_normal': float(miscov_norm),
        'miscoverage_mi': float(miscov_mi),
        'avg_set_size_overall': avg_set_size,
        'avg_set_size_normal': float(avg_size_norm),
        'avg_set_size_mi': float(avg_size_mi),
        'singleton_rate_overall': singleton_rate,
        'singleton_rate_normal': float(singleton_norm),
        'singleton_rate_mi': float(singleton_mi),
        'n_cal_normal': int(len(scores_norm)),
        'n_cal_mi': int(len(scores_mi)),
    }


# ─────────────────────────────────────────────
# Main Experiment Runner
# ─────────────────────────────────────────────

def run_experiment(
    base_path: str,
    fold_config: FoldConfig,
    model_builder: Callable,
    model_name: str,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 32,
    patience: int = 5,
    epsilon: float = 0.1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a complete experiment: data prep → train → evaluate classification → evaluate CP.

    Args:
        base_path:     Path to PTB-XL dataset root
        fold_config:   FoldConfig specifying train/val/test fold assignment
        model_builder: Callable(input_shape) -> compiled Keras model
        model_name:    String identifier for this architecture
        seed:          Random seed for reproducibility
        epsilon:       Conformal significance level
        verbose:       Print progress

    Returns:
        Dictionary with all results
    """
    # Set seeds
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    if verbose:
        print(f"[{model_name}] Fold config: {fold_config.name}, seed={seed}")

    # 1. Prepare data
    if verbose:
        print(f"  Loading data...")
    X_train, y_train, X_cal, y_cal, X_val, y_val, X_test, y_test, class_weights, norm_stats = \
        prepare_data(base_path, fold_config)

    if verbose:
        print(f"  Train: {len(y_train)} | Cal: {len(y_cal)} | Val: {len(y_val)} | Test: {len(y_test)}")
        print(f"  MI ratio — Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    # 2. Build and train model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = model_builder(input_shape)

    if verbose:
        print(f"  Training ({model.count_params()} params)...")
    history = train_model(model, X_train, y_train, X_val, y_val,
                          class_weights, epochs=epochs, batch_size=batch_size, patience=patience)
    n_epochs_trained = len(history.history['loss'])

    # 3. Classification metrics
    if verbose:
        print(f"  Evaluating classification...")
    clf_metrics = compute_classification_metrics(model, X_test, y_test)

    # 4. Conformal prediction metrics
    if verbose:
        print(f"  Evaluating conformal prediction (ε={epsilon})...")
    scores_norm, scores_mi = compute_conformal_scores(model, X_cal, y_cal)
    cp_metrics = compute_conformal_metrics(model, X_test, y_test, scores_norm, scores_mi, epsilon)

    if verbose:
        print(f"  ROC-AUC: {clf_metrics['roc_auc']:.4f} "
              f"({clf_metrics['roc_auc_ci_lo']:.3f}–{clf_metrics['roc_auc_ci_hi']:.3f})")
        print(f"  Miscoverage: {cp_metrics['miscoverage_overall']:.4f} (target: {epsilon})")
        print(f"  Avg set size: {cp_metrics['avg_set_size_overall']:.4f}")
        print(f"  Singleton rate: {cp_metrics['singleton_rate_overall']:.4f}")

    # 5. Assemble results
    results = {
        'model_name': model_name,
        'fold_config': asdict(fold_config),
        'seed': seed,
        'n_params': int(model.count_params()),
        'epochs_trained': n_epochs_trained,
        'data_splits': {
            'n_train': len(y_train),
            'n_cal': len(y_cal),
            'n_val': len(y_val),
            'n_test': len(y_test),
            'mi_ratio_train': float(y_train.mean()),
            'mi_ratio_test': float(y_test.mean()),
        },
        'classification': clf_metrics,
        'conformal': cp_metrics,
    }

    # Clean up to free GPU memory
    del model
    tf.keras.backend.clear_session()

    return results


def save_results(results: Dict[str, Any], output_dir: str, tag: str = ''):
    """Save experiment results as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{results['model_name']}_{results['fold_config']['name']}"
    if tag:
        fname += f"_{tag}"
    fname += ".json"
    path = os.path.join(output_dir, fname)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return path
