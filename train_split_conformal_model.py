import os
import ast
import json
import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

# Reproducibility
try:
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass


def parse_labels(metadata_path, scp_statements_path):
    metadata = pd.read_csv(metadata_path)
    scp_statements = pd.read_csv(scp_statements_path, index_col=0)
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
    return metadata


def load_signals_for_rows(base_path, rows):
    X, y = [], []
    for _, row in rows.iterrows():
        filename = os.path.join(base_path, row['filename_lr'])
        signal, _ = wfdb.rdsamp(filename)
        X.append(signal)
        y.append(row['label'])
    return np.array(X), np.array(y)


def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 5, activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


if __name__ == '__main__':
    dataset_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    metadata_path = os.path.join(dataset_path, 'ptbxl_database.csv')
    scp_statements_path = os.path.join(dataset_path, 'scp_statements.csv')

    print('Loading metadata and assigning labels...')
    metadata = parse_labels(metadata_path, scp_statements_path)

    # Folds: train pool 1-8, val=9, test=10
    train_pool = metadata[metadata['strat_fold'].isin(range(1, 9))]
    val_rows = metadata[metadata['strat_fold'] == 9]
    test_rows = metadata[metadata['strat_fold'] == 10]

    print('Loading signals for train pool, val, and test...')
    X_train_pool, y_train_pool = load_signals_for_rows(dataset_path, train_pool)
    X_val, y_val = load_signals_for_rows(dataset_path, val_rows)
    X_test, y_test = load_signals_for_rows(dataset_path, test_rows)

    print('Splitting train pool into train_proper and calibration...')
    X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
        X_train_pool, y_train_pool, test_size=0.2, random_state=42, stratify=y_train_pool
    )

    print(f'train_proper: {X_train_proper.shape}, calibration: {X_cal.shape}')

    # Per-lead normalization using train_proper only
    lead_mean = X_train_proper.mean(axis=(0, 1), keepdims=True)
    lead_std = X_train_proper.std(axis=(0, 1), keepdims=True)
    lead_std = np.where(lead_std < 1e-8, 1.0, lead_std)

    X_train_proper = (X_train_proper - lead_mean) / lead_std
    X_cal = (X_cal - lead_mean) / lead_std
    X_val = (X_val - lead_mean) / lead_std
    X_test = (X_test - lead_mean) / lead_std

    # Model
    input_shape = (X_train_proper.shape[1], X_train_proper.shape[2])
    model = create_model(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])

    classes = np.unique(y_train_proper)
    class_weights_arr = compute_class_weight('balanced', classes=classes, y=y_train_proper)
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}

    print('Training split-conformal model (train_proper only)...')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_proper, y_train_proper,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[es],
        class_weight=class_weights,
        verbose=1
    )

    print('Evaluating on test set...')
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test: loss={test_loss:.4f}, acc={test_acc:.4f}, auc={test_auc:.4f}')

    # Save artifacts
    out_dir = os.path.join('results', 'split_conformal')
    os.makedirs(out_dir, exist_ok=True)

    model.save(os.path.join(out_dir, 'split_conformal_model.h5'))
    np.save(os.path.join(out_dir, 'lead_mean.npy'), lead_mean)
    np.save(os.path.join(out_dir, 'lead_std.npy'), lead_std)
    np.save(os.path.join(out_dir, 'X_cal.npy'), X_cal)
    np.save(os.path.join(out_dir, 'y_cal.npy'), y_cal)

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'class_weights': {str(k): float(v) for k, v in class_weights.items()}
        }, f, indent=2)

    print(f'Artifacts saved to {out_dir}')
