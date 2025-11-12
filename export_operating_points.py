import os
import ast
import numpy as np
import pandas as pd
import wfdb
from sklearn.metrics import roc_curve
import tensorflow as tf


def parse_metadata(base_path):
    meta = pd.read_csv(os.path.join(base_path, 'ptbxl_database.csv'))
    scp = pd.read_csv(os.path.join(base_path, 'scp_statements.csv'), index_col=0)
    scp = scp[scp.diagnostic_class.isin(['MI', 'NORM'])]

    def to_class(s):
        try:
            d = ast.literal_eval(s)
        except Exception:
            return None
        agg = {'MI': 0.0, 'NORM': 0.0}
        for code, w in d.items():
            if code in scp.index:
                cls = scp.loc[code].diagnostic_class
                if cls in agg:
                    agg[cls] += float(w)
        if agg['MI'] == 0.0 and agg['NORM'] == 0.0:
            return None
        return 'MI' if agg['MI'] >= agg['NORM'] else 'NORM'

    meta['diagnostic_superclass'] = meta.scp_codes.apply(to_class)
    meta = meta.dropna(subset=['diagnostic_superclass']).copy()
    meta['label'] = meta.diagnostic_superclass.apply(lambda x: 1 if x == 'MI' else 0)
    return meta


def load_split_signals(base_path, rows):
    X, y = [], []
    for _, row in rows.iterrows():
        sig, _ = wfdb.rdsamp(os.path.join(base_path, row['filename_lr']))
        X.append(sig)
        y.append(row['label'])
    return np.array(X), np.array(y)


def get_operating_points(y_true, y_score, target_specificities=(0.90, 0.95, 0.97)):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1 - fpr
    rows = []
    for spec_tgt in target_specificities:
        if np.any(spec >= spec_tgt):
            mask = spec >= spec_tgt
            idx = np.argmax(tpr[mask])
            sens = float(tpr[mask][idx])
            thr = float(thresholds[mask][idx])
            spec_real = float(spec[mask][idx])
        else:
            sens, thr, spec_real = float('nan'), float('nan'), float('nan')
        rows.append({'target_specificity': spec_tgt, 'threshold': thr, 'achieved_specificity': spec_real, 'sensitivity': sens})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    base_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    meta = parse_metadata(base_path)

    train_rows = meta[meta['strat_fold'].isin(range(1, 9))]
    test_rows = meta[meta['strat_fold'] == 10]

    # Improved model: recompute mean/std from train folds (as in training) and evaluate
    print('Loading improved model and computing operating points...')
    X_train, y_train = load_split_signals(base_path, train_rows)
    X_test, y_test = load_split_signals(base_path, test_rows)

    mean_train = X_train.mean(axis=(0, 1), keepdims=True)
    std_train = X_train.std(axis=(0, 1), keepdims=True)
    std_train = np.where(std_train < 1e-8, 1.0, std_train)

    X_test_norm = (X_test - mean_train) / std_train

    improved_model = tf.keras.models.load_model('improved_baseline_model.h5')
    y_score_improved = improved_model.predict(X_test_norm, verbose=0).flatten()
    df_imp = get_operating_points(y_test, y_score_improved)
    os.makedirs('results', exist_ok=True)
    df_imp.to_csv('results/operating_points_improved.csv', index=False)
    print('Saved results/operating_points_improved.csv')

    # Split-conformal model: use saved mean/std
    print('Loading split-conformal model and computing operating points...')
    art_dir = os.path.join('results', 'split_conformal')
    mean_sc = np.load(os.path.join(art_dir, 'lead_mean.npy'))
    std_sc = np.load(os.path.join(art_dir, 'lead_std.npy'))
    X_test_sc, y_test_sc = X_test, y_test  # reuse raw test
    X_test_sc = (X_test_sc - mean_sc) / std_sc

    split_model = tf.keras.models.load_model(os.path.join(art_dir, 'split_conformal_model.h5'))
    y_score_sc = split_model.predict(X_test_sc, verbose=0).flatten()
    df_sc = get_operating_points(y_test_sc, y_score_sc)
    df_sc.to_csv('results/operating_points_split_conformal.csv', index=False)
    print('Saved results/operating_points_split_conformal.csv')
