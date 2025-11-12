import os
import ast
import json
import math
import numpy as np
import pandas as pd
import wfdb
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


def load_signals(base_path, rows):
    X, y = [], []
    for _, row in rows.iterrows():
        sig, _ = wfdb.rdsamp(os.path.join(base_path, row['filename_lr']))
        X.append(sig)
        y.append(row['label'])
    return np.array(X), np.array(y)


def two_sided_binom_pvalue(k, n):
    # McNemar exact two-sided p-value using binomial test with p=0.5
    # p = 2 * min(P[X <= k], P[X >= k]) where X ~ Binomial(n, 0.5)
    # Compute lower tail
    prob = 0.0
    for i in range(0, k + 1):
        prob += math.comb(n, i) / (2 ** n)
    p = 2 * min(prob, 1 - prob)
    return min(1.0, max(0.0, p))


if __name__ == '__main__':
    base_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

    meta = parse_metadata(base_path)
    train_rows = meta[meta['strat_fold'].isin(range(1, 9))]
    test_rows = meta[meta['strat_fold'] == 10]

    X_train, y_train = load_signals(base_path, train_rows)
    X_test, y_test = load_signals(base_path, test_rows)

    # Normalization for improved model
    mean_impr = X_train.mean(axis=(0, 1), keepdims=True)
    std_impr = X_train.std(axis=(0, 1), keepdims=True)
    std_impr = np.where(std_impr < 1e-8, 1.0, std_impr)
    X_test_impr = (X_test - mean_impr) / std_impr

    # Normalization for resnet baseline (use same train folds)
    mean_res = mean_impr
    std_res = std_impr
    X_test_res = (X_test - mean_res) / std_res

    # Load models
    improved_model = tf.keras.models.load_model('improved_baseline_model.h5')
    resnet_model = tf.keras.models.load_model(os.path.join('results', 'resnet1d', 'resnet1d_model.h5'))

    y_score_impr = improved_model.predict(X_test_impr, verbose=0).flatten()
    y_pred_impr = (y_score_impr > 0.5).astype(int)

    y_score_res = resnet_model.predict(X_test_res, verbose=0).flatten()
    y_pred_res = (y_score_res > 0.5).astype(int)

    # McNemar contingency
    correct_impr = (y_pred_impr == y_test)
    correct_res = (y_pred_res == y_test)

    n01 = int(np.sum((correct_impr == False) & (correct_res == True)))  # impr wrong, res right
    n10 = int(np.sum((correct_impr == True) & (correct_res == False)))  # impr right, res wrong
    n00 = int(np.sum((correct_impr == False) & (correct_res == False)))
    n11 = int(np.sum((correct_impr == True) & (correct_res == True)))

    n = n01 + n10
    p_value = two_sided_binom_pvalue(min(n01, n10), n) if n > 0 else 1.0

    out_dir = os.path.join('results', 'significance')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'mcnemar.json'), 'w') as f:
        json.dump({
            'n01_improved_wrong_resnet_right': n01,
            'n10_improved_right_resnet_wrong': n10,
            'n00_both_wrong': n00,
            'n11_both_right': n11,
            'p_value_two_sided': p_value,
            'n_discordant': n,
            'accuracy_improved': float(np.mean(correct_impr)),
            'accuracy_resnet': float(np.mean(correct_res))
        }, f, indent=2)

    print(f"Saved McNemar results to {out_dir}/mcnemar.json (p={p_value:.4g})")
