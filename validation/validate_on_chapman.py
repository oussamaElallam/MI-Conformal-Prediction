import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
from validation.chapman_loader import load_chapman_data

# Paths
ART_DIR = os.path.join('results', 'split_conformal')
MODEL_PATH = os.path.join(ART_DIR, 'split_conformal_model.h5')
EPSILON = 0.1


def get_conformal_sets(probs, cal_scores_norm, cal_scores_mi, epsilon):
    sets = []
    for p in probs:
        s_norm = p
        s_mi = 1.0 - p
        p_norm = (np.sum(cal_scores_norm >= s_norm) + 1) / (len(cal_scores_norm) + 1)
        p_mi = (np.sum(cal_scores_mi >= s_mi) + 1) / (len(cal_scores_mi) + 1)
        s = set()
        if p_norm > epsilon: s.add(0)
        if p_mi > epsilon: s.add(1)
        sets.append(s)
    return sets


def evaluate_sets(sets, y_true):
    miscoverage = np.mean([y_true[i] not in sets[i] for i in range(len(y_true))])
    avg_set_size = np.mean([len(s) for s in sets])
    singleton_coverage = np.mean([len(s) == 1 for s in sets])
    return {
        'miscoverage': miscoverage,
        'avg_set_size': avg_set_size,
        'singleton_coverage': singleton_coverage
    }


if __name__ == '__main__':
    print('Loading model and PTB-XL calibration data...')
    model = tf.keras.models.load_model(MODEL_PATH)
    X_cal = np.load(os.path.join(ART_DIR, 'X_cal.npy'))
    y_cal = np.load(os.path.join(ART_DIR, 'y_cal.npy'))
    mean_ptbxl = np.load(os.path.join(ART_DIR, 'lead_mean.npy'))
    std_ptbxl = np.load(os.path.join(ART_DIR, 'lead_std.npy'))

    # Compute conformal thresholds from PTB-XL calibration set
    cal_probs = model.predict(X_cal, verbose=0).flatten()
    cal_scores_norm = cal_probs[y_cal == 0]
    cal_scores_mi = (1.0 - cal_probs)[y_cal == 1]

    print('Loading and filtering Chapman-Shaoxing dataset...')
    X_chapman, y_chapman = load_chapman_data()

    if len(X_chapman) == 0:
        print('No MI or Normal records found in Chapman-Shaoxing dataset. Exiting.')
        exit()

    print('Normalizing Chapman data with PTB-XL statistics...')
    X_chapman_norm = (X_chapman - mean_ptbxl) / std_ptbxl

    print('Evaluating on Chapman-Shaoxing data...')
    chapman_probs = model.predict(X_chapman_norm, verbose=0).flatten()
    y_pred = (chapman_probs > 0.5).astype(int)

    # Standard metrics
    accuracy = accuracy_score(y_chapman, y_pred)
    roc_auc = roc_auc_score(y_chapman, chapman_probs)
    print(f'Chapman Test Accuracy: {accuracy:.4f}')
    print(f'Chapman Test ROC-AUC: {roc_auc:.4f}')

    # Conformal metrics
    cp_sets = get_conformal_sets(chapman_probs, cal_scores_norm, cal_scores_mi, EPSILON)
    cp_results = evaluate_sets(cp_sets, y_chapman)
    print(f'Conformal Results on Chapman: {cp_results}')

    # Save results
    results = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'conformal_miscoverage': cp_results['miscoverage'],
        'conformal_avg_set_size': cp_results['avg_set_size'],
        'conformal_singleton_coverage': cp_results['singleton_coverage'],
        'num_samples': len(X_chapman),
        'num_mi': int(np.sum(y_chapman)),
        'num_normal': len(y_chapman) - int(np.sum(y_chapman))
    }
    os.makedirs('results/validation', exist_ok=True)
    pd.DataFrame([results]).to_csv('results/validation/chapman_validation_results.csv', index=False)
    print('Saved validation results to results/validation/chapman_validation_results.csv')
