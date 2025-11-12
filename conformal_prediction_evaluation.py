import pandas as pd
import numpy as np
import wfdb
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import ast

# --- Re-used Data Loading Function ---
def load_and_preprocess_data(base_path):
    """Loads and preprocesses the PTB-XL dataset using all 12 leads."""
    metadata = pd.read_csv(os.path.join(base_path, 'ptbxl_database.csv'))
    scp_statements = pd.read_csv(os.path.join(base_path, 'scp_statements.csv'), index_col=0)
    scp_statements = scp_statements[scp_statements.diagnostic_class.isin(['MI', 'NORM'])]

    def get_diagnostic_superclass(scp_codes_str):
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

    metadata['diagnostic_superclass'] = metadata.scp_codes.apply(get_diagnostic_superclass)
    metadata.dropna(subset=['diagnostic_superclass'], inplace=True)
    data = metadata[metadata.diagnostic_superclass.isin(['MI', 'NORM'])]
    data['label'] = data.diagnostic_superclass.apply(lambda x: 1 if x == 'MI' else 0)

    # Patient-wise fold assignment: train folds 1-8, test fold 10
    train_folds = set(range(1, 9))
    test_fold = 10

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for _, row in data.iterrows():
        filename = os.path.join(base_path, row['filename_lr'])
        signal, _ = wfdb.rdsamp(filename)
        fold = int(row['strat_fold'])
        if fold in train_folds:
            X_train_list.append(signal)
            y_train_list.append(row['label'])
        elif fold == test_fold:
            X_test_list.append(signal)
            y_test_list.append(row['label'])

    X_train_orig = np.array(X_train_list)
    y_train_orig = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    # Per-lead z-score normalization using only training folds
    lead_mean = X_train_orig.mean(axis=(0, 1), keepdims=True)
    lead_std = X_train_orig.std(axis=(0, 1), keepdims=True)
    lead_std = np.where(lead_std < 1e-8, 1.0, lead_std)

    X_train_orig = (X_train_orig - lead_mean) / lead_std
    X_test = (X_test - lead_mean) / lead_std

    return X_train_orig, y_train_orig, X_test, y_test

# --- Conformal Prediction Implementation ---

def calculate_non_conformity_scores(model, calibration_set):
    """Calculates class-conditional (Mondrian) non-conformity scores.
    Returns two arrays: scores for Normal-hypothesis calibration (y=0) and MI-hypothesis calibration (y=1)."""
    X_cal, y_cal = calibration_set
    cal_probs_mi = model.predict(X_cal, verbose=0).flatten()
    # For hypothesis "Normal": nonconformity is p(MI) but calibrate using y=0 samples
    scores_norm = cal_probs_mi[y_cal == 0]
    # For hypothesis "MI": nonconformity is 1 - p(MI) but calibrate using y=1 samples
    scores_mi = (1.0 - cal_probs_mi)[y_cal == 1]
    return scores_norm, scores_mi

def conformal_inference(test_sample, model, cal_scores_norm, cal_scores_mi, epsilon):
    """Performs Mondrian conformal inference for a single test sample."""
    test_prob_mi = model.predict(np.expand_dims(test_sample, axis=0), verbose=0).flatten()[0]

    # Hypothesis: Normal (0)
    score_norm = test_prob_mi
    p_value_norm = (np.sum(cal_scores_norm >= score_norm) + 1) / (len(cal_scores_norm) + 1)

    # Hypothesis: MI (1)
    score_mi = 1.0 - test_prob_mi
    p_value_mi = (np.sum(cal_scores_mi >= score_mi) + 1) / (len(cal_scores_mi) + 1)

    prediction_set = set()
    if p_value_norm > epsilon:
        prediction_set.add('Normal')
    if p_value_mi > epsilon:
        prediction_set.add('MI')

    return prediction_set

# --- Main Execution Block ---
if __name__ == '__main__':
    # Step 1: Load Model and Re-organize Data
    print("Loading model and data...")
    model = tf.keras.models.load_model('improved_baseline_model.h5')
    dataset_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    X_train_orig, y_train_orig, X_test, y_test = load_and_preprocess_data(dataset_path)

    # Prefer split-conformal artifacts if available
    artifact_dir = os.path.join('results', 'split_conformal')
    use_split_artifacts = os.path.exists(os.path.join(artifact_dir, 'split_conformal_model.h5'))
    if use_split_artifacts:
        print("Using split-conformal artifacts from results/split_conformal")
        model = tf.keras.models.load_model(os.path.join(artifact_dir, 'split_conformal_model.h5'))
        X_cal = np.load(os.path.join(artifact_dir, 'X_cal.npy'))
        y_cal = np.load(os.path.join(artifact_dir, 'y_cal.npy'))
        lead_mean = np.load(os.path.join(artifact_dir, 'lead_mean.npy'))
        lead_std = np.load(os.path.join(artifact_dir, 'lead_std.npy'))

        # Rebuild raw test set and normalize with saved stats
        metadata = pd.read_csv(os.path.join(dataset_path, 'ptbxl_database.csv'))
        scp_statements = pd.read_csv(os.path.join(dataset_path, 'scp_statements.csv'), index_col=0)
        scp_statements = scp_statements[scp_statements.diagnostic_class.isin(['MI', 'NORM'])]
        def _get_cls(s):
            try:
                d = ast.literal_eval(s)
            except Exception:
                return None
            agg = {'MI': 0.0, 'NORM': 0.0}
            for code, weight in d.items():
                if code in scp_statements.index:
                    cls = scp_statements.loc[code].diagnostic_class
                    if cls in agg:
                        agg[cls] += float(weight)
            if agg['MI'] == 0.0 and agg['NORM'] == 0.0:
                return None
            return 'MI' if agg['MI'] >= agg['NORM'] else 'NORM'
        metadata['diagnostic_superclass'] = metadata.scp_codes.apply(_get_cls)
        metadata.dropna(subset=['diagnostic_superclass'], inplace=True)
        data_df = metadata[metadata.diagnostic_superclass.isin(['MI', 'NORM'])]
        data_df['label'] = data_df.diagnostic_superclass.apply(lambda x: 1 if x == 'MI' else 0)
        test_rows = data_df[data_df['strat_fold'] == 10]
        X_test_raw, y_test_raw = [], []
        for _, row in test_rows.iterrows():
            filename = os.path.join(dataset_path, row['filename_lr'])
            sig, _ = wfdb.rdsamp(filename)
            X_test_raw.append(sig)
            y_test_raw.append(row['label'])
        X_test = (np.array(X_test_raw) - lead_mean) / lead_std
        y_test = np.array(y_test_raw)
        print(f"Calibration set size: {len(X_cal)}")
    else:
        # Split original training data into a new training and a calibration set
        X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
            X_train_orig, y_train_orig, test_size=0.2, random_state=42, stratify=y_train_orig
        )
        print(f"Calibration set size: {len(X_cal)}")

    # Step 2: Calculate Non-Conformity Scores
    print("Calculating non-conformity scores...")
    cal_scores_norm, cal_scores_mi = calculate_non_conformity_scores(model, (X_cal, y_cal))

    # Step 5: Trade-off Analysis
    epsilons = [0.01, 0.05, 0.1, 0.15, 0.2]
    coverages = []  # overall fraction of singleton predictions
    error_rates = []  # overall error on covered
    avg_set_sizes = []
    coverage_norm = []
    coverage_mi = []
    error_norm = []
    error_mi = []
    miscov_overall = []
    miscov_norm = []
    miscov_mi = []

    print("\n--- Conformal Prediction Evaluation ---")
    for epsilon in epsilons:
        prediction_sets = [
            conformal_inference(X_test[i], model, cal_scores_norm, cal_scores_mi, epsilon)
            for i in range(len(X_test))
        ]
        set_sizes = np.array([len(s) for s in prediction_sets], dtype=float)
        avg_set_sizes.append(set_sizes.mean())
        
        # Calculate coverage and error
        covered_indices = [i for i, s in enumerate(prediction_sets) if len(s) == 1]
        coverage = len(covered_indices) / len(X_test)
        
        correct_covered = 0
        for i in covered_indices:
            true_label_str = 'MI' if y_test[i] == 1 else 'Normal'
            if true_label_str in prediction_sets[i]:
                correct_covered += 1
        
        error_rate = 1 - (correct_covered / len(covered_indices)) if len(covered_indices) > 0 else 0
        
        coverages.append(coverage)
        error_rates.append(error_rate)

        # Overall miscoverage (true label not in prediction set)
        correct_in_set = []
        for i_all in range(len(X_test)):
            true_label_str_all = 'MI' if y_test[i_all] == 1 else 'Normal'
            correct_in_set.append(true_label_str_all in prediction_sets[i_all])
        miscoverage = 1 - (np.mean(correct_in_set) if len(correct_in_set) > 0 else 0.0)
        miscov_overall.append(float(miscoverage))

        # Per-class coverage and error (Mondrian reporting)
        idx_norm = np.where(y_test == 0)[0]
        idx_mi = np.where(y_test == 1)[0]

        covered_norm = [i for i in idx_norm if len(prediction_sets[i]) == 1]
        covered_mi = [i for i in idx_mi if len(prediction_sets[i]) == 1]

        coverage_norm.append(len(covered_norm) / len(idx_norm) if len(idx_norm) > 0 else np.nan)
        coverage_mi.append(len(covered_mi) / len(idx_mi) if len(idx_mi) > 0 else np.nan)

        # Per-class miscoverage (true label not in set)
        correct_in_norm = [('Normal' in prediction_sets[i]) for i in idx_norm]
        miscov_norm_val = 1 - np.mean(correct_in_norm) if len(correct_in_norm) > 0 else np.nan
        correct_in_mi = [('MI' in prediction_sets[i]) for i in idx_mi]
        miscov_mi_val = 1 - np.mean(correct_in_mi) if len(correct_in_mi) > 0 else np.nan
        miscov_norm.append(miscov_norm_val)
        miscov_mi.append(miscov_mi_val)

        err_norm = 0.0
        if len(covered_norm) > 0:
            err_norm = 1 - np.mean(['Normal' in prediction_sets[i] for i in covered_norm])
        else:
            err_norm = np.nan

        err_mi_val = 0.0
        if len(covered_mi) > 0:
            err_mi_val = 1 - np.mean(['MI' in prediction_sets[i] for i in covered_mi])
        else:
            err_mi_val = np.nan

        error_norm.append(err_norm)
        error_mi.append(err_mi_val)

        if epsilon == 0.1:
            print(f"\nResults for epsilon = {epsilon} (90% Confidence):")
            print(f"  Coverage (overall): {coverage:.2%}")
            print(f"  Error Rate on Covered (overall): {error_rate:.2%}")
            print(f"  Miscoverage (overall): {miscov_overall[-1]:.2%}")
            print(f"  Average Set Size: {avg_set_sizes[-1]:.3f}")
            print(f"  Coverage (Normal): {coverage_norm[-1]:.2%}, Error on Covered (Normal): {error_norm[-1]:.2%}, Miscoverage (Normal): {miscov_norm[-1]:.2%}")
            print(f"  Coverage (MI): {coverage_mi[-1]:.2%}, Error on Covered (MI): {error_mi[-1]:.2%}, Miscoverage (MI): {miscov_mi[-1]:.2%}")

    # Generate and save the plot and results
    confidence_levels = [1 - e for e in epsilons]
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(confidence_levels, coverages, marker='o', label='Overall Coverage')
    plt.plot(confidence_levels, coverage_norm, marker='s', label='Coverage (Normal)')
    plt.plot(confidence_levels, coverage_mi, marker='^', label='Coverage (MI)')
    plt.title('Confidence vs. Coverage (Mondrian) for Conformal ECG Diagnosis')
    plt.xlabel('Confidence Level (1 - epsilon)')
    plt.ylabel('Coverage (Percentage of Unambiguous Predictions)')
    plt.grid(True)
    plt.gca().invert_xaxis() # Higher confidence on the right
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/mondrian_confidence_vs_coverage.png')
    print("\nPlot saved to 'results/mondrian_confidence_vs_coverage.png'.")

    pd.DataFrame({
        'epsilon': epsilons,
        'confidence_level': confidence_levels,
        'coverage_overall': coverages,
        'coverage_normal': coverage_norm,
        'coverage_mi': coverage_mi,
        'miscoverage_overall': miscov_overall,
        'miscoverage_normal': miscov_norm,
        'miscoverage_mi': miscov_mi,
        'error_overall_on_covered': error_rates,
        'error_normal_on_covered': error_norm,
        'error_mi_on_covered': error_mi,
        'avg_set_size': avg_set_sizes
    }).to_csv('results/mondrian_conformal_tradeoff.csv', index=False)
