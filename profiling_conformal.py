import numpy as np
import time
import os
import pandas as pd
import wfdb
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- Data Loading and Score Calculation (Adapted from previous scripts) ---
def get_calibration_scores():
    """Loads data, pre-trained model, and calculates calibration scores."""
    # Load model
    model = tf.keras.models.load_model('improved_baseline_model.h5')

    # Load data
    base_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    metadata = pd.read_csv(os.path.join(base_path, 'ptbxl_database.csv'))
    scp_statements = pd.read_csv(os.path.join(base_path, 'scp_statements.csv'), index_col=0)
    scp_statements = scp_statements[scp_statements.diagnostic_class.isin(['MI', 'NORM'])]

    def get_diagnostic_superclass(scp_codes_str):
        try:
            scp_codes = eval(scp_codes_str)
            for code in scp_codes.keys():
                if code in scp_statements.index:
                    return scp_statements.loc[code].diagnostic_class
        except (SyntaxError, NameError): pass
        return None

    metadata['diagnostic_superclass'] = metadata.scp_codes.apply(get_diagnostic_superclass)
    metadata.dropna(subset=['diagnostic_superclass'], inplace=True)
    data = metadata[metadata.diagnostic_superclass.isin(['MI', 'NORM'])]
    data['label'] = data.diagnostic_superclass.apply(lambda x: 1 if x == 'MI' else 0)

    X = np.array([wfdb.rdsamp(os.path.join(base_path, f))[0] for f in data['filename_lr']])
    y = np.array(data['label'])

    X_train_orig, _, y_train_orig, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    _, X_cal, _, y_cal = train_test_split(X_train_orig, y_train_orig, test_size=0.2, random_state=42, stratify=y_train_orig)

    # Calculate scores
    cal_probs = model.predict(X_cal, verbose=0).flatten()
    scores = np.where(y_cal == 0, cal_probs, 1 - cal_probs)
    return scores

# Step 1: Isolate the Conformal Logic
def run_conformal_inference(model_prediction_prob, calibration_scores, epsilon=0.1):
    """Calculates the prediction set for a single model output."""
    # Hypothesis 1: True class is Normal (0). Score is p(MI).
    score_norm = model_prediction_prob
    p_value_norm = (np.sum(calibration_scores >= score_norm) + 1) / (len(calibration_scores) + 1)

    # Hypothesis 2: True class is MI (1). Score is 1 - p(MI).
    score_mi = 1 - model_prediction_prob
    p_value_mi = (np.sum(calibration_scores >= score_mi) + 1) / (len(calibration_scores) + 1)

    prediction_set = set()
    if p_value_norm > epsilon: prediction_set.add('Normal')
    if p_value_mi > epsilon: prediction_set.add('MI')
    return prediction_set

# --- Main Execution Block ---
if __name__ == '__main__':
    print("Loading data and calculating calibration scores...")
    calibration_scores = get_calibration_scores()
    
    # Step 2: Profile the Latency
    print("Profiling latency...")
    dummy_predictions = np.random.rand(1000)
    start_time = time.perf_counter()
    for pred in dummy_predictions:
        run_conformal_inference(pred, calibration_scores)
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    average_latency_ms = total_time_ms / len(dummy_predictions)

    # Step 3: Profile the Memory
    N = len(calibration_scores)
    # Assuming 32-bit float (4 bytes)
    memory_kb = (N * 4) / 1024

    # Step 4: Produce the Final Output
    print("\n--- Conformal Prediction Overhead Analysis ---")
    print(f"Calibration Set Size: {N} samples")
    print(f"Memory Requirement for Scores: {memory_kb:.2f} KB")
    print(f"Average Inference Latency (CPU): {average_latency_ms:.4f} ms")
