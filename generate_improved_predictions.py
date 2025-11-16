"""
Simple script to load the trained improved model and generate predictions.
This avoids re-training and just uses the existing model file.
"""
import pandas as pd
import numpy as np
import wfdb
import os

print("Loading PTB-XL metadata...")

# Load PTB-XL metadata
ptbxl_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
df = pd.read_csv(os.path.join(ptbxl_path, 'ptbxl_database.csv'), index_col='ecg_id')
scp_statements = pd.read_csv(os.path.join(ptbxl_path, 'scp_statements.csv'), index_col=0)

# Label derivation (same as in improved_mi_classification.py)
def aggregate_diagnostic(y_dic, scp_statements):
    tmp = []
    for key in y_dic.keys():
        if key in scp_statements.index:
            tmp.append((scp_statements.loc[key].diagnostic_class, y_dic[key]))
    return tmp

df['diagnostic_superclass'] = df.scp_codes.apply(lambda x: aggregate_diagnostic(eval(x), scp_statements))

def derive_mi_label(row):
    mi_conf = sum([conf for cls, conf in row if cls == 'MI'])
    norm_conf = sum([conf for cls, conf in row if cls == 'NORM'])
    return 1 if mi_conf >= norm_conf else 0

df['mi_label'] = df['diagnostic_superclass'].apply(derive_mi_label)

# Get test fold (fold 10)
test_fold = df[df.strat_fold == 10].copy()
print(f"Test fold size: {len(test_fold)}")

# Load ECG signals
print("Loading ECG signals...")
def load_raw_data(df, sampling_rate, path):
    data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    data = np.array([signal for signal, meta in data])
    return data

X_test = load_raw_data(test_fold, 100, ptbxl_path)
y_test = test_fold['mi_label'].values

# Normalize using saved statistics
print("Normalizing data...")
lead_mean = np.load('results/split_conformal/lead_mean.npy')
lead_std = np.load('results/split_conformal/lead_std.npy')
X_test = (X_test - lead_mean) / lead_std

# Load improved model
print("Loading model...")
try:
    from keras.models import load_model
    improved_model = load_model('improved_baseline_model.h5')
except Exception as e:
    print(f"Error loading model with Keras: {e}")
    print("\nYou need TensorFlow/Keras to load the model.")
    print("Please install Python 3.10 or 3.11 and run:")
    print("  pip install tensorflow wfdb pandas numpy")
    exit(1)

# Generate predictions
print("Generating predictions...")
y_pred_prob = improved_model.predict(X_test, verbose=0).flatten()
y_pred_binary = (y_pred_prob > 0.5).astype(int)

# Save predictions
print("Saving predictions...")
pd.DataFrame({
    'y_true': y_test,
    'y_score': y_pred_prob,
    'y_pred': y_pred_binary
}).to_csv('results/improved_predictions.csv', index=False)

print(f"\n✓ Successfully saved {len(y_test)} predictions to results/improved_predictions.csv")
print(f"  - Positive samples: {sum(y_test)}")
print(f"  - Negative samples: {len(y_test) - sum(y_test)}")
