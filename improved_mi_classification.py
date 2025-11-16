import pandas as pd
import numpy as np
import wfdb
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import ast
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

# Step 1: Data Loading and Preprocessing with 12 Leads and Class Weights

try:
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

def load_and_preprocess_data(base_path):
    """Loads and preprocesses the PTB-XL dataset using all 12 leads."""
    # Load metadata
    metadata_path = os.path.join(base_path, 'ptbxl_database.csv')
    metadata = pd.read_csv(metadata_path)
    
    # Load scp_statements to map codes to diagnostic classes
    scp_statements_path = os.path.join(base_path, 'scp_statements.csv')
    scp_statements = pd.read_csv(scp_statements_path, index_col=0)
    
    # Filter for MI and NORM diagnostic classes
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

    # Filter out records with no valid diagnostic superclass
    metadata.dropna(subset=['diagnostic_superclass'], inplace=True)

    # Filter for MI and Normal classes
    mi_cases = metadata[metadata.diagnostic_superclass == 'MI']
    normal_cases = metadata[metadata.diagnostic_superclass == 'NORM']
    
    data = pd.concat([mi_cases, normal_cases])
    data['label'] = data.diagnostic_superclass.apply(lambda x: 1 if x == 'MI' else 0)

    train_folds = set(range(1, 9))
    val_fold = 9
    test_fold = 10

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    for _, row in data.iterrows():
        filename = os.path.join(base_path, row['filename_lr'])
        signal, _ = wfdb.rdsamp(filename)
        fold = int(row['strat_fold'])
        if fold in train_folds:
            X_train_list.append(signal)
            y_train_list.append(row['label'])
        elif fold == val_fold:
            X_val_list.append(signal)
            y_val_list.append(row['label'])
        elif fold == test_fold:
            X_test_list.append(signal)
            y_test_list.append(row['label'])

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_val = np.array(X_val_list)
    y_val = np.array(y_val_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    assert X_train.ndim == 3 and X_train.shape[2] == 12

    lead_mean = X_train.mean(axis=(0, 1), keepdims=True)
    lead_std = X_train.std(axis=(0, 1), keepdims=True)
    lead_std = np.where(lead_std < 1e-8, 1.0, lead_std)

    X_train = (X_train - lead_mean) / lead_std
    X_val = (X_val - lead_mean) / lead_std
    X_test = (X_test - lead_mean) / lead_std

    classes = np.unique(y_train)
    class_weights_arr = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}

    return X_train, y_train, X_val, y_val, X_test, y_test, class_weight_dict

# Step 2: Model Architecture

def create_model(input_shape):
    """Creates the 1D-CNN model for 12-lead input."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Main execution block
if __name__ == '__main__':
    dataset_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

    # Load data and class weights
    X_train, y_train, X_val, y_val, X_test, y_test, class_weights = load_and_preprocess_data(dataset_path)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Class weights: {class_weights}")

    # Create model with updated input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_model(input_shape)

    # Step 3: Training with Class Weights and AUC metric
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        batch_size=32,
        class_weight=class_weights
    )
    print("Model training finished.")

    # Step 4: Evaluation
    print("\nEvaluating model on the test set...")
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test AUC: {test_auc:.4f}')

    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    spec = 1 - fpr
    if np.any(spec >= 0.95):
        mask = spec >= 0.95
        idx = np.argmax(tpr[mask])
        sens_at_95spec = float(tpr[mask][idx])
        thresh_at_95spec = float(thresholds[mask][idx])
    else:
        sens_at_95spec = float('nan')
        thresh_at_95spec = float('nan')

    def bootstrap_ci_y(metric_fn, y_true, y_score, n_boot=200, seed=42):
        rng = np.random.default_rng(seed)
        n = len(y_true)
        stats = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            stats.append(metric_fn(y_true[idx], y_score[idx]))
        lo, hi = np.percentile(stats, [2.5, 97.5])
        return float(lo), float(hi)

    auc_ci = bootstrap_ci_y(lambda yt, yp: roc_auc_score(yt, yp), y_test, y_pred_prob)
    acc_ci = bootstrap_ci_y(lambda yt, yp: (yt == (yp > 0.5).astype(int)).mean(), y_test, y_pred_prob)

    report_text = classification_report(y_test, y_pred, target_names=['Normal', 'MI'], digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    os.makedirs('results', exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results/roc_curve.png')
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(recall, precision, label=f'AP = {auprc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('results/pr_curve.png')
    plt.close()

    # Save predictions for comparison plots
    y_pred_binary = (y_pred_prob > 0.5).astype(int)
    pd.DataFrame({
        'y_true': y_test,
        'y_score': y_pred_prob,
        'y_pred': y_pred_binary
    }).to_csv('results/improved_predictions.csv', index=False)
    print("Saved predictions to results/improved_predictions.csv")

    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'test_auc_keras': float(test_auc),
        'roc_auc_sklearn': float(roc_auc),
        'auprc': float(auprc),
        'balanced_accuracy': float(bal_acc),
        'sensitivity_at_95_specificity': sens_at_95spec,
        'threshold_at_95_specificity': thresh_at_95spec,
        'auc_ci_95': [auc_ci[0], auc_ci[1]],
        'accuracy_ci_95': [acc_ci[0], acc_ci[1]],
        'class_weights': {str(k): float(v) for k, v in class_weights.items()}
    }
    with open('results/improved_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(cm, index=['Normal','MI'], columns=['Pred_Normal','Pred_MI']).to_csv('results/improved_confusion_matrix.csv')
    with open('results/improved_classification_report.txt','w') as f:
        f.write(report_text)

    with open('results/environment.txt','w') as f:
        f.write(f"numpy={np.__version__}\n")
        f.write(f"pandas={pd.__version__}\n")
        f.write(f"scikit_learn={(__import__('sklearn').__version__)}\n")
        f.write(f"tensorflow={tf.__version__}\n")

    # Step 5: Save the improved model
    model.save('improved_baseline_model.h5')
    print("\nModel saved to 'improved_baseline_model.h5'. Artifacts saved to 'results/'.")
