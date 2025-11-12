import pandas as pd
import numpy as np
import wfdb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import ast

# Step 1: Data Loading and Preprocessing

def load_and_preprocess_data(base_path):
    """Loads and preprocesses the PTB-XL dataset."""
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
            # Safely parse the string to a dictionary
            scp_codes = ast.literal_eval(scp_codes_str)
            for code, confidence in scp_codes.items():
                if code in scp_statements.index:
                    return scp_statements.loc[code].diagnostic_class
        except Exception:
            pass  # Ignore parsing errors
        return None

    # Apply the function to create the diagnostic_superclass column
    metadata['diagnostic_superclass'] = metadata.scp_codes.apply(get_diagnostic_superclass)

    # Filter for MI and Normal classes
    mi_cases = metadata[metadata.diagnostic_superclass == 'MI']
    normal_cases = metadata[metadata.diagnostic_superclass == 'NORM']
    
    # Combine and create labels (1 for MI, 0 for Normal)
    data = pd.concat([mi_cases, normal_cases])
    data['label'] = data.diagnostic_superclass.apply(lambda x: 1 if x == 'MI' else 0)

    # Prepare signals (X) and labels (y)
    X = []
    y = []

    for index, row in data.iterrows():
        # Construct file path
        filename = os.path.join(base_path, row['filename_lr'])
        
        # Load signal data
        signal, fields = wfdb.rdsamp(filename, channels=[0]) # Lead I
        X.append(signal.flatten())
        y.append(row['label'])

    X = np.array(X)
    y = np.array(y)

    # Reshape X for CNN input (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data (70% train, 15% validation, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

# Step 2: Model Architecture

def create_model(input_shape):
    """Creates the 1D-CNN model."""
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
    # Path to the dataset directory
    dataset_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(dataset_path)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Create model
    input_shape = (X_train.shape[1], 1)
    model = create_model(input_shape)

    # Step 3: Training
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        batch_size=32
    )
    print("Model training finished.")

    # Step 4: Evaluation
    print("\nEvaluating model on the test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # Generate predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'MI']))

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Step 5: Save the model
    model.save('baseline_model.h5')
    print("\nModel saved to 'baseline_model.h5'.")
