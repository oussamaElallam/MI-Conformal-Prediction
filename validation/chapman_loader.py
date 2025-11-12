import os
import wfdb
import numpy as np
import pandas as pd

# SNOMED-CT codes for filtering
# Myocardial Infarction related codes
MI_CODES = {
    '426177001', # Old Myocardial Infarction
    '164865005', # Myocardial infarction
    '427395009', # Acute myocardial infarction
    # Add other relevant MI codes if needed
}

# Normal Sinus Rhythm
NORMAL_CODE = '426783006'

DB_NAME = 'ecg-arrhythmia'
DATA_DIR = 'chapman_shaoxing'


def load_chapman_data(data_dir=DATA_DIR, max_records=None):
    """Loads and filters a pre-downloaded Chapman-Shaoxing dataset."""
    if not os.path.exists(data_dir):
        print(f"Dataset not found at {data_dir}. Please download it first using wget.")
        return np.array([]), np.array([])

    record_path = os.path.join(data_dir, 'RECORDS')
    if not os.path.exists(record_path):
        print(f"RECORDS file not found in {data_dir}. The directory may be incorrect.")
        return np.array([]), np.array([])

    with open(record_path, 'r') as f:
        records = [line.strip() for line in f.readlines()]

    if max_records:
        records = records[:max_records]

    X, y, filtered_records = [], [], []
    for rec_name in records:
        try:
            record = wfdb.rdrecord(os.path.join(data_dir, rec_name))
            header = record.comments

            # Find the Dx line
            dx_line = [line for line in header if line.startswith('Dx:')]
            if not dx_line:
                continue

            # Dx: 426783006,270492004 -> ['426783006', '270492004']
            dx_codes = set(dx_line[0].replace('Dx:', '').strip().split(','))

            is_mi = any(code in MI_CODES for code in dx_codes)
            is_normal = NORMAL_CODE in dx_codes

            label = -1
            # Clean cases: only MI codes or only Normal code
            if is_mi and not is_normal:
                label = 1
            elif is_normal and not is_mi:
                label = 0

            if label != -1:
                signal = record.p_signal
                # Resample/pad to 1000 samples (10s @ 100Hz)
                # Original is 500Hz, so we take every 5th sample for the first 10s
                if signal.shape[0] >= 5000:
                    signal_resampled = signal[:5000:5, :]
                else:
                    pad_len = 5000 - signal.shape[0]
                    signal_padded = np.pad(signal, ((0, pad_len), (0, 0)), 'constant')
                    signal_resampled = signal_padded[::5, :]

                X.append(signal_resampled)
                y.append(label)
                filtered_records.append(rec_name)

        except Exception as e:
            print(f'Could not process record {rec_name}: {e}')

    print(f'Filtered {len(filtered_records)} records (MI: {np.sum(y)}, Normal: {len(y) - np.sum(y)})')
    return np.array(X), np.array(y)

if __name__ == '__main__':
    X_chapman, y_chapman = load_chapman_data(max_records=500) # Process a subset for quick test
    print(f'Loaded Chapman-Shaoxing subset: X shape {X_chapman.shape}, y shape {y_chapman.shape}')
