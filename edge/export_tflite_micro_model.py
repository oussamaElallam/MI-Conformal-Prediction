import os
import ast
import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf

# Paths
DATASET_PATH = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
ART_DIR = os.path.join('results', 'split_conformal')
MODEL_PATH = os.path.join(ART_DIR, 'split_conformal_model.h5')
EDGE_DIR = os.path.join('edge', 'model')
TFLITE_PATH = os.path.join(EDGE_DIR, 'model_int8.tflite')

os.makedirs(EDGE_DIR, exist_ok=True)


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
    X = []
    for _, row in rows.iterrows():
        sig, _ = wfdb.rdsamp(os.path.join(base_path, row['filename_lr']))
        X.append(sig)
    return np.array(X)


if __name__ == '__main__':
    print('Loading model and normalization stats...')
    model = tf.keras.models.load_model(MODEL_PATH)
    mean = np.load(os.path.join(ART_DIR, 'lead_mean.npy'))
    std = np.load(os.path.join(ART_DIR, 'lead_std.npy'))

    print('Preparing representative dataset...')
    meta = parse_metadata(DATASET_PATH)
    train_rows = meta[meta['strat_fold'].isin(range(1, 9))]
    # sample up to 2000 examples for calibration
    reps = train_rows.sample(n=min(2000, len(train_rows)), random_state=42)
    X_rep = load_signals(DATASET_PATH, reps)
    X_rep = (X_rep - mean) / std

    def representative_dataset():
        for i in range(len(X_rep)):
            # yield one sample as list of arrays
            yield [X_rep[i:i+1].astype(np.float32)]

    print('Converting to full-int8 TFLite...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f'Saved {TFLITE_PATH}')

    # Inspect IO quantization
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    print('Input details:', in_det['dtype'], in_det['shape'], 'scale', in_det['quantization'][0], 'zp', in_det['quantization'][1])
    print('Output details:', out_det['dtype'], out_det['shape'], 'scale', out_det['quantization'][0], 'zp', out_det['quantization'][1])
