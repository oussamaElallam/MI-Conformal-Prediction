import os
import ast
import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf
import matplotlib.pyplot as plt


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


def integrated_gradients(model, x, baseline=None, m_steps=64):
    # x: (1, T, C)
    if baseline is None:
        baseline = tf.zeros_like(x)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

    scaled_inputs = [baseline + (float(k)/m_steps) * (x - baseline) for k in range(1, m_steps+1)]
    grads = []
    for s in scaled_inputs:
        with tf.GradientTape() as tape:
            tape.watch(s)
            y = model(s, training=False)
            y_scalar = tf.squeeze(y, axis=-1)  # (1,)
        grad = tape.gradient(y_scalar, s)
        grads.append(grad)
    avg_grads = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)  # (1, T, C)
    ig = (x - baseline) * avg_grads
    return ig.numpy()[0]


if __name__ == '__main__':
    base_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    art_dir = os.path.join('results', 'split_conformal')
    out_dir = os.path.join('results', 'interpretability')
    os.makedirs(out_dir, exist_ok=True)

    print('Loading model and normalization...')
    model = tf.keras.models.load_model(os.path.join(art_dir, 'split_conformal_model.h5'))
    mean = np.load(os.path.join(art_dir, 'lead_mean.npy'))
    std = np.load(os.path.join(art_dir, 'lead_std.npy'))

    print('Loading test set (fold 10)...')
    meta = parse_metadata(base_path)
    test_rows = meta[meta['strat_fold'] == 10]
    X_test_raw, y_test = load_signals(base_path, test_rows)
    X_test = (X_test_raw - mean) / std

    print('Scoring test set...')
    y_score = model.predict(X_test, verbose=0).flatten()

    # Select 2 MI and 2 Normal examples by highest confidence for their class
    idx_mi = np.where(y_test == 1)[0]
    idx_norm = np.where(y_test == 0)[0]

    top_mi = idx_mi[np.argsort(-y_score[idx_mi])[:2]] if len(idx_mi) >= 2 else idx_mi
    top_norm = idx_norm[np.argsort(y_score[idx_norm])[:2]] if len(idx_norm) >= 2 else idx_norm

    selected = list(top_mi) + list(top_norm)

    leads = [f'L{i+1}' for i in range(X_test.shape[2])]

    print(f'Computing Integrated Gradients for indices: {selected}')
    for i in selected:
        x = X_test[i:i+1]
        ig = integrated_gradients(model, x, baseline=np.zeros_like(x), m_steps=64)
        ig_abs = np.abs(ig)

        # Heatmap (time x leads)
        plt.figure(figsize=(10, 4))
        plt.imshow(ig_abs.T, aspect='auto', origin='lower', cmap='hot')
        plt.yticks(ticks=np.arange(len(leads)), labels=leads)
        plt.xlabel('Time (samples)')
        plt.ylabel('Leads')
        plt.title(f'Integrated Gradients Heatmap | idx={i} | y={y_test[i]} | pMI={y_score[i]:.3f}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'ig_heatmap_idx{i}.png'))
        plt.close()

        # Per-lead importance (sum over time)
        per_lead = ig_abs.sum(axis=0)
        plt.figure(figsize=(8, 3))
        plt.bar(np.arange(len(leads)), per_lead)
        plt.xticks(ticks=np.arange(len(leads)), labels=leads, rotation=0)
        plt.ylabel('IG (sum |attribution|)')
        plt.title(f'Per-Lead IG | idx={i} | y={y_test[i]} | pMI={y_score[i]:.3f}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'ig_per_lead_idx{i}.png'))
        plt.close()

    # Save selection info
    pd.DataFrame({'index': selected, 'y_true': y_test[selected], 'p_mi': y_score[selected]}).to_csv(
        os.path.join(out_dir, 'selected_cases.csv'), index=False
    )
    print(f'Saved interpretability artifacts to {out_dir}')
