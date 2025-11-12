import os
import ast
import json
import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, average_precision_score, balanced_accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

# Reproducibility
try:
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass


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


def res_block(x, filters, kernel_size, stride=1, downsample=False):
    shortcut = x
    y = Conv1D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    out = Add()([shortcut, y])
    out = Activation('relu')(out)
    return out


def build_resnet1d(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = res_block(x, 64, 3)
    x = res_block(x, 64, 3)

    x = res_block(x, 128, 3, stride=2, downsample=True)
    x = res_block(x, 128, 3)

    x = res_block(x, 256, 3, stride=2, downsample=True)
    x = res_block(x, 256, 3)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model


if __name__ == '__main__':
    base_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    out_dir = os.path.join('results', 'resnet1d')
    os.makedirs(out_dir, exist_ok=True)

    meta = parse_metadata(base_path)

    train_rows = meta[meta['strat_fold'].isin(range(1, 9))]
    val_rows = meta[meta['strat_fold'] == 9]
    test_rows = meta[meta['strat_fold'] == 10]

    print('Loading signals...')
    X_train, y_train = load_signals(base_path, train_rows)
    X_val, y_val = load_signals(base_path, val_rows)
    X_test, y_test = load_signals(base_path, test_rows)

    # Normalize using training folds only
    mean_train = X_train.mean(axis=(0, 1), keepdims=True)
    std_train = X_train.std(axis=(0, 1), keepdims=True)
    std_train = np.where(std_train < 1e-8, 1.0, std_train)

    X_train = (X_train - mean_train) / std_train
    X_val = (X_val - mean_train) / std_train
    X_test = (X_test - mean_train) / std_train

    # Model
    model = build_resnet1d((X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])

    classes = np.unique(y_train)
    cw_arr = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw_arr)}

    print('Training ResNet1D...')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[es],
        class_weight=class_weights,
        verbose=1
    )

    print('Evaluating...')
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    y_score = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_score > 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal','MI'])

    print(f'Test: loss={test_loss:.4f}, acc={test_acc:.4f}, auc={test_auc:.4f}, roc_auc={roc_auc:.4f}, pr_auc={pr_auc:.4f}')

    # Save artifacts
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_auc_keras': float(test_auc),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'balanced_accuracy': float(bal_acc),
            'class_weights': class_weights
        }, f, indent=2)

    pd.DataFrame(cm, index=['Normal','MI'], columns=['Pred_Normal','Pred_MI']).to_csv(os.path.join(out_dir, 'confusion_matrix.csv'))
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    import matplotlib.pyplot as plt
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'k--'); plt.title('ResNet1D ROC'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'roc_curve.png')); plt.close()
    plt.figure(); plt.plot(recall, precision); plt.title('ResNet1D PR'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'pr_curve.png')); plt.close()

    pd.DataFrame({'y_true': y_test, 'y_score': y_score, 'y_pred': y_pred}).to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)

    model.save(os.path.join(out_dir, 'resnet1d_model.h5'))
    print(f'Artifacts saved to {out_dir}')
