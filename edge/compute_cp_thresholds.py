import os
import json
import numpy as np
import tensorflow as tf

ART_DIR = os.path.join('results', 'split_conformal')
MODEL_PATH = os.path.join(ART_DIR, 'split_conformal_model.h5')
OUT_DIR = os.path.join('edge', 'model')
EPSILON = 0.10

os.makedirs(OUT_DIR, exist_ok=True)


def smoothed_quantile_threshold(scores: np.ndarray, epsilon: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    if n == 0:
        raise ValueError('Empty scores array')
    # Sorted ascending
    s = np.sort(scores)
    # k = ceil((n+1)*(1-eps)) - 1 is the index (0-based)
    k = int(np.ceil((n + 1) * (1.0 - epsilon)) - 1)
    k = np.clip(k, 0, n - 1)
    return float(s[k])


if __name__ == '__main__':
    print('Loading model and calibration arrays...')
    model = tf.keras.models.load_model(MODEL_PATH)
    X_cal = np.load(os.path.join(ART_DIR, 'X_cal.npy'))
    y_cal = np.load(os.path.join(ART_DIR, 'y_cal.npy'))
    mean = np.load(os.path.join(ART_DIR, 'lead_mean.npy'))
    std = np.load(os.path.join(ART_DIR, 'lead_std.npy'))

    print('Scoring calibration set with TFLite int8 (to match device outputs)...')
    # Inspect TFLite IO quantization for input/output scale/zp
    from export_tflite_micro_model import TFLITE_PATH  # reuse path
    if not os.path.exists(TFLITE_PATH):
        raise FileNotFoundError('Missing TFLite model. Run export_tflite_micro_model.py first.')
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']

    # Normalize using saved stats, then quantize to int8 per TFLite input quantization
    X_cal_norm = (X_cal - mean) / std
    def quantize(xf):
        q = np.round(xf / in_scale + in_zp)
        return np.clip(q, -128, 127).astype(np.int8)

    probs = []
    for i in range(len(X_cal_norm)):
        xq = quantize(X_cal_norm[i:i+1])
        interp.set_tensor(in_det['index'], xq)
        interp.invoke()
        qout = interp.get_tensor(out_det['index']).reshape(-1)[0].astype(np.int32)
        p = float(out_scale * (qout - out_zp))
        p = min(max(p, 0.0), 1.0)
        probs.append(p)
    probs = np.array(probs, dtype=np.float64)

    scores_norm = probs[y_cal == 0]  # s = p(MI) for Normal hypothesis
    scores_mi = (1.0 - probs)[y_cal == 1]  # s = 1 - p(MI) for MI hypothesis

    tau_norm = smoothed_quantile_threshold(scores_norm, EPSILON)
    tau_mi = smoothed_quantile_threshold(scores_mi, EPSILON)

    # in_scale/out_scale already computed above

    # Export JSON
    params = {
        'epsilon': EPSILON,
        'tau_norm': tau_norm,
        'tau_mi': tau_mi,
        'lead_mean': mean.reshape(-1).tolist(),
        'lead_std': std.reshape(-1).tolist(),
        'in_scale': float(in_scale),
        'in_zp': int(in_zp),
        'out_scale': float(out_scale),
        'out_zp': int(out_zp)
    }
    with open(os.path.join(OUT_DIR, 'cp_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print('Saved edge/model/cp_params.json')

    # Export C header
    header = f"""
#ifndef CP_PARAMS_H
#define CP_PARAMS_H

#include <stdint.h>

#define CP_EPSILON {EPSILON:.6f}f
#define CP_TAU_NORM {tau_norm:.8f}f
#define CP_TAU_MI {tau_mi:.8f}f

// TFLite output dequantization: y = scale * (q - zp)
#define CP_IN_SCALE {in_scale:.10f}f
#define CP_IN_ZP {int(in_zp)}
#define CP_OUT_SCALE {out_scale:.10f}f
#define CP_OUT_ZP {int(out_zp)}

static const float CP_LEAD_MEAN[12] = {{ {', '.join(f'{v:.8f}f' for v in mean.reshape(-1))} }};
static const float CP_LEAD_STD[12]  = {{ {', '.join(f'{v:.8f}f' for v in std.reshape(-1))} }};

#endif // CP_PARAMS_H
"""
    with open(os.path.join(OUT_DIR, 'cp_params.h'), 'w') as f:
        f.write(header)
    print('Saved edge/model/cp_params.h')
