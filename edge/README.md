# ESP32S3 Edge Deployment: 12‑Lead MI + Split‑Conformal Prediction

This toolkit converts the trained split‑conformal model to INT8 TFLite, exports conformal thresholds and normalization stats, and builds an ESP‑IDF firmware for ESP32S3 using TFLite Micro. The firmware logs RAM usage, inference latency, and conformal set decisions in real time.

## Prerequisites
- ESP‑IDF v5.x installed and configured (export the ESP‑IDF environment)
- ESP32S3 board (preferably with PSRAM)
- Python 3.10+ with TensorFlow 2.16+, pandas, numpy, wfdb
- Generated split‑conformal artifacts from the repo (see below)

## 1) Generate INT8 TFLite and Conformal Params

1. Export INT8 TFLite model with representative dataset calibration:
   - `python edge/export_tflite_micro_model.py`
   - Outputs: `edge/model/model_int8.tflite`

2. Export conformal thresholds, normalization stats, and quantization constants:
   - `python edge/compute_cp_thresholds.py`
   - Outputs:
     - `edge/model/cp_params.json`
     - `edge/model/cp_params.h`

3. Convert `.tflite` to a C array header:
   - `python edge/tflite_to_cc.py edge/model/model_int8.tflite edge/model/model_data.h`

## 2) Build and Flash Firmware

1. Ensure headers exist:
   - `edge/model/model_data.h`
   - `edge/model/cp_params.h`

2. Build and flash (from `edge/firmware`):
   - `idf.py set-target esp32s3`
   - `idf.py build`
   - `idf.py -p <COM_PORT> flash monitor`

3. Expected serial log output (examples):
   - Free heap before/after (bytes)
   - Average inference latency (µs) over 50 runs
   - pMI, nonconformity scores, thresholds, prediction set {Normal, MI}

Notes:
- If `AllocateTensors` fails, increase `kTensorArenaSize` in `main.c`.
- If using PSRAM, enable PSRAM in `menuconfig` and consider placing arena in PSRAM.

## 3) Real‑Time ECG Input

Replace the demo sinusoid in `main.c` with one of the following:
- **ADC**: Sample 12 leads via external ADC or multiplexed inputs; fill `x_demo` buffer in a sliding window.
- **UART/BLE**: Stream pre‑processed ECG samples from a host or sensor module.
- Keep 1000×12 window (10s@100Hz). For lower latency, evaluate shorter windows retrained appropriately.

## 4) Resource Metrics

- **RAM**: Logged via `heap_caps_get_free_size()` before/after interpreter allocation and inference.
- **Latency**: Measured across 50 invokes with `esp_timer_get_time()`; reported in microseconds.
- **Power**: Use an external power monitor (e.g., Otii Arc/Monsoon Power Monitor) inline with the 5V/3.3V supply. Record average and peak current during inference bursts and idle. Report mW at nominal voltage.

Tip: Fix CPU frequency, disable Wi‑Fi/BT for deterministic measurements.

## 5) Guaranteed Error Bounds On‑Device

We precompute Mondrian split‑conformal thresholds (\tau_norm, \tau_mi) at epsilon=0.10 with TFLite INT8 inference quantization accounted for. On the device:
- Compute pMI from the INT8 output using the provided scale/zero‑point
- Nonconformity scores: s_norm=pMI, s_mi=1−pMI
- Include class if s<=tau (strict inequality variant uses smoothing; we export smoothed quantiles)
This preserves the split‑conformal validity approximately under consistent quantization and data preprocessing.

## 6) Validation and Reproducibility

- PTB‑XL experiments and artifacts are documented in `Q1_MI_Conformal_Notes.md`.
- Operating‑point CSVs and conformal trade‑off CSV/plots included in `results/`.
- For comparison with MC Dropout and Ensembles, see `compare_uncertainty_methods.py` (host‑side evaluation).

## 7) Known Limits / TODO
- Streaming ECG acquisition drivers are placeholders; integrate your sensor/ADC.
- Power metrics require external instrumentation; see Section 4.
- For multiple window rates (e.g., 2s), retrain model and recompute conformal thresholds.
- CPSC2018 support is provided as a loader/eval skeleton; you must supply dataset path and mapping (see repo root scripts).
