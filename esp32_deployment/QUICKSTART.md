# Quick Start Guide - ESP32-S3 Deployment

## 🚀 5-Minute Setup

### Step 1: Prepare the Deployment Package

Run the setup script from the main project directory:

```bash
cd esp32_deployment
python setup_deployment.py
```

This will:
- ✓ Generate the TFLite INT8 model
- ✓ Compute conformal prediction thresholds
- ✓ Copy all files to the deployment folder

### Step 2: Install PlatformIO in VSCode

1. Open **Visual Studio Code**
2. Go to **Extensions** (Ctrl+Shift+X)
3. Search for **"PlatformIO IDE"**
4. Click **Install**
5. Restart VSCode

### Step 3: Open the Project

1. In VSCode: **File → Open Folder**
2. Select the `esp32_deployment` folder
3. Wait for PlatformIO to initialize (bottom toolbar will appear)

### Step 4: Connect ESP32-S3

1. Connect ESP32-S3 to your computer via USB-C cable
2. Check if it's detected:
   - Windows: Device Manager → Ports (COM & LPT)
   - Linux/Mac: `ls /dev/tty*`

### Step 5: Build and Upload

**Option A: Using PlatformIO GUI**

1. Look at the bottom toolbar in VSCode
2. Click the **✓** (checkmark) icon to **Build**
3. Click the **→** (arrow) icon to **Upload**
4. Click the **🔌** (plug) icon to **Monitor** serial output

**Option B: Using Terminal**

```bash
# Build
pio run

# Upload (ESP32 must be connected)
pio run --target upload

# Monitor serial output
pio device monitor
```

### Step 6: View Results

After upload, open the Serial Monitor (115200 baud). You should see:

```
========================================
ESP32-S3 MI Detection with Conformal Prediction
========================================

Free heap: XXXXX bytes
Free PSRAM: XXXXX bytes
Allocated 200 KB tensor arena in PSRAM
✓ Model loaded successfully
✓ Tensors allocated successfully
Input shape: [1, 1000, 12]
Output shape: [1]

Conformal Prediction Parameters:
  Epsilon (ε): 0.10 (90% coverage target)
  τ_normal: 0.4101
  τ_MI:     0.7338

Free heap after init: XXXXX bytes
Free PSRAM after init: XXXXX bytes

========================================
Setup complete! Running demo inference...

Running inference on demo ECG signal...

========== INFERENCE RESULTS ==========
Inference latency: XX.XX ms
P(MI) = 0.XXXX

Conformal Prediction:
  s_norm = 0.XXXX <= tau_norm = 0.4101 ? YES/NO
  s_mi   = 0.XXXX <= tau_mi   = 0.7338 ? YES/NO
Prediction Set: {Normal} or {MI} or {Normal, MI}
→ LIKELY NORMAL (high confidence)
========================================
```

## 🎯 What the Demo Does

The current code:
1. Generates a **demo sinusoid signal** (not real ECG)
2. Preprocesses it (z-score normalization)
3. Runs **INT8 inference** on the quantized model
4. Applies **Mondrian conformal prediction**
5. Reports the prediction set with confidence

## 📊 Expected Performance

On ESP32-S3 @ 240 MHz:
- **Inference latency**: 50-100 ms per 10-second window
- **Memory usage**: ~1.5 MB (model + tensors)
- **Power consumption**: ~150 mA during inference

## 🔧 Next Steps: Add Real ECG Input

### Option 1: Serial Input (Easiest for Testing)

Modify `src/main.cpp`:

```cpp
void loop() {
  if (Serial.available() >= SAMPLES * LEADS * sizeof(float)) {
    float ecg_buffer[SAMPLES * LEADS];
    Serial.readBytes((char*)ecg_buffer, sizeof(ecg_buffer));
    run_inference_and_cp(ecg_buffer);
  }
}
```

Then send ECG data from Python:

```python
import serial
import numpy as np

ser = serial.Serial('COM3', 115200)  # Adjust port
ecg_data = np.random.randn(1000, 12).astype(np.float32)  # Replace with real ECG
ser.write(ecg_data.tobytes())
```

### Option 2: SD Card (For Offline Testing)

Add SD card library to `platformio.ini`:
```ini
lib_deps = 
    SD
```

Read ECG from CSV file on SD card.

### Option 3: ADC Input (For Real-Time Acquisition)

Connect ECG sensor to ESP32-S3 ADC pins and sample at 100 Hz.

## ❓ Troubleshooting

### Build fails with "model_data.h not found"
→ Run `python setup_deployment.py` first

### Upload fails
→ Press and hold **BOOT** button on ESP32-S3, then click Upload

### Out of memory error
→ Increase `TENSOR_ARENA_SIZE` in `main.cpp` or enable PSRAM

### Wrong predictions
→ Verify input data is in **millivolts** and has shape [1000, 12]

## 📚 Documentation

- Full README: `README.md`
- Main project: `../README.md`
- Edge deployment: `../edge/README.md`

## ✅ Success Checklist

- [ ] Setup script completed without errors
- [ ] PlatformIO installed in VSCode
- [ ] Project builds successfully
- [ ] ESP32-S3 detected and uploads work
- [ ] Serial monitor shows inference results
- [ ] Inference latency < 100 ms
- [ ] Memory usage fits in PSRAM

You're ready to deploy! 🎉
