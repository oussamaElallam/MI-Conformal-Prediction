# ESP32-S3 Deployment Package

This folder contains everything needed to deploy the MI detection model with conformal prediction on ESP32-S3.

## Hardware Requirements

- **ESP32-S3-DevKitC-1** (or compatible)
- USB-C cable
- Computer with VSCode

## Software Requirements

1. **Visual Studio Code**
2. **PlatformIO Extension** for VSCode
   - Install from VSCode Extensions marketplace
   - Search for "PlatformIO IDE"

## Project Structure

```
esp32_deployment/
├── platformio.ini          # PlatformIO configuration
├── src/
│   └── main.cpp           # Main application code
├── include/
│   ├── model_data.h       # TFLite model (auto-generated)
│   └── cp_params.h        # Conformal prediction parameters
└── README.md              # This file
```

## Setup Instructions

### 1. Open Project in VSCode

1. Open VSCode
2. Click **File → Open Folder**
3. Select the `esp32_deployment` folder
4. PlatformIO will automatically detect the project

### 2. Generate Model Files (First Time Only)

Before building, you need to generate the model files:

```bash
# From the main project directory
cd ..

# 1. Export TFLite model
python edge/export_tflite_micro_model.py

# 2. Compute conformal thresholds
python edge/compute_cp_thresholds.py

# 3. Convert model to C array
python edge/tflite_to_cc.py
```

This will create:
- `edge/model/model_int8.tflite` - Quantized model
- `edge/model/model_data.cc` - Model as C array
- `edge/model/model_data.h` - Model header
- `edge/model/cp_params.h` - Conformal parameters

### 3. Copy Generated Files

The build script will automatically copy files from `../edge/model/` to `include/`.

Alternatively, manually copy:
```bash
cp ../edge/model/model_data.h include/
cp ../edge/model/cp_params.h include/
```

### 4. Build and Upload

#### Using PlatformIO GUI:
1. Click the PlatformIO icon in VSCode sidebar
2. Click **Build** (checkmark icon)
3. Connect ESP32-S3 via USB
4. Click **Upload** (arrow icon)
5. Click **Monitor** (plug icon) to view serial output

#### Using PlatformIO CLI:
```bash
# Build
pio run

# Upload
pio run --target upload

# Monitor serial output
pio device monitor
```

## Testing the Deployment

### 1. Initial Test (Demo Signal)

The code includes a demo sinusoid signal. After upload, you should see:

```
Starting ECG CP demo (ESP32S3)
Free heap before: XXXXX bytes
pMI=0.XXXX | s_norm=0.XXXX <= tau_norm=0.4101 ? yes/no | s_mi=0.XXXX <= tau_mi=0.7338 ? yes/no
Prediction set: {Normal, MI} or {Normal} or {MI}
Average inference latency: XX.XX us (50 runs)
Free heap after: XXXXX bytes
```

### 2. Real ECG Data Test

To test with real ECG data:

1. **Option A: Serial Input**
   - Modify `main.cpp` to read ECG samples from Serial
   - Send 12000 floats (1000 samples × 12 leads) via Serial

2. **Option B: SD Card**
   - Add SD card support
   - Load ECG samples from CSV file

3. **Option C: ADC Input**
   - Connect ECG sensor to ESP32-S3 ADC pins
   - Sample at 100 Hz per lead

## Performance Metrics

Expected performance on ESP32-S3 @ 240 MHz:

- **Inference Latency**: ~50-100 ms per 10-second window
- **Memory Usage**: 
  - Model: ~1 MB
  - Tensor Arena: ~200 KB
  - Total RAM: ~1.5 MB (fits in 8 MB PSRAM)
- **Power Consumption**: ~150 mA @ 3.3V during inference

## Troubleshooting

### Build Errors

**Error: `model_data.h` not found**
- Run the model generation scripts first (see Setup step 2)

**Error: Out of memory**
- Increase `kTensorArenaSize` in `main.cpp`
- Enable PSRAM in `platformio.ini` (already enabled)

**Error: Upload failed**
- Check USB cable connection
- Press BOOT button on ESP32-S3 while uploading
- Try lower upload speed: `upload_speed = 115200`

### Runtime Errors

**Error: Model schema mismatch**
- Regenerate the TFLite model with matching TensorFlow version

**Error: AllocateTensors failed**
- Increase `kTensorArenaSize` in `main.cpp`

**Unexpected predictions**
- Verify normalization statistics match training
- Check input data format (1000 samples × 12 leads)

## Modifying the Code

### Change Conformal Prediction Threshold (ε)

Edit `../edge/compute_cp_thresholds.py`:
```python
EPSILON = 0.10  # Change this (0.05 = 95% coverage, 0.10 = 90% coverage)
```

Then regenerate `cp_params.h`.

### Add Custom Input Source

Edit `src/main.cpp`:
```cpp
// Replace the demo signal generation with your input source
// Example: Read from Serial, SD card, or ADC
```

### Optimize Performance

1. **Reduce Tensor Arena Size**: Lower `kTensorArenaSize` if memory is tight
2. **Disable Logging**: Set `CORE_DEBUG_LEVEL=0` in `platformio.ini`
3. **Overclock**: Increase CPU frequency (up to 240 MHz)

## Next Steps

1. ✅ Build and upload the demo
2. ✅ Verify serial output shows predictions
3. ✅ Measure actual inference latency
4. ⬜ Integrate real ECG input source
5. ⬜ Add BLE/WiFi for remote monitoring
6. ⬜ Implement continuous inference with sliding window

## Support

For issues:
1. Check the main project `edge/README.md`
2. Review PlatformIO documentation: https://docs.platformio.org/
3. Check ESP32-S3 documentation: https://docs.espressif.com/

## License

Same as main project.
