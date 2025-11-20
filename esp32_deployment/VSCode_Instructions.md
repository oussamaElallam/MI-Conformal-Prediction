# VSCode + PlatformIO Instructions for ESP32-S3

## ✅ Your Deployment Package is Ready!

All files have been generated and copied. You can now open this folder in VSCode.

## 📂 What's in This Folder

```
esp32_deployment/
├── platformio.ini              ← PlatformIO configuration
├── src/
│   └── main.cpp               ← Your ESP32 code (8.5 KB)
├── include/
│   ├── model_data.h           ← TFLite model (6.5 MB)
│   └── cp_params.h            ← Conformal parameters
├── QUICKSTART.md              ← 5-minute setup guide
├── README.md                  ← Full documentation
└── VSCode_Instructions.md     ← This file
```

## 🎯 Step-by-Step: Open in VSCode

### 1. Open VSCode

Launch Visual Studio Code on your computer.

### 2. Install PlatformIO Extension (First Time Only)

If you haven't installed PlatformIO yet:

1. Click the **Extensions** icon in the left sidebar (or press `Ctrl+Shift+X`)
2. Search for: **PlatformIO IDE**
3. Click **Install** on the official extension by PlatformIO
4. Wait for installation to complete (~1-2 minutes)
5. **Restart VSCode** when prompted

You'll know it's installed when you see a new **PlatformIO** icon (alien head) in the left sidebar.

### 3. Open This Folder as a Project

**Method A: File Menu**
1. Click **File → Open Folder**
2. Navigate to: `C:\Users\OussamaELallam\Desktop\MI paper\esp32_deployment`
3. Click **Select Folder**

**Method B: Drag and Drop**
1. Drag the `esp32_deployment` folder onto VSCode window

**Method C: Command Line**
```bash
cd "C:\Users\OussamaELallam\Desktop\MI paper\esp32_deployment"
code .
```

### 4. Wait for PlatformIO to Initialize

After opening the folder:
1. PlatformIO will detect the `platformio.ini` file
2. You'll see a progress bar at the bottom: "PlatformIO: Initializing..."
3. Wait for it to complete (~30 seconds first time)
4. A new toolbar will appear at the bottom with icons: ✓ → 🗑️ 🔌 etc.

### 5. Verify the Project Structure

In the VSCode Explorer (left sidebar), you should see:
- ✓ `platformio.ini` (with PlatformIO icon)
- ✓ `src/main.cpp`
- ✓ `include/model_data.h`
- ✓ `include/cp_params.h`

## 🔌 Connect Your ESP32-S3

### Physical Connection

1. Get a **USB-C cable** (data cable, not just charging)
2. Connect ESP32-S3 to your computer
3. Wait for Windows to detect it (~10 seconds)

### Verify Connection

**Windows:**
1. Open **Device Manager** (`Win+X` → Device Manager)
2. Expand **Ports (COM & LPT)**
3. Look for: **USB Serial Port (COM3)** or similar
4. Note the COM port number (e.g., COM3)

**If not detected:**
- Install CP210x or CH340 USB driver (depends on your ESP32-S3 board)
- Try a different USB cable (some are charge-only)
- Try a different USB port

## 🚀 Build and Upload

### Using PlatformIO Toolbar (Easiest)

Look at the **bottom toolbar** in VSCode. You'll see these icons:

| Icon | Name | Function |
|------|------|----------|
| ✓ | Build | Compile the code |
| → | Upload | Flash to ESP32 |
| 🗑️ | Clean | Delete build files |
| 🔌 | Monitor | Open serial monitor |
| 🔄 | Upload & Monitor | Upload then monitor |

**To deploy:**
1. Click **✓ Build** first (verify code compiles)
2. Click **→ Upload** (flash to ESP32)
3. Click **🔌 Monitor** (view output)

### Using PlatformIO Sidebar (Alternative)

1. Click the **PlatformIO** icon (alien head) in left sidebar
2. Expand **PROJECT TASKS → esp32-s3-devkitc-1**
3. Click **General → Build**
4. Click **General → Upload**
5. Click **General → Monitor**

### Using Terminal (Advanced)

Open VSCode terminal (`Ctrl+` ` or View → Terminal):

```bash
# Build
pio run

# Upload
pio run --target upload

# Monitor
pio device monitor

# Upload and monitor in one command
pio run --target upload && pio device monitor
```

## 📺 View Serial Output

### After Upload:

1. Click **🔌 Monitor** in bottom toolbar
2. Or run: `pio device monitor`
3. Set baud rate to **115200** (should be automatic)

### Expected Output:

```
========================================
ESP32-S3 MI Detection with Conformal Prediction
========================================

Free heap: 8388607 bytes
Free PSRAM: 8388607 bytes
Allocated 200 KB tensor arena in PSRAM
✓ Model loaded successfully
✓ Tensors allocated successfully
Input shape: [1, 1000, 12]
Output shape: [1]

Conformal Prediction Parameters:
  Epsilon (ε): 0.10 (90% coverage target)
  τ_normal: 0.4101
  τ_MI:     0.7338

========================================
Setup complete! Running demo inference...

Running inference on demo ECG signal...

========== INFERENCE RESULTS ==========
Inference latency: 87.23 ms
P(MI) = 0.3456

Conformal Prediction:
  s_norm = 0.3456 <= tau_norm = 0.4101 ? YES
  s_mi   = 0.6544 <= tau_mi   = 0.7338 ? YES
Prediction Set: {Normal, MI}
→ UNCERTAIN (both classes possible)
========================================
```

## 🎯 What the Code Does

The current `main.cpp`:
1. Initializes TensorFlow Lite Micro on ESP32-S3
2. Loads the quantized INT8 model (~6.5 MB)
3. Generates a **demo sinusoid signal** (not real ECG)
4. Runs inference every 5 seconds
5. Applies conformal prediction
6. Prints results to Serial

## 🔧 Modify the Code

### To change inference interval:

Edit `src/main.cpp`, line ~280:
```cpp
delay(5000);  // Change to 1000 for 1 second, etc.
```

### To add real ECG input:

Replace the `generate_demo_signal()` function with your ECG source:
- Serial input from Python
- SD card CSV file
- ADC sampling from ECG sensor

See `QUICKSTART.md` for examples.

## ❓ Troubleshooting

### "PlatformIO: command not found"
→ Restart VSCode after installing PlatformIO extension

### Build fails: "model_data.h: No such file"
→ Run `python setup_deployment.py` again

### Upload fails: "Serial port not found"
→ Check Device Manager for COM port
→ Press and hold **BOOT** button on ESP32 during upload

### Upload fails: "Timed out waiting for packet header"
→ Press **BOOT** button and try again
→ Try lower upload speed: add to `platformio.ini`:
```ini
upload_speed = 115200
```

### Out of memory during runtime
→ Increase `TENSOR_ARENA_SIZE` in `main.cpp`
→ Verify PSRAM is enabled (should be automatic)

### Garbage in serial monitor
→ Check baud rate is 115200
→ Press **RESET** button on ESP32

## 📊 Performance Monitoring

### Measure Inference Latency:

The code automatically reports latency. Look for:
```
Inference latency: XX.XX ms
```

**Target:** < 100 ms for real-time processing

### Check Memory Usage:

Look for these lines in serial output:
```
Free heap: XXXXX bytes
Free PSRAM: XXXXX bytes
```

**ESP32-S3 has:**
- 512 KB SRAM
- 8 MB PSRAM (external)

### Monitor Power Consumption:

Use a USB power meter to measure:
- Idle: ~50 mA
- During inference: ~150 mA
- Peak: ~200 mA

## 🎓 Learning Resources

### PlatformIO Docs:
- https://docs.platformio.org/en/latest/

### ESP32-S3 Docs:
- https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/

### TensorFlow Lite Micro:
- https://www.tensorflow.org/lite/microcontrollers

## ✅ Success Checklist

- [ ] VSCode installed
- [ ] PlatformIO extension installed
- [ ] Project opens without errors
- [ ] ESP32-S3 detected in Device Manager
- [ ] Build completes successfully (✓ icon)
- [ ] Upload completes successfully (→ icon)
- [ ] Serial monitor shows output (🔌 icon)
- [ ] Inference latency < 100 ms
- [ ] No memory errors

## 🎉 You're Ready!

Once you see the inference results in the serial monitor, you have successfully deployed the MI detection model on ESP32-S3!

**Next steps:**
1. Replace demo signal with real ECG data
2. Optimize inference latency
3. Add BLE/WiFi for remote monitoring
4. Implement continuous sliding-window inference

See `QUICKSTART.md` and `README.md` for more details.
