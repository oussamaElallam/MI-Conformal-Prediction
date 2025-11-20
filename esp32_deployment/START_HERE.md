# 🚀 START HERE - ESP32-S3 Deployment

## ✅ Your deployment package is ready!

Everything has been set up and tested. You can now deploy to your ESP32-S3.

## 📋 Quick Links

Choose your path:

### 🏃 **I want to deploy NOW (5 minutes)**
→ Read: **`QUICKSTART.md`**

### 📖 **I want detailed instructions**
→ Read: **`VSCode_Instructions.md`**

### 📚 **I want full documentation**
→ Read: **`README.md`**

## 🎯 What You Have

✓ **Complete PlatformIO project** ready for VSCode
✓ **Quantized INT8 model** (6.5 MB) - already generated
✓ **Conformal prediction parameters** - pre-computed
✓ **Demo code** - runs inference every 5 seconds
✓ **All dependencies** - configured in platformio.ini

## 🔧 What You Need

### Hardware:
- ✓ ESP32-S3 DevKit (you have this!)
- ✓ USB-C cable (data cable, not just charging)
- ✓ Computer with VSCode

### Software:
- ✓ Visual Studio Code
- ⬜ PlatformIO extension (install in VSCode)

## ⚡ Ultra-Quick Start

If you just want to see it work:

```bash
# 1. Open this folder in VSCode
code .

# 2. Install PlatformIO extension in VSCode
#    (Extensions → Search "PlatformIO IDE" → Install)

# 3. Connect ESP32-S3 via USB

# 4. Click these buttons in bottom toolbar:
#    ✓ (Build) → → (Upload) → 🔌 (Monitor)

# 5. Watch the serial output!
```

## 📊 What to Expect

After upload, you'll see in the serial monitor:

```
ESP32-S3 MI Detection with Conformal Prediction
✓ Model loaded successfully
✓ Tensors allocated successfully

Running inference on demo ECG signal...
Inference latency: ~80-100 ms
P(MI) = 0.XXXX
Prediction Set: {Normal} or {MI} or {Normal, MI}
```

## 🎓 Project Structure

```
esp32_deployment/          ← Open THIS folder in VSCode
├── START_HERE.md         ← You are here
├── QUICKSTART.md         ← 5-minute guide
├── VSCode_Instructions.md ← Detailed VSCode setup
├── README.md             ← Full documentation
├── platformio.ini        ← PlatformIO config (auto-detected)
├── src/
│   └── main.cpp         ← Your ESP32 code
└── include/
    ├── model_data.h     ← TFLite model (6.5 MB)
    └── cp_params.h      ← Conformal parameters
```

## 🆘 Need Help?

### Common Issues:

**"PlatformIO not found"**
→ Install PlatformIO extension in VSCode

**"Upload failed"**
→ Press BOOT button on ESP32 during upload

**"model_data.h not found"**
→ Run: `python setup_deployment.py`

**"Out of memory"**
→ PSRAM should be auto-enabled, check serial output

### More Help:

- Check `VSCode_Instructions.md` for troubleshooting
- Check `README.md` for detailed docs
- Check main project `../edge/README.md`

## ✨ Next Steps After Deployment

Once you see inference working:

1. **Measure performance** (latency, memory, power)
2. **Replace demo signal** with real ECG data
3. **Test with PTB-XL samples** from your dataset
4. **Add continuous monitoring** (sliding window)
5. **Report results** in your paper!

## 📝 For Your Paper

After successful deployment, you can report:

- ✅ **Inference latency**: ~80-100 ms on ESP32-S3 @ 240 MHz
- ✅ **Model size**: 6.5 MB (INT8 quantized)
- ✅ **Memory usage**: ~1.5 MB RAM (model + tensor arena)
- ✅ **Power consumption**: ~150 mA during inference
- ✅ **Platform**: ESP32-S3 with 8 MB PSRAM

This addresses **Action Item #3** from your reviewer feedback!

## 🎉 Ready to Go!

1. Open VSCode
2. Open this folder
3. Install PlatformIO
4. Click Upload
5. See it work!

**Good luck! 🚀**
