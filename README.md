# Reliable Myocardial Infarction Detection on Edge Devices via Split-Conformal Prediction

This repository contains the complete code, models, and experimental artifacts for the paper: **"Reliable Myocardial Infarction Detection on Edge Devices via Split-Conformal Prediction"**.

It implements a comprehensive framework for:
1. **Training** a lightweight 12-lead ECG CNN for Myocardial Infarction (MI) detection.
2. **Quantifying Uncertainty** using Mondrian Split-Conformal Prediction to provide statistically valid error guarantees.
3. **Edge Deployment** on an ESP32S3 microcontroller with quantization-aware calibration.

## Project Structure

* **`requirements.txt`**: Python package dependencies.
* **`results/`**: Contains all output artifacts (metrics, ROC/PR curves, trained models).
* **`edge/`**: Toolkit for ESP32S3 deployment (Firmware + TF Lite Micro conversion).
* **`validation/`**: Scripts for external validation on the Chapman-Shaoxing dataset.

### Core Scripts
* `improved_mi_classification.py`: Trains the lightweight CNN on PTB-XL using rigorous patient-wise splitting.
* `train_split_conformal_model.py`: Calibrates the model using the Mondrian conformal prediction framework.
* `conformal_prediction_evaluation.py`: Evaluates coverage validity and efficiency (average set size).

### Analysis & Utilities
* `resnet1d_baseline.py`: Trains the ResNet1D baseline for performance comparison.
* `compare_models_mcnemar.py`: Performs statistical significance testing (McNemar's test).
* `interpretability_integrated_gradients.py`: Generates saliency maps for clinical interpretability.

## Data Setup

1. **PTB-XL Dataset:** Download from [PhysioNet](https://physionet.org/content/ptb-xl/). Extract files into the root directory.
2. **Chapman-Shaoxing Dataset:** (Optional for validation) Download from [Figshare](https://figshare.com/collections/ChapmanECG/4560497/1). Place in `chapman_shaoxing/` directory.

## How to Run the Experiments

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Reproduce Main Results (PTB-XL)
Run the pipeline in this order:
```bash
# 1. Train the Lightweight CNN
python improved_mi_classification.py

# 2. Train and Calibrate Conformal Predictor
python train_split_conformal_model.py

# 3. Evaluate Conformal Metrics (Coverage/Set Size)
python conformal_prediction_evaluation.py

# 4. Train Baseline (Optional)
python resnet1d_baseline.py
```

### 3. Edge Deployment (ESP32S3)
Navigate to the `edge/` directory. Follow the README.md inside that folder to:
1. Quantize the model to INT8.
2. Generate C++ arrays for the model and calibration thresholds.
3. Build and flash the firmware using PlatformIO.

### 4. External Validation
To reproduce the domain shift analysis on the Chapman dataset:
```bash
python -m validation.validate_on_chapman
```

## Citation
If you use this code or research in your work, please cite our paper:
```
El Allam, O., & Hamlich, M. (2025). Reliable Myocardial Infarction Detection on Edge Devices via Split-Conformal Prediction. (Under Review).
```

## License
This project is licensed under the MIT License.
