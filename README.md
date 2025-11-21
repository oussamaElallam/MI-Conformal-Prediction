# Reliable Myocardial Infarction Detection on Edge Devices via Split-Conformal Prediction

This repository contains the complete code and artifacts for the paper, "Reliable Myocardial Infarction Detection from 12-Lead ECG via Split-Conformal Prediction on PTB-XL." It provides a framework for training a 12-lead ECG model for MI detection, augmenting it with a statistically rigorous uncertainty quantification layer using split-conformal prediction, and deploying it on an edge device (ESP32S3).

## Project Structure

- `Paper_Draft.md`: The main research paper draft.
- `Q1_MI_Conformal_Notes.md`: Detailed technical notes and experiment log.
- `requirements.txt`: Python package dependencies.

- **Data**: The project uses the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset. The validation part is designed for the [Chapman-Shaoxing](https://physionet.org/content/ecg-arrhythmia/1.0.0/) dataset. Please download them and place them in the root directory.

- **Core Scripts**:
  - `improved_mi_classification.py`: Trains the 12-lead CNN on PTB-XL with patient-wise splits.
  - `train_split_conformal_model.py`: Trains the model on a proper training set and saves a disjoint calibration set.
  - `conformal_prediction_evaluation.py`: Evaluates the model with Mondrian conformal prediction.

- **Analysis & Utilities**:
  - `resnet1d_baseline.py`: Trains a stronger ResNet1D baseline.
  - `compare_models_mcnemar.py`: Performs significance testing.
  - `export_operating_points.py`: Calculates sensitivity at fixed specificities.
  - `interpretability_integrated_gradients.py`: Generates IG saliency maps.

- `results/`: Contains all output artifacts (metrics, plots, models, etc.).
- `edge/`: Contains the toolkit for ESP32S3 deployment, including firmware and helper scripts.
- `validation/`: Contains scripts for external validation on the Chapman-Shaoxing dataset.

## How to Run the Experiments

### 1. Setup

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

Download the PTB-XL dataset and place it in the root of the project.

### 2. Main PTB-XL Experiments

Run the scripts in the following order to reproduce the main results:

1.  **Train the Improved CNN Model**:
    ```bash
    python improved_mi_classification.py
    ```

2.  **Train the Split-Conformal Model**:
    ```bash
    python train_split_conformal_model.py
    ```

3.  **Run Conformal Prediction Evaluation**:
    ```bash
    python conformal_prediction_evaluation.py
    ```

4.  **Train the ResNet1D Baseline**:
    ```bash
    python resnet1d_baseline.py
    ```

5.  **Run Significance Test**:
    ```bash
    python compare_models_mcnemar.py
    ```

### 3. Edge Deployment (ESP32S3)

Follow the detailed instructions in the `edge/README.md` file to build and flash the firmware.

### 4. External Validation

1.  Download the Chapman-Shaoxing dataset and place it in the `chapman_shaoxing` directory in the project root.
2.  Run the validation script:
    ```bash
    python -m validation.validate_on_chapman
    ```

## Citation

If you use this code or research in your work, please cite our paper:

> [Oussama El Allam]. (2025). "Reliable Myocardial Infarction Detection on Edge Devices via Split-Conformal Prediction" *[under review]*. 
