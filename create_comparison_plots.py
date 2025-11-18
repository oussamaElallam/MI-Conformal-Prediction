"""
Script to create combined ROC and PR curves comparing Improved CNN and ResNet1D models.
Uses actual predictions from both models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import json
import os

print("Loading predictions from both models...")

# Check if improved predictions exist
if not os.path.exists('results/improved_predictions.csv'):
    print("\nError: results/improved_predictions.csv not found!")
    print("Please run: python improved_mi_classification.py")
    print("This will generate the predictions file needed for comparison.")
    exit(1)

# Load predictions from both models
improved_preds = pd.read_csv('results/improved_predictions.csv')
resnet_preds = pd.read_csv('results/resnet1d/predictions.csv')

# Extract true labels and predicted probabilities
y_true_improved = improved_preds['y_true'].values
y_pred_improved = improved_preds['y_score'].values

y_true_resnet = resnet_preds['y_true'].values
y_pred_resnet = resnet_preds['y_score'].values

print(f"✓ Loaded {len(y_true_improved)} improved model predictions")
print(f"✓ Loaded {len(y_true_resnet)} ResNet1D predictions")

# Compute ROC curves
fpr_improved, tpr_improved, _ = roc_curve(y_true_improved, y_pred_improved)
roc_auc_improved = auc(fpr_improved, tpr_improved)

fpr_resnet, tpr_resnet, _ = roc_curve(y_true_resnet, y_pred_resnet)
roc_auc_resnet = auc(fpr_resnet, tpr_resnet)

# Compute PR curves
precision_improved, recall_improved, _ = precision_recall_curve(y_true_improved, y_pred_improved)
pr_auc_improved = average_precision_score(y_true_improved, y_pred_improved)

precision_resnet, recall_resnet, _ = precision_recall_curve(y_true_resnet, y_pred_resnet)
pr_auc_resnet = average_precision_score(y_true_resnet, y_pred_resnet)

print(f"\nMetrics:")
print(f"  - Improved CNN: ROC-AUC = {roc_auc_improved:.3f}, PR-AUC = {pr_auc_improved:.3f}")
print(f"  - ResNet1D: ROC-AUC = {roc_auc_resnet:.3f}, PR-AUC = {pr_auc_resnet:.3f}")

# Create combined ROC curve plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_improved, tpr_improved, color='#2E86AB', lw=2.5, 
         label=f'Improved CNN (AUC = {roc_auc_improved:.3f})')
plt.plot(fpr_resnet, tpr_resnet, color='#A23B72', lw=2.5, 
         label=f'ResNet1D (AUC = {roc_auc_resnet:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.title('ROC Curve Comparison: Improved CNN vs ResNet1D', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11, framealpha=0.95)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('results/roc_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: results/roc_comparison.png")
plt.close()

# Create combined PR curve plot
plt.figure(figsize=(8, 6))
plt.plot(recall_improved, precision_improved, color='#2E86AB', lw=2.5, 
         label=f'Improved CNN (AP = {pr_auc_improved:.3f})')
plt.plot(recall_resnet, precision_resnet, color='#A23B72', lw=2.5, 
         label=f'ResNet1D (AP = {pr_auc_resnet:.3f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=13)
plt.ylabel('Precision', fontsize=13)
plt.title('Precision-Recall Curve Comparison: Improved CNN vs ResNet1D', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=11, framealpha=0.95)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('results/pr_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/pr_comparison.png")
plt.close()

print("\n✓ Accurate comparison plots created successfully!")
print("These plots use the actual predictions from both models.")
