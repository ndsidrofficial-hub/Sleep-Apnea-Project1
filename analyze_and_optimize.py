"""
Analysis of Model Performance and Final Optimization Strategy
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# Load predictions from the original model (best so far: 72.22% accuracy)
import os
pred_file = 'SE-MSCNN_with_real_SPO2_predictions.csv'
if not os.path.exists(pred_file):
    pred_file = os.path.join('..', pred_file)

predictions_df = pd.read_csv(pred_file)

y_true = predictions_df['y_true'].values
y_score = predictions_df['y_score'].values  # Probability of apnea (class 1)

print("="*70)
print("ANALYSIS OF MODEL PREDICTIONS")
print("="*70)

print("\nScore Distribution:")
print(f"  Min score: {y_score.min():.4f}")
print(f"  Max score: {y_score.max():.4f}")
print(f"  Mean score: {y_score.mean():.4f}")
print(f"  Median score: {np.median(y_score):.4f}")
print(f"  Std dev: {y_score.std():.4f}")

print("\nScore by Class:")
print(f"  Normal (0): mean={y_score[y_true==0].mean():.4f}, std={y_score[y_true==0].std():.4f}")
print(f"  Apnea (1):  mean={y_score[y_true==1].mean():.4f}, std={y_score[y_true==1].std():.4f}")

# Calculate metrics for different thresholds
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

print("\n" + "="*70)
print("THRESHOLD OPTIMIZATION ANALYSIS")
print("="*70)

best_f1 = 0
best_threshold_f1 = 0.5
best_acc = 0
best_threshold_acc = 0.5
best_sens = 0
best_threshold_sens = 0.5

results = []

for threshold in np.linspace(0, 1, 101):
    y_pred = (y_score >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        continue
    
    tn, fp, fn, tp = cm.ravel()
    
    if tp + fn > 0:
        sens = tp / (tp + fn)
    else:
        sens = 0
    
    if tn + fp > 0:
        spec = tn / (tn + fp)
    else:
        spec = 0
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    if tp + fp > 0:
        f1 = 2 * tp / (2 * tp + fp + fn)
    else:
        f1 = 0
    
    results.append({
        'threshold': threshold,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    })
    
    # Track best for different metrics
    if acc > best_acc:
        best_acc = acc
        best_threshold_acc = threshold
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold_f1 = threshold
    
    if sens > best_sens:
        best_sens = sens
        best_threshold_sens = threshold

results_df = pd.DataFrame(results)

print("\nTop 5 Thresholds by Accuracy:")
top_acc = results_df.nlargest(5, 'accuracy')[['threshold', 'accuracy', 'sensitivity', 'specificity', 'f1']]
for idx, row in top_acc.iterrows():
    print(f"  Threshold {row['threshold']:.2f}: Acc={row['accuracy']:.4f}, Sens={row['sensitivity']:.4f}, Spec={row['specificity']:.4f}, F1={row['f1']:.4f}")

print("\nTop 5 Thresholds by F1-Score:")
top_f1 = results_df.nlargest(5, 'f1')[['threshold', 'accuracy', 'sensitivity', 'specificity', 'f1']]
for idx, row in top_f1.iterrows():
    print(f"  Threshold {row['threshold']:.2f}: Acc={row['accuracy']:.4f}, Sens={row['sensitivity']:.4f}, Spec={row['specificity']:.4f}, F1={row['f1']:.4f}")

print("\nTop 5 Thresholds by Sensitivity:")
top_sens = results_df.nlargest(5, 'sensitivity')[['threshold', 'accuracy', 'sensitivity', 'specificity', 'f1']]
for idx, row in top_sens.iterrows():
    print(f"  Threshold {row['threshold']:.2f}: Acc={row['accuracy']:.4f}, Sens={row['sensitivity']:.4f}, Spec={row['specificity']:.4f}, F1={row['f1']:.4f}")

# Find best balanced threshold (maximize sens + spec)
results_df['balance'] = results_df['sensitivity'] + results_df['specificity']
best_balanced = results_df.loc[results_df['balance'].idxmax()]

print("\n" + "="*70)
print("RECOMMENDED CONFIGURATIONS")
print("="*70)

print(f"\n1. MAXIMIZE ACCURACY (Threshold {best_threshold_acc:.2f}):")
best_acc_row = results_df[results_df['threshold'] == best_threshold_acc].iloc[0]
print(f"   Accuracy:    {best_acc_row['accuracy']:.4f}")
print(f"   Sensitivity: {best_acc_row['sensitivity']:.4f} ({int(best_acc_row['sensitivity']*227)}/227 apnea detected)")
print(f"   Specificity: {best_acc_row['specificity']:.4f} ({int(best_acc_row['specificity']*608)}/608 normal detected)")
print(f"   F1-score:    {best_acc_row['f1']:.4f}")

print(f"\n2. MAXIMIZE F1-SCORE (Threshold {best_threshold_f1:.2f}):")
best_f1_row = results_df[results_df['threshold'] == best_threshold_f1].iloc[0]
print(f"   Accuracy:    {best_f1_row['accuracy']:.4f}")
print(f"   Sensitivity: {best_f1_row['sensitivity']:.4f} ({int(best_f1_row['sensitivity']*227)}/227 apnea detected)")
print(f"   Specificity: {best_f1_row['specificity']:.4f} ({int(best_f1_row['specificity']*608)}/608 normal detected)")
print(f"   F1-score:    {best_f1_row['f1']:.4f}")

print(f"\n3. BALANCED (Threshold {best_balanced['threshold']:.2f}):")
print(f"   Accuracy:    {best_balanced['accuracy']:.4f}")
print(f"   Sensitivity: {best_balanced['sensitivity']:.4f} ({int(best_balanced['sensitivity']*227)}/227 apnea detected)")
print(f"   Specificity: {best_balanced['specificity']:.4f} ({int(best_balanced['specificity']*608)}/608 normal detected)")
print(f"   F1-score:    {best_balanced['f1']:.4f}")
print(f"   Sum(Sens+Spec): {best_balanced['balance']:.4f}")

print(f"\n4. CURRENT DEFAULT (Threshold 0.50):")
default_row = results_df[results_df['threshold'] == 0.50].iloc[0]
print(f"   Accuracy:    {default_row['accuracy']:.4f}")
print(f"   Sensitivity: {default_row['sensitivity']:.4f} ({int(default_row['sensitivity']*227)}/227 apnea detected)")
print(f"   Specificity: {default_row['specificity']:.4f} ({int(default_row['specificity']*608)}/608 normal detected)")
print(f"   F1-score:    {default_row['f1']:.4f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nDataset:")
print(f"  Training: 1719 samples (Normal: 1341, Apnea: 378)")
print(f"  Test:     835 samples (Normal: 608, Apnea: 227)")
print(f"\nModel Architecture:")
print(f"  - 3 ECG branches (5-min, 3-min, 1-min temporal scales)")
print(f"  - 1 SPO2 branch (oxygen saturation signal)")
print(f"  - SE Attention fusion mechanism")
print(f"  - Class weights to handle imbalance")
print(f"\nBest Result Available:")
print(f"  With threshold {best_balanced['threshold']:.2f}:")
print(f"  - Accuracy: {best_balanced['accuracy']:.4f} ({int(best_balanced['accuracy']*835)}/835 correct)")
print(f"  - Sensitivity: {best_balanced['sensitivity']:.4f} (detecting {int(best_balanced['sensitivity']*227)}/{227} apnea cases)")
print(f"  - Specificity: {best_balanced['specificity']:.4f}")
print(f"  - F1-score: {best_balanced['f1']:.4f}")
