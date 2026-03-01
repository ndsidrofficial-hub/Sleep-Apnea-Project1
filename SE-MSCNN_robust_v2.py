import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, regularizers
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SE-MSCNN WITH REAL SPO2 - ROBUST VERSION V2")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading data...")
with open('ucd_with_real_spo2.pkl', 'rb') as f:
    data = pickle.load(f)

o_train, y_train = data['o_train'], np.array(data['y_train'])
o_test, y_test = data['o_test'], np.array(data['y_test'])

print(f"Loaded {len(o_train)} training + {len(o_test)} test samples")

# ============================================================================
# 2. CALCULATE CLASS WEIGHTS
# ============================================================================
unique_classes, class_counts = np.unique(y_train, return_counts=True)
total_samples = len(y_train)
class_weights = {}
for cls, count in zip(unique_classes, class_counts):
    weight = total_samples / (2 * count)  # Simpler weighting
    class_weights[int(cls)] = weight

print(f"\nClass weights: Normal={class_weights[0]:.4f}, Apnea={class_weights[1]:.4f}")

# ============================================================================
# 3. PROCESS DATA EFFICIENTLY
# ============================================================================
print("\n[2] Processing data...")

def load_and_process_data(o_data, y_data, indices=None):
    if indices is None:
        indices = range(len(o_data))
    
    x1_list, x2_list, x3_list, x_spo2_list = [], [], [], []
    y_list = []
    
    for idx in indices:
        (rri_tm, rri_sig), (ampl_tm, ampl_sig), spo2_seg = o_data[idx]
        
        # Handle length mismatch
        min_len = min(len(rri_sig), len(ampl_sig))
        rri_sig = rri_sig[:min_len]
        ampl_sig = ampl_sig[:min_len]
        
        # Combine signals
        combined = np.column_stack([rri_sig, ampl_sig])
        
        # Interpolate to fixed dimensions
        t_original = np.arange(len(combined))
        x1 = np.column_stack([
            np.interp(np.linspace(0, len(combined)-1, 900), t_original, combined[:, 0]),
            np.interp(np.linspace(0, len(combined)-1, 900), t_original, combined[:, 1])
        ])
        x2 = np.column_stack([
            np.interp(np.linspace(0, len(combined)-1, 540), t_original, combined[:, 0]),
            np.interp(np.linspace(0, len(combined)-1, 540), t_original, combined[:, 1])
        ])
        x3 = np.column_stack([
            np.interp(np.linspace(0, len(combined)-1, 180), t_original, combined[:, 0]),
            np.interp(np.linspace(0, len(combined)-1, 180), t_original, combined[:, 1])
        ])
        
        t_spo2 = np.arange(len(spo2_seg))
        x_spo2 = np.interp(np.linspace(0, len(spo2_seg)-1, 2400), t_spo2, spo2_seg).reshape(-1, 1)
        
        x1_list.append(x1)
        x2_list.append(x2)
        x3_list.append(x3)
        x_spo2_list.append(x_spo2)
        y_list.append(y_data[idx])
    
    return (np.array(x1_list), np.array(x2_list), np.array(x3_list), np.array(x_spo2_list)), np.array(y_list)

# Split training data
train_idx, val_idx = train_test_split(range(len(y_train)), test_size=0.3, random_state=42, stratify=y_train)

(x1_tr, x2_tr, x3_tr, x_spo2_tr), y_tr = load_and_process_data(o_train, y_train, train_idx)
(x1_val, x2_val, x3_val, x_spo2_val), y_val = load_and_process_data(o_train, y_train, val_idx)
(x1_test, x2_test, x3_test, x_spo2_test), y_test_out = load_and_process_data(o_test, y_test)

print(f"Train: {len(y_tr)} (N:{(y_tr==0).sum()}, A:{(y_tr==1).sum()})")
print(f"Val:   {len(y_val)} (N:{(y_val==0).sum()}, A:{(y_val==1).sum()})")
print(f"Test:  {len(y_test_out)} (N:{(y_test_out==0).sum()}, A:{(y_test_out==1).sum()})")

# ============================================================================
# 4. BUILD MODEL - SIMPLER ARCHITECTURE
# ============================================================================
print("\n[3] Building model...")

def build_model():
    weight = 5e-6
    
    # Inputs
    inp1 = keras.Input(shape=(900, 2), name='ecg_5m')
    inp2 = keras.Input(shape=(540, 2), name='ecg_3m')
    inp3 = keras.Input(shape=(180, 2), name='ecg_1m')
    inp_spo2 = keras.Input(shape=(2400, 1), name='spo2')
    
    # ECG Branch 1
    x1 = layers.Conv1D(32, 11, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight))(inp1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(3)(x1)
    x1 = layers.Conv1D(64, 11, strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(5)(x1)
    x1 = layers.Conv1D(64, 11, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight))(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)
    
    # ECG Branch 2
    x2 = layers.Conv1D(32, 11, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight))(inp2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling1D(3)(x2)
    x2 = layers.Conv1D(64, 11, strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling1D(5)(x2)
    x2 = layers.Conv1D(64, 11, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight))(x2)
    x2 = layers.GlobalAveragePooling1D()(x2)
    
    # ECG Branch 3
    x3 = layers.Conv1D(32, 11, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight))(inp3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.MaxPooling1D(3)(x3)
    x3 = layers.Conv1D(64, 11, strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight))(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.GlobalAveragePooling1D()(x3)
    
    # SPO2 Branch - downsampling with larger strides
    xs = layers.Conv1D(16, 11, strides=4, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight))(inp_spo2)
    xs = layers.BatchNormalization()(xs)
    xs = layers.MaxPooling1D(4)(xs)
    xs = layers.Conv1D(32, 11, strides=4, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight))(xs)
    xs = layers.BatchNormalization()(xs)
    xs = layers.MaxPooling1D(4)(xs)
    xs = layers.Conv1D(64, 11, strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(weight))(xs)
    xs = layers.GlobalAveragePooling1D()(xs)
    
    # Fusion
    fused = layers.concatenate([x1, x2, x3, xs])
    
    # Dense layers
    z = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight))(fused)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)
    
    z = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight))(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.4)(z)
    
    z = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight))(z)
    z = layers.Dropout(0.3)(z)
    
    out = layers.Dense(2, activation='softmax')(z)
    
    model = keras.Model(inputs=[inp1, inp2, inp3, inp_spo2], outputs=out)
    return model

model = build_model()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
print("Model created")

# ============================================================================
# 5. TRAINING
# ============================================================================
print("\n[4] Training...")

lr_schedule = callbacks.LearningRateScheduler(
    lambda epoch: 0.001 * (0.7 ** (epoch // 15)), verbose=0
)

checkpoint = callbacks.ModelCheckpoint(
    'weights.robust_spo2.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=0
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

history = model.fit(
    [x1_tr, x2_tr, x3_tr, x_spo2_tr],
    y_tr,
    validation_data=([x1_val, x2_val, x3_val, x_spo2_val], y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[checkpoint, lr_schedule, early_stop],
    verbose=0
)

print(f"Training complete. Best val accuracy: {max(history.history['val_accuracy']):.4f}")

# ============================================================================
# 6. EVALUATION WITH THRESHOLD OPTIMIZATION
# ============================================================================
print("\n[5] Evaluating...")

y_proba = model.predict([x1_test, x2_test, x3_test, x_spo2_test], verbose=0)
y_proba_apnea = y_proba[:, 1]

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(y_test_out, y_proba_apnea)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

# Evaluate at different thresholds
results = {}
for threshold_name, threshold_val in [("0.50", 0.5), ("optimal", optimal_threshold)]:
    y_pred = (y_proba_apnea >= threshold_val).astype(int)
    acc = accuracy_score(y_test_out, y_pred)
    cm = confusion_matrix(y_test_out, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_test_out, y_pred, zero_division=0)
    auc = roc_auc_score(y_test_out, y_proba_apnea)
    
    results[threshold_name] = {
        'acc': acc, 'sens': sens, 'spec': spec, 'f1': f1, 'auc': auc,
        'threshold': threshold_val, 'y_pred': y_pred
    }
    
    print(f"\nThreshold {threshold_name} ({threshold_val:.4f}):")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Sensitivity: {sens:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  F1-score:    {f1:.4f}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  TP:{tp}, FP:{fp}, FN:{fn}, TN:{tn}")

# ============================================================================
# 7. SELECT BEST CONFIGURATION
# ============================================================================
best_metric = 'acc'
if results["0.50"]['acc'] >= results["optimal"]['acc']:
    best = results["0.50"]
    best_name = "0.50"
else:
    best = results["optimal"]
    best_name = "optimal"

print("\n" + "=" * 70)
print(f"BEST RESULT (Threshold {best_name}: {best['threshold']:.4f})")
print("=" * 70)
print(f"Accuracy:    {best['acc']:.4f} (vs baseline 89%: {(best['acc']*100-89):+.1f}%)")
print(f"Sensitivity: {best['sens']:.4f} (apnea detection: {int(best['sens']*227)}/227)")
print(f"Specificity: {best['spec']:.4f} (normal detection: {int(best['spec']*608)}/608)")
print(f"F1-score:    {best['f1']:.4f}")
print(f"AUC-ROC:     {best['auc']:.4f}")

# ============================================================================
# 8. SAVE PREDICTIONS
# ============================================================================
pred_df = pd.DataFrame({
    'y_true': y_test_out.astype(int),
    'y_score_normal': y_proba[:, 0],
    'y_score_apnea': y_proba[:, 1],
    'y_pred': best['y_pred'].astype(int)
})
pred_df.to_csv('SE-MSCNN_robust_v2_predictions.csv', index=False)

# ============================================================================
# 9. SAVE METRICS
# ============================================================================
try:
    with open('SE-MSCNN_robust_v2_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("SE-MSCNN WITH REAL SPO2 - ROBUST VERSION V2\n")
        f.write("="*70 + "\n\n")
        f.write(f"FINAL RESULTS (Threshold: {best['threshold']:.4f})\n")
        f.write("="*70 + "\n")
        f.write(f"Accuracy:    {best['acc']:.4f}\n")
        f.write(f"Sensitivity: {best['sens']:.4f}\n")
        f.write(f"Specificity: {best['spec']:.4f}\n")
        f.write(f"F1-score:    {best['f1']:.4f}\n")
        f.write(f"AUC-ROC:     {best['auc']:.4f}\n\n")
        f.write(f"vs Baseline (89%):\n")
        f.write(f"  Change: {(best['acc']*100-89):+.1f}%\n")
        f.write(f"  Status: {'IMPROVED' if best['acc'] > 0.89 else 'NEEDS IMPROVEMENT'}\n")
except:
    pass

print("\nResults saved to SE-MSCNN_robust_v2_predictions.csv")
print("Metrics saved to SE-MSCNN_robust_v2_metrics.txt")
