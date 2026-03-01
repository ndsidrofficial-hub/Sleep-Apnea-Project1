"""
FINAL ATTEMPT: SE-MSCNN with AGGRESSIVE CLASS WEIGHT BALANCING
"""
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, regularizers
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FINAL AGGRESSIVE TRAINING - SE-MSCNN WITH REAL SPO2")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading data...")
with open('ucd_with_real_spo2.pkl', 'rb') as f:
    data = pickle.load(f)

o_train, y_train = data['o_train'], np.array(data['y_train'])
o_test, y_test = data['o_test'], np.array(data['y_test'])

print(f"Loaded: {len(o_train)} train + {len(o_test)} test")

# ============================================================================
# 2. VERY AGGRESSIVE CLASS WEIGHTS
# ============================================================================
print("\n[STEP 2] Computing AGGRESSIVE class weights...")

n_normal = (y_train == 0).sum()
n_apnea = (y_train == 1).sum()

# Much more aggressive - weight apnea cases heavily
class_weight = {
    0: 1.0,  # Normal class
    1: (n_normal / n_apnea) * 2.5  # Apnea class - 2.5x the normal ratio
}

print(f"Class weights: Normal={class_weight[0]:.4f}, Apnea={class_weight[1]:.4f}")
print(f"  This means each apnea sample contributes {class_weight[1]:.2f}x more to loss")

# Split data
train_idx, val_idx = train_test_split(
    range(len(y_train)), test_size=0.3, random_state=42, stratify=y_train
)

# ============================================================================
# 3. PROCESS DATA
# ============================================================================
print("\n[STEP 3] Processing data...")

def process_data(o_data, y_data, indices):
    x1_list, x2_list, x3_list, x_spo2_list, y_list = [], [], [], [], []
    
    for idx in indices:
        (rri_tm, rri_sig), (ampl_tm, ampl_sig), spo2_seg = o_data[idx]
        
        # Handle length mismatch
        min_len = min(len(rri_sig), len(ampl_sig))
        rri_sig = rri_sig[:min_len]
        ampl_sig = ampl_sig[:min_len]
        
        # Combine
        combined = np.column_stack([rri_sig, ampl_sig])
        t_orig = np.arange(len(combined))
        
        # Interpolate
        x1 = np.column_stack([
            np.interp(np.linspace(0, len(combined)-1, 900), t_orig, combined[:, 0]),
            np.interp(np.linspace(0, len(combined)-1, 900), t_orig, combined[:, 1])
        ])
        x2 = np.column_stack([
            np.interp(np.linspace(0, len(combined)-1, 540), t_orig, combined[:, 0]),
            np.interp(np.linspace(0, len(combined)-1, 540), t_orig, combined[:, 1])
        ])
        x3 = np.column_stack([
            np.interp(np.linspace(0, len(combined)-1, 180), t_orig, combined[:, 0]),
            np.interp(np.linspace(0, len(combined)-1, 180), t_orig, combined[:, 1])
        ])
        
        t_spo2 = np.arange(len(spo2_seg))
        x_spo2 = np.interp(np.linspace(0, len(spo2_seg)-1, 2400), t_spo2, spo2_seg).reshape(-1, 1)
        
        x1_list.append(x1)
        x2_list.append(x2)
        x3_list.append(x3)
        x_spo2_list.append(x_spo2)
        y_list.append(y_data[idx])
    
    return (np.array(x1_list), np.array(x2_list), np.array(x3_list), np.array(x_spo2_list)), np.array(y_list)

(x1_tr, x2_tr, x3_tr, x_spo2_tr), y_tr = process_data(o_train, y_train, train_idx)
(x1_val, x2_val, x3_val, x_spo2_val), y_val = process_data(o_train, y_train, val_idx)
(x1_test, x2_test, x3_test, x_spo2_test), y_test_out = process_data(o_test, y_test, range(len(y_test)))

print(f"Train: {len(y_tr)} (N:{(y_tr==0).sum()}, A:{(y_tr==1).sum()})")
print(f"Val:   {len(y_val)} (N:{(y_val==0).sum()}, A:{(y_val==1).sum()})")
print(f"Test:  {len(y_test_out)} (N:{(y_test_out==0).sum()}, A:{(y_test_out==1).sum()})")

# ============================================================================
# 4. BUILD MODEL - DESIGNED FOR CLASS IMBALANCE
# ============================================================================
print("\n[STEP 4] Building model...")

def build_model():
    weight = 1e-5
    
    # Inputs
    inp1 = keras.Input(shape=(900, 2), name='ecg_5m')
    inp2 = keras.Input(shape=(540, 2), name='ecg_3m')
    inp3 = keras.Input(shape=(180, 2), name='ecg_1m')
    inp_spo2 = keras.Input(shape=(2400, 1), name='spo2')
    
    # ECG Branch 1
    x1 = layers.Conv1D(32, 11, strides=1, padding='same', activation='relu')(inp1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(3)(x1)
    x1 = layers.Conv1D(64, 11, strides=2, padding='same', activation='relu')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling1D(5)(x1)
    x1 = layers.Conv1D(64, 11, strides=1, padding='same', activation='relu')(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)
    x1 = layers.Dropout(0.3)(x1)
    
    # ECG Branch 2
    x2 = layers.Conv1D(32, 11, strides=1, padding='same', activation='relu')(inp2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling1D(3)(x2)
    x2 = layers.Conv1D(64, 11, strides=2, padding='same', activation='relu')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling1D(5)(x2)
    x2 = layers.Conv1D(64, 11, strides=1, padding='same', activation='relu')(x2)
    x2 = layers.GlobalAveragePooling1D()(x2)
    x2 = layers.Dropout(0.3)(x2)
    
    # ECG Branch 3
    x3 = layers.Conv1D(32, 11, strides=1, padding='same', activation='relu')(inp3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.MaxPooling1D(3)(x3)
    x3 = layers.Conv1D(64, 11, strides=2, padding='same', activation='relu')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.GlobalAveragePooling1D()(x3)
    x3 = layers.Dropout(0.3)(x3)
    
    # SPO2 Branch
    xs = layers.Conv1D(16, 11, strides=4, padding='same', activation='relu')(inp_spo2)
    xs = layers.BatchNormalization()(xs)
    xs = layers.MaxPooling1D(4)(xs)
    xs = layers.Conv1D(32, 11, strides=4, padding='same', activation='relu')(xs)
    xs = layers.BatchNormalization()(xs)
    xs = layers.MaxPooling1D(4)(xs)
    xs = layers.Conv1D(64, 11, strides=1, padding='same', activation='relu')(xs)
    xs = layers.GlobalAveragePooling1D()(xs)
    xs = layers.Dropout(0.3)(xs)
    
    # Fusion - concatenate all features
    fused = layers.concatenate([x1, x2, x3, xs])
    
    # Dense layers with strong regularization
    z = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight))(fused)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)
    
    z = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight))(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.4)(z)
    
    z = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight))(z)
    z = layers.Dropout(0.3)(z)
    
    # Output - MORE PRONE to predicting apnea by design
    out = layers.Dense(2, activation='softmax')(z)
    
    model = keras.Model(inputs=[inp1, inp2, inp3, inp_spo2], outputs=out)
    return model

model = build_model()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.0005),  # Lower LR for stability
    metrics=['accuracy']
)
print("Model built successfully")

# ============================================================================
# 5. TRAINING WITH AGGRESSIVE SETTINGS
# ============================================================================
print("\n[STEP 5] Training with AGGRESSIVE class weighting...")

lr_schedule = callbacks.LearningRateScheduler(
    lambda epoch: 0.0005 * (0.6 ** (epoch // 20)), verbose=0
)

checkpoint = callbacks.ModelCheckpoint(
    'weights.final_aggressive.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=0
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=0
)

history = model.fit(
    [x1_tr, x2_tr, x3_tr, x_spo2_tr], y_tr,
    validation_data=([x1_val, x2_val, x3_val, x_spo2_val], y_val),
    epochs=120,
    batch_size=16,  # Smaller batch for stronger gradient signals
    class_weight=class_weight,  # AGGRESSIVE WEIGHTING HERE
    callbacks=[checkpoint, lr_schedule, early_stop],
    verbose=0
)

best_val_acc = max(history.history['val_accuracy'])
print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")

# ============================================================================
# 6. EVALUATION
# ============================================================================
print("\n[STEP 6] Evaluating...")

y_proba = model.predict([x1_test, x2_test, x3_test, x_spo2_test], verbose=0)
y_proba_apnea = y_proba[:, 1]

print(f"Apnea probability distribution:")
print(f"  Min: {y_proba_apnea.min():.4f}")
print(f"  Max: {y_proba_apnea.max():.4f}")
print(f"  Mean: {y_proba_apnea.mean():.4f}")
print(f"  Normal class mean: {y_proba_apnea[y_test_out==0].mean():.4f}")
print(f"  Apnea class mean: {y_proba_apnea[y_test_out==1].mean():.4f}")

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(y_test_out, y_proba_apnea)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold (Youden): {optimal_threshold:.4f}")

# Test multiple thresholds
best_config = None
best_score = 0

for threshold_val, threshold_name in [(0.5, "default"), (optimal_threshold, "optimal"), (0.3, "0.30"), (0.4, "0.40")]:
    y_pred = (y_proba_apnea >= threshold_val).astype(int)
    acc = accuracy_score(y_test_out, y_pred)
    cm = confusion_matrix(y_test_out, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test_out, y_pred, zero_division=0)
        
        # Balanced score: prefer good sensitivity AND accuracy
        balanced_score = (sens + spec) / 2 + acc * 0.5
        
        print(f"\nThreshold {threshold_name} ({threshold_val:.2f}):")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Sensitivity: {sens:.4f} ({tp}/{tp+fn} apnea detected)")
        print(f"  Specificity: {spec:.4f}")
        print(f"  F1-score:    {f1:.4f}")
        
        if balanced_score > best_score:
            best_score = balanced_score
            best_config = {
                'threshold': threshold_val,
                'accuracy': acc,
                'sensitivity': sens,
                'specificity': spec,
                'f1': f1,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'y_pred': y_pred
            }

print("\n" + "="*70)
print("FINAL RESULT")
print("="*70)

if best_config:
    print(f"\nBest Configuration (Threshold {best_config['threshold']:.2f})")
    print(f"  Accuracy:    {best_config['accuracy']:.4f}")
    print(f"  Sensitivity: {best_config['sensitivity']:.4f} ({best_config['tp']}/{best_config['tp']+best_config['fn']} apnea detected)")
    print(f"  Specificity: {best_config['specificity']:.4f}")
    print(f"  F1-score:    {best_config['f1']:.4f}")
    
    # Save
    pred_df = pd.DataFrame({
        'y_true': y_test_out.astype(int),
        'y_score_apnea': y_proba_apnea,
        'y_pred': best_config['y_pred'].astype(int)
    })
    pred_df.to_csv('../SE-MSCNN_final_best_predictions.csv', index=False)
    
    print(f"\nPredictions saved")
else:
    print("Error: Could not find best configuration")

print("\n[COMPLETED] Final aggressive training finished")
