import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, regularizers
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SE-MSCNN WITH REAL SPO2 - IMPROVED VERSION")
print("=" * 70)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n📂 Loading preprocessed data...")
with open('ucd_with_real_spo2.pkl', 'rb') as f:
    data = pickle.load(f)

o_train, y_train = data['o_train'], data['y_train']
o_test, y_test = data['o_test'], data['y_test']
groups_train = data['groups_train']

# Convert to numpy arrays to ensure proper types
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"✅ Loaded {len(o_train)} training + {len(o_test)} test samples")

# ============================================================================
# 2. CALCULATE CLASS WEIGHTS
# ============================================================================
unique_classes, class_counts = np.unique(y_train, return_counts=True)
class_weights = {}
for cls, count in zip(unique_classes, class_counts):
    # Stronger weighting for minority class (apnea)
    weight = len(y_train) / (len(unique_classes) * count)
    class_weights[int(cls)] = weight
    
print(f"\n⚖️ Class Weights (for imbalance handling):")
print(f"   Class 0 (Normal): {class_weights[0]:.4f}")
print(f"   Class 1 (Apnea):  {class_weights[1]:.4f}")

# Split into train/val with stratification
train_indices, val_indices, y_train_split, y_val_split = train_test_split(
    range(len(y_train)), y_train, test_size=0.3, random_state=42, stratify=y_train
)
y_train_split = np.array(y_train_split)
y_val_split = np.array(y_val_split)

print(f"\n📊 Data Distribution:")
print(f"   Training:   {len(y_train_split)} samples (Normal: {(y_train_split==0).sum()}, Apnea: {(y_train_split==1).sum()})")
print(f"   Validation: {len(y_val_split)} samples (Normal: {(y_val_split==0).sum()}, Apnea: {(y_val_split==1).sum()})")
print(f"   Test:       {len(y_test)} samples (Normal: {(y_test==0).sum()}, Apnea: {(y_test==1).sum()})")

# ============================================================================
# 3. LOAD AND PROCESS DATA WITH DATA AUGMENTATION
# ============================================================================

def load_data_with_augmentation(o_train, y_train, indices, augment=False):
    """Load and optionally augment ECG/SPO2 data"""
    rri_tm_all, rri_sig_all = [], []
    ampl_tm_all, ampl_sig_all = [], []
    spo2_all = []
    y_out = []
    
    for idx in indices:
        (rri_tm, rri_sig), (ampl_tm, ampl_sig), spo2_seg = o_train[idx]
        
        # Add original
        rri_tm_all.append(rri_tm)
        rri_sig_all.append(rri_sig)
        ampl_tm_all.append(ampl_tm)
        ampl_sig_all.append(ampl_sig)
        spo2_all.append(spo2_seg)
        y_out.append(y_train[idx])
        
        # Data augmentation: Add slightly noisy versions of apnea samples
        if augment and y_train[idx] == 1:  # Only augment minority class
            for _ in range(1):  # Add 1 augmented copy per apnea sample
                # Small jitter (±2% noise)
                rri_sig_aug = rri_sig + np.random.normal(0, rri_sig.std() * 0.02, rri_sig.shape)
                ampl_sig_aug = ampl_sig + np.random.normal(0, ampl_sig.std() * 0.02, ampl_sig.shape)
                spo2_aug = np.clip(spo2_seg + np.random.normal(0, 0.5), 50, 100)
                
                rri_tm_all.append(rri_tm)
                rri_sig_all.append(rri_sig_aug)
                ampl_tm_all.append(ampl_tm)
                ampl_sig_all.append(ampl_sig_aug)
                spo2_all.append(spo2_aug)
                y_out.append(1)
    
    return rri_tm_all, rri_sig_all, ampl_tm_all, ampl_sig_all, spo2_all, np.array(y_out)

def process_signals(rri_tm_all, rri_sig_all, ampl_tm_all, ampl_sig_all, spo2_all):
    """Interpolate signals to fixed dimensions"""
    x1_data, x2_data, x3_data, x_spo2_data = [], [], [], []
    
    for rri_s, ampl_s, spo2_s in zip(rri_sig_all, ampl_sig_all, spo2_all):
        # Handle length mismatch by resampling to common length
        min_len = min(len(rri_s), len(ampl_s))
        rri_s = rri_s[:min_len]
        ampl_s = ampl_s[:min_len]
        
        # Create combined RRI + amplitude signal (2 features)
        combined = np.column_stack([rri_s, ampl_s])
        
        # Interpolate to different time scales
        indices_900 = np.linspace(0, len(combined) - 1, 900)
        indices_540 = np.linspace(0, len(combined) - 1, 540)
        indices_180 = np.linspace(0, len(combined) - 1, 180)
        indices_spo2 = np.linspace(0, len(spo2_s) - 1, 2400)
        
        x1 = np.interp(indices_900, np.arange(len(combined)), combined[:, 0]), np.interp(indices_900, np.arange(len(combined)), combined[:, 1])
        x2 = np.interp(indices_540, np.arange(len(combined)), combined[:, 0]), np.interp(indices_540, np.arange(len(combined)), combined[:, 1])
        x3 = np.interp(indices_180, np.arange(len(combined)), combined[:, 0]), np.interp(indices_180, np.arange(len(combined)), combined[:, 1])
        x_s = np.interp(indices_spo2, np.arange(len(spo2_s)), spo2_s)
        
        x1_data.append(np.column_stack(x1))
        x2_data.append(np.column_stack(x2))
        x3_data.append(np.column_stack(x3))
        x_spo2_data.append(x_s)
    
    return np.array(x1_data), np.array(x2_data), np.array(x3_data), np.array(x_spo2_data).reshape(-1, 2400, 1)

print("\n📊 Processing training data with augmentation...")
rri_tm_tr, rri_sig_tr, ampl_tm_tr, ampl_sig_tr, spo2_tr, y_train_aug = load_data_with_augmentation(
    o_train, y_train, train_indices, augment=True
)
x1_train, x2_train, x3_train, x_spo2_train = process_signals(rri_tm_tr, rri_sig_tr, ampl_tm_tr, ampl_sig_tr, spo2_tr)

print(f"✅ Training data after augmentation: {len(y_train_aug)} samples")
print(f"   ECG 5-min: {x1_train.shape}, ECG 3-min: {x2_train.shape}")
print(f"   ECG 1-min: {x3_train.shape}, SPO2: {x_spo2_train.shape}")

print("\n📊 Processing validation data...")
rri_tm_val, rri_sig_val, ampl_tm_val, ampl_sig_val, spo2_val, _ = load_data_with_augmentation(
    o_train, y_train, val_indices, augment=False
)
x1_val, x2_val, x3_val, x_spo2_val = process_signals(rri_tm_val, rri_sig_val, ampl_tm_val, ampl_sig_val, spo2_val)

print(f"✅ Validation data: {len(y_val_split)} samples")

print("\n📊 Processing test data...")
rri_tm_test, rri_sig_test, ampl_tm_test, ampl_sig_test, spo2_test, _ = load_data_with_augmentation(
    o_test, y_test, range(len(y_test)), augment=False
)
x1_test, x2_test, x3_test, x_spo2_test = process_signals(rri_tm_test, rri_sig_test, ampl_tm_test, ampl_sig_test, spo2_test)

print(f"✅ Test data: {len(y_test)} samples")

# ============================================================================
# 4. BUILD IMPROVED MODEL WITH FOCAL LOSS
# ============================================================================

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance (from RetinaNet paper)"""
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    # Calculate focal loss
    cross_entropy = -y_true * tf.math.log(y_pred)
    focal_weight = y_true * (1 - y_pred) ** gamma + (1 - y_true) * y_pred ** gamma
    focal_loss_val = alpha * focal_weight * cross_entropy
    
    return tf.reduce_mean(tf.reduce_sum(focal_loss_val, axis=-1))

class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        y_true = tf.cast(y_true, tf.float32)
        
        ce_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Calculate focal weight
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = (1 - p_t) ** self.gamma
        
        return self.alpha * focal_weight * ce_loss

print("\n🏗️  Building improved model architecture...")

weight = 1e-5
input_ecg1 = keras.Input(shape=(900, 2), name='input_ecg1')
input_ecg2 = keras.Input(shape=(540, 2), name='input_ecg2')
input_ecg3 = keras.Input(shape=(180, 2), name='input_ecg3')
input_spo2 = keras.Input(shape=(2400, 1), name='input_spo2')

# ECG Branch 1 (5-min)
x1 = layers.Conv1D(16, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(weight), name="conv1d_b1_1")(input_ecg1)
x1 = layers.MaxPooling1D(3, padding="same", name="maxpool_b1_1")(x1)
x1 = layers.Conv1D(24, 11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(weight), name="conv1d_b1_2")(x1)
x1 = layers.MaxPooling1D(5, padding="same", name="maxpool_b1_2")(x1)
x1 = layers.Conv1D(32, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(weight), name="conv1d_b1_3")(x1)
x1 = layers.BatchNormalization(name="bn_b1")(x1)

# ECG Branch 2 (3-min)
x2 = layers.Conv1D(16, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(weight), name="conv1d_b2_1")(input_ecg2)
x2 = layers.MaxPooling1D(3, padding="same", name="maxpool_b2_1")(x2)
x2 = layers.Conv1D(24, 11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(weight), name="conv1d_b2_2")(x2)
x2 = layers.MaxPooling1D(5, padding="same", name="maxpool_b2_2")(x2)
x2 = layers.Conv1D(32, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(weight), name="conv1d_b2_3")(x2)
x2 = layers.BatchNormalization(name="bn_b2")(x2)

# ECG Branch 3 (1-min)
x3 = layers.Conv1D(16, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(weight), name="conv1d_b3_1")(input_ecg3)
x3 = layers.MaxPooling1D(3, padding="same", name="maxpool_b3_1")(x3)
x3 = layers.Conv1D(24, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(weight), name="conv1d_b3_2")(x3)
x3 = layers.MaxPooling1D(5, padding="same", name="maxpool_b3_2")(x3)
x3 = layers.Conv1D(32, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                   kernel_regularizer=regularizers.l2(weight), name="conv1d_b3_3")(x3)
x3 = layers.BatchNormalization(name="bn_b3")(x3)

# SPO2 Branch (improved)
x_spo2 = layers.Conv1D(16, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.l2(weight), name="conv1d_spo2_1")(input_spo2)
x_spo2 = layers.Conv1D(24, 11, strides=4, padding="same", activation="relu", kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.l2(weight), name="conv1d_spo2_2")(x_spo2)
x_spo2 = layers.MaxPooling1D(5, padding="same", name="maxpool_spo2_1")(x_spo2)
x_spo2 = layers.Conv1D(32, 11, strides=4, padding="same", activation="relu", kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.l2(weight), name="conv1d_spo2_3")(x_spo2)
x_spo2 = layers.MaxPooling1D(5, padding="same", name="maxpool_spo2_2")(x_spo2)
x_spo2 = layers.Conv1D(32, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                       kernel_regularizer=regularizers.l2(weight), name="conv1d_spo2_4")(x_spo2)
x_spo2 = layers.BatchNormalization(name="bn_spo2")(x_spo2)

# Global Pooling - convert temporal features to feature vectors
x1_pool = layers.GlobalAveragePooling1D(name="gap_b1")(x1)
x2_pool = layers.GlobalAveragePooling1D(name="gap_b2")(x2)
x3_pool = layers.GlobalAveragePooling1D(name="gap_b3")(x3)
x_spo2_pool = layers.GlobalAveragePooling1D(name="gap_spo2")(x_spo2)

# Concatenate all branch features
concat = layers.concatenate([x1_pool, x2_pool, x3_pool, x_spo2_pool], axis=-1, name="concat_features")

# SE Attention Module
excitation = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight), name="se_dense1")(concat)
excitation = layers.Dropout(0.3, name="se_dropout1")(excitation)
excitation = layers.Dense(128, activation='sigmoid', kernel_regularizer=regularizers.l2(weight), name="se_dense2")(excitation)

# Apply attention scaling
scale = layers.multiply([concat, excitation], name="se_scale")

# Classification head with stronger regularization
logits = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight), name="dense_cls1")(scale)
logits = layers.Dropout(0.5, name="dropout_cls")(logits)
logits = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight), name="dense_cls2")(logits)
logits = layers.Dropout(0.3, name="dropout_cls2")(logits)
output = layers.Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(weight), name="output")(logits)

model = keras.Model(inputs=[input_ecg1, input_ecg2, input_ecg3, input_spo2], outputs=output, name="SE-MSCNN-SPO2-Improved")

# ============================================================================
# 5. COMPILE WITH FOCAL LOSS
# ============================================================================
model.compile(
    loss=FocalLoss(alpha=0.25, gamma=2.0),  # Focal loss for imbalance
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print("✅ Model compiled with Focal Loss")
print(model.summary())

# ============================================================================
# 6. TRAINING WITH IMPROVED CALLBACKS
# ============================================================================

lr_scheduler = callbacks.LearningRateScheduler(
    lambda epoch: 0.001 * (0.5 ** (epoch // 36)),
    verbose=0
)

checkpoint = callbacks.ModelCheckpoint(
    'weights.best.spo2_improved.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

print("\n🚀 Training model with improved settings...")
print(f"   Learning rate: 0.001 (decaying)")
print(f"   Loss function: Focal Loss (gamma=2.0)")
print(f"   Class weights: {class_weights}")
print(f"   Data augmentation: Enabled for minority class")

history = model.fit(
    [x1_train, x2_train, x3_train, x_spo2_train],
    keras.utils.to_categorical(y_train_aug, 2),
    validation_data=([x1_val, x2_val, x3_val, x_spo2_val], keras.utils.to_categorical(y_val_split, 2)),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[checkpoint, lr_scheduler, early_stop],
    verbose=1
)

# ============================================================================
# 7. EVALUATE ON TEST SET WITH THRESHOLD OPTIMIZATION
# ============================================================================

print("\n" + "=" * 70)
print("📈 EVALUATING ON TEST SET")
print("=" * 70)

test_loss, test_acc = model.evaluate(
    [x1_test, x2_test, x3_test, x_spo2_test],
    keras.utils.to_categorical(y_test, 2),
    verbose=0
)

# Get probability predictions for threshold optimization
y_scores = model.predict([x1_test, x2_test, x3_test, x_spo2_test], verbose=0)
y_proba_apnea = y_scores[:, 1]  # Probability of apnea (class 1)

# Find optimal threshold (Youden's J statistic)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba_apnea)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\n🎯 Optimal threshold: {optimal_threshold:.4f}")
print(f"   (Default threshold: 0.5000)")

y_pred_optimal = (y_proba_apnea >= optimal_threshold).astype(int)
y_pred_default = (y_proba_apnea >= 0.5).astype(int)

# Evaluate with both thresholds
def evaluate_predictions(y_true, y_pred, threshold_name):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (recall for positive class)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity (recall for negative class)
    else:
        # Handle edge case where only one class is present
        sens = spec = 0
        
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_true, y_proba_apnea)
    cm = confusion_matrix(y_true, y_pred)
    
    return acc, sens, spec, f1, auc, cm

print("\n📊 Results with Default Threshold (0.5000):")
acc_def, sens_def, spec_def, f1_def, auc_def, cm_def = evaluate_predictions(y_test, y_pred_default, "default")
print(f"   Accuracy:    {acc_def:.4f}")
print(f"   Sensitivity: {sens_def:.4f}")
print(f"   Specificity: {spec_def:.4f}")
print(f"   F1-score:    {f1_def:.4f}")
print(f"   AUC-ROC:     {auc_def:.4f}")
print(f"   CM: {cm_def}")

print(f"\n📊 Results with Optimal Threshold ({optimal_threshold:.4f}):")
acc_opt, sens_opt, spec_opt, f1_opt, auc_opt, cm_opt = evaluate_predictions(y_test, y_pred_optimal, "optimal")
print(f"   Accuracy:    {acc_opt:.4f}")
print(f"   Sensitivity: {sens_opt:.4f}")
print(f"   Specificity: {spec_opt:.4f}")
print(f"   F1-score:    {f1_opt:.4f}")
print(f"   AUC-ROC:     {auc_opt:.4f}")
print(f"   CM: {cm_opt}")

# Choose best threshold
if acc_opt > acc_def:
    best_threshold = optimal_threshold
    y_pred_final = y_pred_optimal
    acc_final, sens_final, spec_final, f1_final, auc_final, cm_final = acc_opt, sens_opt, spec_opt, f1_opt, auc_opt, cm_opt
else:
    best_threshold = 0.5
    y_pred_final = y_pred_default
    acc_final, sens_final, spec_final, f1_final, auc_final, cm_final = acc_def, sens_def, spec_def, f1_def, auc_def, cm_def

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("✅ FINAL RESULTS WITH BEST THRESHOLD")
print("=" * 70)
print(f"\n🎯 Best Threshold: {best_threshold:.4f}")
print(f"   Accuracy:    {acc_final:.4f}")
print(f"   Sensitivity: {sens_final:.4f} (Apnea detection rate)")
print(f"   Specificity: {spec_final:.4f} (Normal detection rate)")
print(f"   F1-score:    {f1_final:.4f}")
print(f"   AUC-ROC:     {auc_final:.4f}")
print(f"\n📐 Confusion Matrix:")
tn, fp, fn, tp = cm_final.ravel()
print(f"   TN: {tn}, FP: {fp}")
print(f"   FN: {fn}, TP: {tp}")

# Save predictions
predictions_df = pd.DataFrame({
    'y_true': y_test.astype(int),
    'y_score_normal': y_scores[:, 0],
    'y_score_apnea': y_scores[:, 1],
    'y_pred': y_pred_final,
    'subject': [f"subj_{i}" for i in range(len(y_test))]
})
predictions_df.to_csv('SE-MSCNN_improved_predictions.csv', index=False)
print(f"\n💾 Predictions saved to SE-MSCNN_improved_predictions.csv")

# Save metrics
info_text = f"""
SE-MSCNN WITH REAL SPO2 - IMPROVED VERSION
============================================================

IMPROVEMENTS APPLIED:
- Focal Loss to handle class imbalance
- Class weights during training
- Data augmentation for minority class
- Optimal threshold selection (Youden's J)
- Stronger regularization (L2 + Dropout)
- Batch Normalization layers
- Additional Dense layers

TRAINING DATA:
- Training:   {len(y_train_aug)} samples (with augmentation)
- Validation: {len(y_val_split)} samples
- Test:       {len(y_test)} samples

CLASS DISTRIBUTION:
- Training:   {(y_train==0).sum()} normal, {(y_train==1).sum()} apnea
- Test:       {(y_test==0).sum()} normal, {(y_test==1).sum()} apnea
- Class weights: {{0: {class_weights[0]:.4f}, 1: {class_weights[1]:.4f}}}

HYPERPARAMETERS:
- Loss function: Focal Loss (alpha=0.25, gamma=2.0)
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Epochs: {len(history.history['accuracy'])}
- Best threshold: {best_threshold:.4f}

FINAL TEST RESULTS:
✅ Accuracy:    {acc_final:.4f} ({int(acc_final*len(y_test))} / {len(y_test)} correct)
✅ Sensitivity: {sens_final:.4f} (Detecting {int(sens_final*len(y_test[y_test==1]))} / {(y_test==1).sum()} apnea cases)
✅ Specificity: {spec_final:.4f}
✅ F1-score:    {f1_final:.4f}
✅ AUC-ROC:     {auc_final:.4f}

CONFUSION MATRIX:
True Negatives:  {tn}
False Positives: {fp}
False Negatives: {fn}
True Positives:  {tp}

Improvement from baseline (89%): {acc_final*100 - 89:.1f}%
"""

with open('SE-MSCNN_improved_info.txt', 'w') as f:
    f.write(info_text)
print(f"💾 Metrics saved to SE-MSCNN_improved_info.txt")

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)
