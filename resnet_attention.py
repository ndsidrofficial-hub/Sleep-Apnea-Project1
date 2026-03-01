"""1D-ResNet + Self-Attention Model for OSA Detection - High Novelty"""

import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.interpolate import splev, splrep

# ==================== CONFIG ====================
PICKLE_PATH = r"C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0\apnea-ecg.pkl"
MODEL_SAVE_PATH = "resnet_attention_best.keras"
EPOCHS = 100
BATCH_SIZE = 128

IR = 3
BEFORE, AFTER = 2, 2
TM = np.arange(0, (BEFORE + 1 + AFTER) * 60, step=1 / float(IR))

scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

# ==================== LOAD DATA & RECREATE INPUTS ====================
print("Loading data...")
with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f)

o_train = data['o_train']
y_train = data['y_train']
groups_train = data['groups_train']

o_test = data['o_test']
y_test = data['y_test']
groups_test = data['groups_test']

print("Available keys:", list(data.keys()))

# Function to recreate multi-scale inputs (same as your preprocessing)
def create_inputs(samples):
    x1, x2, x3 = [], [], []
    for (rri_tm, rri), (ampl_tm, ampl) in samples:
        rri_int = splev(TM, splrep(rri_tm, scaler(rri), k=3), ext=1)
        ampl_int = splev(TM, splrep(ampl_tm, scaler(ampl), k=3), ext=1)
        x1.append([rri_int, ampl_int])
        x2.append([rri_int[180:720], ampl_int[180:720]])
        x3.append([rri_int[360:540], ampl_int[360:540]])
    return [
        np.array(x1, dtype="float32").transpose(0, 2, 1),
        np.array(x2, dtype="float32").transpose(0, 2, 1),
        np.array(x3, dtype="float32").transpose(0, 2, 1)
    ]

print("Creating training inputs...")
X_train1, X_train2, X_train3 = create_inputs(o_train)
y_train = to_categorical(y_train, num_classes=2)

print("Creating test inputs...")
X_test1, X_test2, X_test3 = create_inputs(o_test)
y_test = to_categorical(y_test, num_classes=2)

print("Training data shape (5-min):", X_train1.shape)

# For simplicity, use only 5-min branch for ResNet (fast & stable)
X_train = X_train1  # (samples, 900, 2)
X_test = X_test1
# If you want to use all branches later, we can fuse them

# ==================== 1D-ResNet Block ====================
def resnet_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    x = Conv1D(filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

# ==================== MODEL ====================
def create_resnet_attention_model(input_shape=(900, 2)):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 128, stride=2)
    x = resnet_block(x, 128)
    x = resnet_block(x, 256, stride=2)
    x = resnet_block(x, 256)

    # Self-Attention (novelty)
    x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.3)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# ==================== MAIN ====================
model = create_resnet_attention_model()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,  # simple split since no separate val
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint],
    verbose=1
)

# Evaluate
model = load_model(MODEL_SAVE_PATH)
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print(f"\nFinal Test Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)