"""SE-MSCNN Baseline Model - Clean Replication"""

import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import os
import random
from tensorflow.keras.layers import Dropout, MaxPooling1D, Reshape, multiply, Conv1D, GlobalAveragePooling1D, Dense, Input, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

# ----------------------------- CONFIG -----------------------------
base_dir = r"C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0"
pickle_path = os.path.join(base_dir, "apnea-ecg.pkl")

ir = 3
before = 2
after = 2
tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

# ----------------------------- LOAD DATA -----------------------------
def load_data():
    print("Loading processed data from:", pickle_path)
    with open(pickle_path, 'rb') as f:
        apnea_ecg = pickle.load(f)

    o_train = apnea_ecg['o_train']
    y_train = apnea_ecg['y_train']
    groups_train = apnea_ecg['groups_train']
    o_test = apnea_ecg['o_test']
    y_test = apnea_ecg['y_test']
    groups_test = apnea_ecg['groups_test']

    x_train1, x_train2, x_train3 = [], [], []
    print("Processing training segments...")
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_train[i]
        rri_interp = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        x_train1.append([rri_interp, ampl_interp])
        x_train2.append([rri_interp[180:720], ampl_interp[180:720]])
        x_train3.append([rri_interp[360:540], ampl_interp[360:540]])

    x_test1, x_test2, x_test3 = [], [], []
    print("Processing test segments...")
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_test[i]
        rri_interp = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        x_test1.append([rri_interp, ampl_interp])
        x_test2.append([rri_interp[180:720], ampl_interp[180:720]])
        x_test3.append([rri_interp[360:540], ampl_interp[360:540]])

    indices = list(range(len(o_train)))
    random.shuffle(indices)
    train_idx = indices[:int(0.7 * len(indices))]
    val_idx = indices[int(0.7 * len(indices)):]

    def build_set(idx_list):
        x1, x2, x3, y, g = [], [], [], [], []
        for i in idx_list:
            x1.append(x_train1[i])
            x2.append(x_train2[i])
            x3.append(x_train3[i])
            y.append(y_train[i])
            g.append(groups_train[i])
        return np.array(x1, dtype="float32").transpose((0, 2, 1)), \
               np.array(x2, dtype="float32").transpose((0, 2, 1)), \
               np.array(x3, dtype="float32").transpose((0, 2, 1)), \
               np.array(y, dtype="float32"), g

    x_training1, x_training2, x_training3, y_training, groups_training = build_set(train_idx)
    x_val1, x_val2, x_val3, y_val, groups_val = build_set(val_idx)

    x_test1 = np.array(x_test1, dtype="float32").transpose((0, 2, 1))
    x_test2 = np.array(x_test2, dtype="float32").transpose((0, 2, 1))
    x_test3 = np.array(x_test3, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    return (x_training1, x_training2, x_training3, y_training, groups_training,
            x_val1, x_val2, x_val3, y_val, groups_val,
            x_test1, x_test2, x_test3, y_test, groups_test)


# ----------------------------- MODEL -----------------------------
def lr_schedule(epoch, lr):
    if epoch > 70 and (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate:", lr)
    return lr


def create_model(input_a_shape, input_b_shape, input_c_shape, weight=1e-3):
    input1 = Input(shape=input_a_shape)
    x1 = Conv1D(16, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight))(input1)
    x1 = Conv1D(24, 11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight))(x1)
    x1 = MaxPooling1D(3, padding="same")(x1)
    x1 = Conv1D(32, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight))(x1)
    x1 = MaxPooling1D(5, padding="same")(x1)

    input2 = Input(shape=input_b_shape)
    x2 = Conv1D(16, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight))(input2)
    x2 = Conv1D(24, 11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight))(x2)
    x2 = MaxPooling1D(3, padding="same")(x2)
    x2 = Conv1D(32, 11, strides=3, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight))(x2)

    input3 = Input(shape=input_c_shape)
    x3 = Conv1D(16, 11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight))(input3)
    x3 = Conv1D(24, 11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight))(x3)
    x3 = MaxPooling1D(3, padding="same")(x3)
    x3 = Conv1D(32, 1, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight))(x3)

    concat = concatenate([x1, x2, x3], axis=-1)
    squeeze = GlobalAveragePooling1D()(concat)
    excitation = Dense(48, activation='relu')(squeeze)
    excitation = Dense(concat.shape[-1], activation='sigmoid')(excitation)
    excitation = Reshape((1, concat.shape[-1]))(excitation)
    scale = multiply([concat, excitation])

    x = GlobalAveragePooling1D()(scale)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=[input1, input2, input3], outputs=outputs)
    return model


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    (x_training1, x_training2, x_training3, y_training, groups_training,
     x_val1, x_val2, x_val3, y_val, groups_val,
     x_test1, x_test2, x_test3, y_test, groups_test) = load_data()

    y_training = to_categorical(y_training, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    print("Input shapes:", x_training1.shape, x_training2.shape, x_training3.shape)

    model = create_model(x_training1.shape[1:], x_training2.shape[1:], x_training3.shape[1:])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint('weights.best.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks_list = [checkpoint, lr_scheduler]

    history = model.fit(
        [x_training1, x_training2, x_training3], y_training,
        batch_size=128, epochs=100,
        validation_data=([x_val1, x_val2, x_val3], y_val),
        callbacks=callbacks_list
    )

    model = load_model('weights.best.keras')
    loss, accuracy = model.evaluate([x_test1, x_test2, x_test3], y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    y_pred_prob = model.predict([x_test1, x_test2, x_test3])
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    C = confusion_matrix(y_true, y_pred, labels=[1, 0])
    TP, TN = C[0,0], C[1,1]
    FP, FN = C[1,0], C[0,1]
    acc = (TP + TN) / (TP + TN + FP + FN)
    sn = TP / (TP + FN)
    sp = TN / (TN + FP)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}, Sensitivity: {sn:.4f}, Specificity: {sp:.4f}, F1-score: {f1:.4f}")

    output = pd.DataFrame({"y_true": y_true, "y_score": y_pred_prob[:,1], "subject": groups_test})
    output.to_csv("SE-MSCNN_predictions.csv", index=False)
    print("Predictions saved to SE-MSCNN_predictions.csv")