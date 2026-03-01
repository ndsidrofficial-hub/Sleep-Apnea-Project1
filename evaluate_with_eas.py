# evaluate_with_eas.py - Evaluation + EAS without retraining

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from tensorflow.keras.models import load_model
import shap
from scipy.interpolate import splev, splrep

# Paths
MODEL_PATH = r"D:\Sleep Apnea Project1\code_baseline\weights.best.keras"
PICKLE_PATH = r"C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0\apnea-ecg.pkl"

# Load model
print("Loading saved model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

# Load data
print("Loading processed data...")
with open(PICKLE_PATH, 'rb') as f:
    apnea_ecg = pickle.load(f)

# Use small test subset for speed
o_test = apnea_ecg['o_test'][:50]
y_test = apnea_ecg['y_test'][:50]
groups_test = apnea_ecg['groups_test'][:50]

# Recreate inputs
def create_inputs(samples):
    x1, x2, x3 = [], [], []
    IR = 3
    TM = np.arange(0, (2 + 1 + 2) * 60, step=1 / float(IR))
    scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
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

X_explain = create_inputs(o_test)

# Predictions
print("Running predictions...")
y_pred_prob = model.predict(X_explain, batch_size=32, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = y_test  # assuming y_test is 0/1

# EAS function (copy from your earlier file)
def compute_eas(shap_5min, predicted_probs, num_epochs=10):
    eas_scores = []
    for i in range(len(shap_5min)):
        time_steps = shap_5min.shape[1]
        steps_per_epoch = time_steps // num_epochs
        eas_epochs = []
        for e in range(num_epochs):
            start = e * steps_per_epoch
            end = min(start + steps_per_epoch, time_steps)
            epoch_shap = shap_5min[i, start:end, :]
            mean_abs = np.mean(np.abs(epoch_shap))
            eas_epochs.append(mean_abs)
        eas_epochs = np.array(eas_epochs)
        weight = predicted_probs[i]
        eas_weighted = eas_epochs * weight
        eas_score = eas_weighted.sum() / (eas_weighted.sum() + 1e-8)
        eas_scores.append(eas_score)
    return np.array(eas_scores)

# For EAS, we need SHAP — compute on small batch
print("Computing SHAP for EAS...")
background = create_inputs(apnea_ecg['o_train'][:50])  # small background
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_explain)
shap_apnea_5min = shap_values[0][..., 1]

eas_scores = compute_eas(shap_apnea_5min, y_pred_prob[:, 1])

# Save results
df = pd.DataFrame({
    'true_label': y_true,
    'pred_prob': y_pred_prob[:, 1],
    'pred_class': y_pred,
    'eas_score': eas_scores,
    'subject': groups_test
})
df.to_csv("test_with_eas.csv", index=False)
print("Saved: test_with_eas.csv")

print(f"Avg EAS - True Apnea: {df[df['true_label']==1]['eas_score'].mean():.4f}")
print(f"Avg EAS - Normal: {df[df['true_label']==0]['eas_score'].mean():.4f}")