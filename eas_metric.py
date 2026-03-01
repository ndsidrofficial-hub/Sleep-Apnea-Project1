import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
import shap
from scipy.interpolate import splev, splrep
from tqdm import tqdm
import seaborn as sns
import pandas as pd

# ==================== CONFIG ====================
MODEL_PATH = r"D:\Sleep Apnea Project1\code_baseline\weights.best.keras"
PICKLE_PATH = r"C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0\apnea-ecg.pkl"

NUM_BACKGROUND = 80
NUM_EXPLAIN = 250

IR = 3
BEFORE, AFTER = 2, 2
TM = np.arange(0, (BEFORE + 1 + AFTER) * 60, step=1 / float(IR))

scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

# ==================== LOAD MODEL & DATA ====================
print("Loading model...")
model = load_model(MODEL_PATH)

print("Loading data...")
with open(PICKLE_PATH, 'rb') as f:
    apnea_ecg = pickle.load(f)

o_test = apnea_ecg['o_test'][:NUM_EXPLAIN]
y_test = apnea_ecg['y_test'][:NUM_EXPLAIN]

# ==================== CREATE INPUTS ====================
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

X_explain = create_inputs(o_test)
background = create_inputs(apnea_ecg['o_train'][:NUM_BACKGROUND])

# ==================== SHAP & PREDICTIONS ====================
print("Building explainer...")
explainer = shap.DeepExplainer(model, background)

print("Computing SHAP values...")
shap_values = explainer.shap_values(X_explain)

shap_apnea_5min = shap_values[0][..., 1]  # class 1 (apnea), 5-min branch

predicted_probs = model.predict(X_explain, verbose=0)[:, 1]

# ==================== NOVEL EAS METRIC ====================
def compute_eas(shap_5min_sample, predicted_prob, num_epochs=10):
    """
    Compute Event Attribution Score (EAS):
    - Divide 5-min window into num_epochs (default 10 × 30s epochs)
    - Average absolute SHAP per epoch (across both features)
    - Weight by predicted probability (higher confidence → higher score)
    - Normalize to [0,1]
    """
    time_steps = shap_5min_sample.shape[0]
    steps_per_epoch = time_steps // num_epochs
    eas_epochs = []

    for e in range(num_epochs):
        start = e * steps_per_epoch
        end = min(start + steps_per_epoch, time_steps)
        epoch_shap = shap_5min_sample[start:end, :]
        mean_abs = np.mean(np.abs(epoch_shap))  # average influence
        eas_epochs.append(mean_abs)

    eas_epochs = np.array(eas_epochs)
    # Improved weighting: sigmoid to avoid suppressing low-confidence samples too much
    confidence_weight = 1 / (1 + np.exp(-10 * (predicted_prob - 0.5)))  # sigmoid centered at 0.5
    eas_weighted = eas_epochs * confidence_weight
    # Normalize to [0,1] using max possible contribution
    eas_score = eas_weighted.sum() / (eas_epochs.sum() + 1e-8)  # relative to total SHAP
    return eas_score, eas_weighted

# ==================== COMPUTE & VISUALIZE ====================
eas_scores = []
eas_per_sample = []

for i in range(len(shap_apnea_5min)):
    eas_score, eas_vec = compute_eas(shap_apnea_5min[i], predicted_probs[i])
    eas_scores.append(eas_score)
    eas_per_sample.append(eas_vec)

# Plot EAS breakdown for the strongest predicted sample
highest_idx = np.argmax(predicted_probs)
eas_vec = eas_per_sample[highest_idx]

plt.figure(figsize=(10, 5))
plt.bar(range(len(eas_vec)), eas_vec, color='salmon')
plt.xlabel('30-second Epoch (0 to 9)')
plt.ylabel('Weighted SHAP Contribution')
plt.title(f"EAS Breakdown – Highest Apnea Sample\nTotal EAS = {eas_scores[highest_idx]:.4f}")
plt.tight_layout()
plt.savefig("eas_breakdown_highest.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: eas_breakdown_highest.png")

print("\nEAS Results:")
print(f"Average EAS score: {np.mean(eas_scores):.4f}")
print(f"Max EAS score: {np.max(eas_scores):.4f} (sample {highest_idx})")
print("Novel EAS metric computed successfully!")

# ==================== EAS ANALYSIS & VISUALIZATION ====================

# Create DataFrame for plotting
df_eas = pd.DataFrame({
    'EAS_score': eas_scores,
    'True_Label': y_test,  # <--- FIXED: using 'y_test' instead of 'y_test_sub'
    'Predicted_Prob': predicted_probs,
    'Prediction': ['Apnea' if p > 0.5 else 'Normal' for p in predicted_probs]
})

# Boxplot: EAS by true label
plt.figure(figsize=(8, 6))
sns.boxplot(x='True_Label', y='EAS_score', data=df_eas, palette='Set2')
plt.title('EAS Score Distribution: True Apnea vs Normal')
plt.xlabel('True Label (0 = Normal, 1 = Apnea)')
plt.ylabel('EAS Score')
plt.xticks([0, 1], ['Normal', 'Apnea'])
plt.savefig("eas_distribution_by_true_label.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: eas_distribution_by_true_label.png")

# Boxplot: EAS by model prediction
plt.figure(figsize=(8, 6))
sns.boxplot(x='Prediction', y='EAS_score', data=df_eas, palette='Set1')
plt.title('EAS Score Distribution: Model Predicted Apnea vs Normal')
plt.xlabel('Model Prediction')
plt.ylabel('EAS Score')
plt.savefig("eas_distribution_by_prediction.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: eas_distribution_by_prediction.png")

# Quick stats
print("\nEAS Statistics by True Label:")
print(df_eas.groupby('True_Label')['EAS_score'].agg(['mean', 'std', 'count', 'min', 'max']))

print("\nCorrelation between EAS and predicted probability:", 
      df_eas['EAS_score'].corr(df_eas['Predicted_Prob']).round(4))