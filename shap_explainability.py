import numpy as np
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from scipy.interpolate import splev, splrep

# ==================== CONFIG ====================
MODEL_PATH = r"D:\Sleep Apnea Project1\code_baseline\weights.best.keras"
PICKLE_PATH = r"C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0\apnea-ecg.pkl"

NUM_BACKGROUND = 80
NUM_EXPLAIN = 250

IR = 3
BEFORE, AFTER = 2, 2
TM = np.arange(0, (BEFORE + 1 + AFTER) * 60, step=1 / float(IR))

scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

feature_names = ['R-R Interval (norm)', 'R-peak Amplitude (norm)']

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

# ==================== SHAP ====================
print("Building explainer...")
explainer = shap.DeepExplainer(model, background)

print("Computing SHAP values...")
shap_values = explainer.shap_values(X_explain)

# Apnea class (1) from 5-min branch
shap_apnea_5min = shap_values[0][..., 1]  # (samples, time, features)

# Predictions for selecting best sample
predicted_probs = model.predict(X_explain, verbose=0)[:, 1]
highest_idx = np.argmax(predicted_probs)

# ==================== PLOTS ====================
# 1. Global summary
shap_mean_abs = np.abs(shap_apnea_5min).mean(axis=1)
X_mean = X_explain[0].mean(axis=1)

plt.figure(figsize=(10, 5))
shap.summary_plot(shap_mean_abs, X_mean, feature_names=feature_names, show=False)
plt.title("SHAP Global Importance (5-min branch)")
plt.savefig("shap_summary_global.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: shap_summary_global.png")

# 2. Heatmap for strongest prediction
plt.figure(figsize=(14, 5))
plt.imshow(shap_apnea_5min[highest_idx].T, aspect='auto', cmap='RdBu_r', vmin=-0.08, vmax=0.08)
plt.colorbar(label='SHAP value')
plt.xlabel('Time steps')
plt.ylabel('Feature (0=RRI, 1=Amplitude)')
plt.title(f"SHAP Heatmap – Highest Apnea Prob {predicted_probs[highest_idx]:.3f}")
plt.savefig("shap_heatmap_highest.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: shap_heatmap_highest.png")

# 3. Waterfall – time-step contributions for RRI channel
values_rri = shap_apnea_5min[highest_idx, :, 0]  # only RRI channel
data_rri = X_explain[0][highest_idx, :, 0]

# Safe base value extraction
ev = explainer.expected_value
if isinstance(ev, (list, tuple)):
    base_value = float(ev[1]) if len(ev) >= 2 else float(ev[0])
elif hasattr(ev, 'shape') and len(ev.shape) > 0:
    base_value = float(ev[1]) if len(ev) >= 2 else float(ev[0])
else:
    base_value = float(ev)

plt.figure(figsize=(12, 8))
shap.waterfall_plot(
    shap.Explanation(
        values=values_rri,
        base_values=base_value,
        data=data_rri,
        feature_names=[f"t{t+1}" for t in range(len(values_rri))]
    ),
    max_display=25, show=False
)
plt.title("SHAP Waterfall – R-R Interval Time Steps (Highest Apnea)")
plt.savefig("shap_waterfall_rri_time.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved: shap_waterfall_rri_time.png")

print("\nAll plots generated successfully.")
print("Check your project folder for:")
print("- shap_summary_global.png")
print("- shap_heatmap_highest.png")
print("- shap_waterfall_rri_time.png")