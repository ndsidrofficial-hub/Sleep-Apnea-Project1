# Week 1: Dependency Check & Real Data Loading

## ✅ Environment Verified

All required dependencies installed and ready:

### Core ML/Data
- TensorFlow: 2.20.0 ✓
- Keras: Integrated ✓
- NumPy: 2.2.3 ✓
- Pandas: 2.2.3 ✓
- Scikit-learn: 1.6.1 ✓
- SciPy: 1.15.2 ✓

### Signal Processing (Newly Installed)
- wfdb: 4.3.1 ✓ (PhysioNet data reader)
- biosppy: Latest ✓ (ECG signal processing)
- peakutils: Latest ✓ (Peak detection for biosppy)

### Visualization & Analysis
- Matplotlib: 3.10.1 ✓
- Seaborn: 0.13.2 ✓
- SHAP: 0.50.0 ✓

### Utilities
- tqdm: Available ✓ (Progress bars)

---

## 📊 What We're About to Do

### Step 1: Load Real Apnea-ECG Dataset
Using the existing `Preprocessing.py` which:
- Reads raw .dat/.hea files from PhysioNet
- Processes ECG signal (filtering, R-peak detection)
- Extracts RRI (heart rate variability) and amplitude
- Saves preprocessed data to pickle format

Dataset location: `C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0`
Expected: 65 subjects, 900+ recordings total

### Step 2: Run Improved Model on Real Data
Using `SE-MSCNN_improved_baseline.py` which:
- Loads preprocessed data
- Trains with focal loss + class weighting
- Extracts SPO2 features (8 clinical features)
- Validates on held-out test set
- Expected: 92-94% accuracy, 88-92% sensitivity

### Step 3: Generate Results
Output files:
- `weights.improved_baseline.keras` - Best model weights
- `SE-MSCNN_improved_baseline.csv` - Test predictions
- `SE-MSCNN_improved_baseline_results.txt` - Summary metrics

---

## 🚀 Ready to Proceed?

All systems: GO ✓

Next command:
```bash
cd d:\Sleep Apnea Project1\code_baseline
python Preprocessing.py
```

Estimated runtime: 30-45 minutes depending on dataset size

---
