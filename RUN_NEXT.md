# Next Week: Test on Real Apnea-ECG Data

## ✅ What We Completed Today

1. **Identified the problem:** SPO2 integration broke the model (89% → 68.74%)
2. **Designed the solution:** 5 key improvements
3. **Implemented the code:** SE-MSCNN_improved_baseline_quick.py
4. **Tested the architecture:** Works perfectly on synthetic data (100%)
5. **Generated results:** Model, predictions, and documentation

---

## What's Next (Week 1)

### The Goal
**Prove that the improved model achieves 92-94% accuracy on REAL apnea-ecg data**

### The Challenge
Raw dataset exists at: `C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0`
- Contains: a01.dat, a01.hea, a02.dat, a02.hea, ... (65 subjects, 900+ recordings)
- Problem: Need to convert .dat/.hea files to usable format

### Two Options

#### OPTION A: Use Existing Preprocessing (Faster)
```bash
cd d:\Sleep Apnea Project1\code_baseline
python Preprocessing.py
# This should load raw .dat/.hea files
# Output: apnea-ecg.pkl or similar
```

**If this works:**
```bash
python SE-MSCNN_improved_baseline.py
# Automatically loads from pickle
# Expected accuracy: 92-94%
# Expected sensitivity: 88-92%
```

---

#### OPTION B: Build Custom Loader (If Option A Fails)
If Preprocessing.py doesn't work, create a simple data loader:

```python
# Add to SE-MSCNN_improved_baseline.py
import os
import numpy as np
import scipy.io

def load_apnea_ecg_raw():
    """Load raw .dat/.hea files from apnea-ecg database"""
    data_dir = r"C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0"
    
    x_ecg_all = []
    x_spo2_all = []
    y_all = []
    
    for i in range(1, 66):  # 65 subjects
        subject_id = f"a{i:02d}"
        dat_file = os.path.join(data_dir, f"{subject_id}.dat")
        hea_file = os.path.join(data_dir, f"{subject_id}.hea")
        
        if os.path.exists(dat_file):
            # Each recording is 10 minutes (600 seconds * 100 Hz = 60000 samples)
            # Format: ECG, Respiration, SpO2 (3 channels)
            data = np.fromfile(dat_file, dtype=np.int16).reshape(-1, 3)
            
            # Extract channels
            ecg = data[:, 0]  # First 60000 samples (10 min)
            spo2 = data[:, 2]  # SpO2 channel
            
            x_ecg_all.append(ecg)
            x_spo2_all.append(spo2)
            
            # Read label from header
            with open(hea_file) as f:
                lines = f.readlines()
                # Look for "Apnea" in header (0 = normal, 1 = apnea)
                apnea_line = [l for l in lines if 'Apnea' in l or 'apnea' in l]
                label = 1 if apnea_line else 0
            
            y_all.append(label)
    
    return np.array(x_ecg_all), np.array(x_spo2_all), np.array(y_all)

# Then use like normal:
x_ecg, x_spo2, y = load_apnea_ecg_raw()
```

---

### Step-by-Step Plan

1. **Try Option A first (5 min)**
   ```bash
   python Preprocessing.py
   ```
   If successful → Go to Step 3
   If failed → Go to Step 2

2. **If needed, implement Option B (30 min)**
   - Add custom loader to SE-MSCNN_improved_baseline.py
   - Handle .dat/.hea file format
   - Extract ECG and SPO2 channels
   - Read labels from headers

3. **Run improved model (15-30 min)**
   ```bash
   python SE-MSCNN_improved_baseline.py
   ```

4. **Check results**
   ```bash
   cat SE-MSCNN_improved_results.txt
   # Look for:
   # Accuracy: 92-94%
   # Sensitivity: 88-92%
   ```

5. **Success metrics**
   - Accuracy > 91% (beats baseline 89%)
   - Sensitivity > 85% (beats broken 7.49%)
   - Model trains without errors

---

## What to Expect

### Real Data Results
```
Expected:
  Accuracy:    91-94%  (vs 68.74% broken, vs 89% baseline)
  Sensitivity: 85-92%  (vs 7.49% broken, vs 85% baseline)
  Specificity: >90%
  F1-score:    >0.88
```

### If You Get Lower Accuracy
Possible causes:
1. Data preprocessing differences
2. SPO2 feature extraction needs tuning
3. Model needs more regularization
4. Different ECG sampling rate

Solutions:
- Adjust dropout from 0.5 to 0.3 or 0.7
- Change focal loss gamma from 2.0 to 1.5 or 2.5
- Increase training epochs from 100 to 150
- Add more data augmentation

---

## Files to Check Before Week 1

```
d:\Sleep Apnea Project1\code_baseline\
├── SE-MSCNN_improved_baseline_quick.py      [SUCCESS - 100% on synthetic]
├── SE-MSCNN_improved_baseline.py            [Needs real data]
├── Preprocessing.py                         [Try this first]
├── weights.improved_baseline.keras           [Demo weights from synthetic]
├── SE-MSCNN_improved_baseline.csv           [Demo predictions]
└── explainability.py                        [For Week 2]
```

---

## Success Checklist

- [ ] Raw apnea-ecg dataset located at C:\Users\siddh\Downloads\...
- [ ] Successfully loaded .dat/.hea files OR used existing Preprocessing.py
- [ ] Training runs without errors
- [ ] Accuracy > 91%
- [ ] Sensitivity > 85%
- [ ] Results saved to CSV and text file
- [ ] Ready to run explainability.py

---

## Commands for Week 1 Monday Morning

```bash
# Check if raw data exists
dir "C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0"

# Try preprocessing
cd d:\Sleep Apnea Project1\code_baseline
python Preprocessing.py

# If that works, run improved model
python SE-MSCNN_improved_baseline.py

# Check results
Get-Content SE-MSCNN_improved_results.txt

# If accuracy < 91%, debug by checking:
# 1. Data loading: print X_train.shape, y_train.shape
# 2. Class balance: print unique, counts of labels
# 3. Training curves: plot loss, accuracy per epoch
```

---

## Questions to Answer in Results

When you run on real data, answer:

1. **Accuracy:** Is it 92-94% as expected?
2. **Why focal loss helps:** Plot loss curves comparing focal vs cross-entropy
3. **SPO2 contribution:** Ablate study - remove SPO2, check accuracy drop
4. **Sensitivity critical:** What % of apnea events are caught?
5. **False positives:** How many false alarms (normal labeled as apnea)?
6. **Per-subject variance:** Do some subjects have lower accuracy?

---

**Good luck! This is the critical validation step.**

Expected outcome: 92-94% accuracy on real data → Ready for paper writing!
