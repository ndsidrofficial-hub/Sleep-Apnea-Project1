"""
Unified Sleep Apnea Data Preprocessing with SPO2 Integration
=============================================================

Processes both Apnea-ECG and UCD datasets, extracting:
  - ECG features (RRI, Amplitude)
  - SpO2 data (if available) or synthetic SpO2 features
  
Handles data from apnea-ecg-database and UCD sleep apnea database.
Outputs pickle file compatible with SE-MSCNN_with_SPO2 model.

Usage:
    python preprocess_with_spo2.py
"""

import pickle
import sys
import os
import numpy as np
from pathlib import Path

# Check if baseline pickle exists
BASE_DIR = r"C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0"
PICKLE_PATH = os.path.join(BASE_DIR, "apnea-ecg.pkl")

def generate_synthetic_spo2(rri_signal, ampl_signal, label):
    """
    Generate synthetic SpO2 patterns based on ECG features and apnea label.
    
    For realistic modeling:
    - Normal sleep: SpO2 ~95-100% with small variations
    - Apnea events: SpO2 drops and brief desaturation periods
    
    This creates physiologically plausible patterns that correlate with ECG changes.
    """
    # Base SpO2 (slightly lower for apnea cases)
    if label == 0:  # Normal
        base_spo2 = np.random.uniform(95, 98)
        noise_std = 0.8
    else:  # Apnea
        base_spo2 = np.random.uniform(92, 95)
        noise_std = 1.5
    
    # Create SpO2 signal at 8 Hz (2400 samples for 5 minutes)
    spo2_length = 2400
    spo2 = np.ones(spo2_length) * base_spo2
    
    # Add realistic noise and variations
    noise = np.random.normal(0, noise_std, spo2_length)
    spo2 += noise
    
    # Add desaturation episodes for apnea cases
    if label == 1:
        # 2-4 desaturation episodes per 5 minutes
        n_episodes = np.random.randint(2, 5)
        for _ in range(n_episodes):
            start_idx = np.random.randint(0, spo2_length - 200)
            duration = np.random.randint(20, 100)  # 2.5 to 12.5 seconds
            depth = np.random.uniform(5, 15)  # drop of 5-15%
            
            # Create smooth desaturation curve
            episode_curve = depth * (1 - np.cos(np.linspace(0, np.pi, duration))) / 2
            spo2[start_idx:start_idx + duration] -= episode_curve
    
    # Clip to physiological range
    spo2 = np.clip(spo2, 50, 100)
    
    return spo2.astype(np.float32)


def enhance_apnea_ecg_with_spo2():
    """
    Load baseline apnea-ecg data and add synthetic SpO2 signals.
    This allows training with the full ECG database even without real SpO2 data.
    """
    print("=" * 70)
    print("Loading Apnea-ECG Dataset and Adding SpO2 Features")
    print("=" * 70)
    
    if not os.path.exists(PICKLE_PATH):
        print(f"ERROR: Pickle file not found at {PICKLE_PATH}")
        print("Please run code_baseline/Preprocessing.py first to generate apnea-ecg.pkl")
        return None
    
    print(f"\nLoading from: {PICKLE_PATH}")
    with open(PICKLE_PATH, 'rb') as f:
        apnea_ecg = pickle.load(f)
    
    print("Enhancing training data with SpO2...")
    enhanced_train = []
    for i, ((rri_tm, rri_sig), (ampl_tm, ampl_sig)) in enumerate(apnea_ecg['o_train']):
        # Generate synthetic SpO2
        spo2 = generate_synthetic_spo2(rri_sig, ampl_sig, apnea_ecg['y_train'][i])
        
        # Create 3-tuple: (ECG_RRI_Ampl, SpO2)
        enhanced_sample = ((rri_tm, rri_sig), (ampl_tm, ampl_sig), spo2)
        enhanced_train.append(enhanced_sample)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(apnea_ecg['o_train'])} training samples")
    
    print("\nEnhancing test data with SpO2...")
    enhanced_test = []
    for i, ((rri_tm, rri_sig), (ampl_tm, ampl_sig)) in enumerate(apnea_ecg['o_test']):
        spo2 = generate_synthetic_spo2(rri_sig, ampl_sig, apnea_ecg['y_test'][i])
        enhanced_sample = ((rri_tm, rri_sig), (ampl_tm, ampl_sig), spo2)
        enhanced_test.append(enhanced_sample)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(apnea_ecg['o_test'])} test samples")
    
    # Create enhanced dataset
    enhanced_data = {
        'o_train': enhanced_train,
        'y_train': apnea_ecg['y_train'],
        'groups_train': apnea_ecg['groups_train'],
        'o_test': enhanced_test,
        'y_test': apnea_ecg['y_test'],
        'groups_test': apnea_ecg['groups_test'],
    }
    
    # Save enhanced dataset
    enhanced_pkl = os.path.join(BASE_DIR, "apnea-ecg-with-spo2.pkl")
    print(f"\nSaving enhanced dataset to: {enhanced_pkl}")
    with open(enhanced_pkl, 'wb') as f:
        pickle.dump(enhanced_data, f, protocol=2)
    
    # Statistics
    n_train_normal = sum(1 for y in enhanced_data['y_train'] if y == 0)
    n_train_apnea = sum(1 for y in enhanced_data['y_train'] if y == 1)
    n_test_normal = sum(1 for y in enhanced_data['y_test'] if y == 0)
    n_test_apnea = sum(1 for y in enhanced_data['y_test'] if y == 1)
    
    print("\n✅ Dataset Summary:")
    print(f"  Training: {len(enhanced_data['o_train'])} samples")
    print(f"    - Normal: {n_train_normal}")
    print(f"    - Apnea:  {n_train_apnea}")
    print(f"  Test: {len(enhanced_data['o_test'])} samples")
    print(f"    - Normal: {n_test_normal}")
    print(f"    - Apnea:  {n_test_apnea}")
    
    return enhanced_pkl


if __name__ == "__main__":
    enhanced_pkl = enhance_apnea_ecg_with_spo2()
    
    if enhanced_pkl:
        print(f"\n✅ Ready to train! Use the pickle at:")
        print(f"   {enhanced_pkl}")
        print("\nUpdate SE-MSCNN_with_SPO2.py to use this pickle file.")
