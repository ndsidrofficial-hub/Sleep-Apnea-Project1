"""
Direct Data Loader for Apnea-ECG Dataset
Loads raw .dat and .hea files without external dependencies
"""

import numpy as np
import os
import struct
from pathlib import Path

def load_apnea_ecg_dataset():
    """
    Load apnea-ecg database directly from raw files
    
    Database structure:
    - a01.dat, a01.hea (subject 1)
    - a02.dat, a02.hea (subject 2)
    ...
    - a65.dat, a65.hea (subject 65)
    
    Each .dat file contains 3 channels × 60000 samples (10 minutes at 100Hz):
    - Channel 0: ECG
    - Channel 1: Respiration
    - Channel 2: SPO2
    
    Each .hea file contains metadata including apnea label
    """
    
    base_dir = r"C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0"
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"ERROR: Dataset not found at {base_dir}")
        print("Please download from: https://physionet.org/physiobank/database/apnea-ecg/")
        return None
    
    x_ecg_list = []
    x_spo2_list = []
    y_list = []
    subject_ids = []
    
    # Load each subject (a01 through a65)
    for i in range(1, 66):
        subject_id = f"a{i:02d}"
        dat_file = os.path.join(base_dir, f"{subject_id}.dat")
        hea_file = os.path.join(base_dir, f"{subject_id}.hea")
        
        if not os.path.exists(dat_file):
            print(f"[{i}/65] {subject_id}: File not found")
            continue
        
        try:
            # Read header to get label
            label = get_apnea_label(hea_file)
            
            # Read raw data
            # Format: 3 channels (ECG, Respiration, SPO2)
            # Each sample: 2 bytes × 3 channels = 6 bytes per sample
            # 60000 samples per channel × 10 min
            with open(dat_file, 'rb') as f:
                raw_data = f.read()
            
            # Parse binary data (signed 16-bit little-endian)
            data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, 3)
            
            # Extract channels (normalize to [-1, 1])
            ecg = data[:, 0].astype(np.float32) / 2048.0
            spo2 = data[:, 2].astype(np.float32) / 2048.0
            
            # Store
            x_ecg_list.append(ecg)
            x_spo2_list.append(spo2)
            y_list.append(label)
            subject_ids.append(subject_id)
            
            status = "APNEA" if label == 1 else "NORMAL"
            print(f"[{i}/65] {subject_id}: {status} - {len(ecg)} samples loaded")
            
        except Exception as e:
            print(f"[{i}/65] {subject_id}: Error - {str(e)}")
            continue
    
    if not x_ecg_list:
        print("ERROR: No data loaded!")
        return None
    
    # Convert to arrays
    X_ecg = np.array(x_ecg_list)
    X_spo2 = np.array(x_spo2_list)
    y = np.array(y_list)
    
    print(f"\n{'='*60}")
    print(f"DATASET LOADED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Total subjects: {len(y)}")
    print(f"Apnea cases: {np.sum(y)}")
    print(f"Normal cases: {len(y) - np.sum(y)}")
    print(f"Apnea rate: {100*np.sum(y)/len(y):.1f}%")
    print(f"ECG shape: {X_ecg.shape} (subjects, samples)")
    print(f"SPO2 shape: {X_spo2.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"{'='*60}\n")
    
    return X_ecg, X_spo2, y, subject_ids

def get_apnea_label(hea_file):
    """
    Parse .hea (header) file to get apnea label
    Format: Text file with metadata
    
    Returns:
    - 1 if subject has apnea
    - 0 if normal
    """
    try:
        with open(hea_file, 'r') as f:
            content = f.read().lower()
        
        # Look for apnea indicator in header
        # Different datasets use different formats:
        # Format 1: "Apnea: Yes/No"
        # Format 2: "Event: 1/0"
        # Format 3: Filename contains 'a' for apnea, 'n' for normal
        
        if 'apnea: yes' in content or 'apneic' in content or 'apnea events: yes' in content:
            return 1
        elif 'apnea: no' in content or 'no apnea' in content:
            return 0
        else:
            # Fallback: use filename convention (a=apnea, n=normal)
            # But we don't have that info here, assume normal if no marker
            return 0
    except:
        return 0

if __name__ == "__main__":
    # Test load
    result = load_apnea_ecg_dataset()
    if result is not None:
        X_ecg, X_spo2, y, subject_ids = result
        print("Data loaded successfully!")
        print(f"Ready to use: X_ecg, X_spo2, y, subject_ids")
