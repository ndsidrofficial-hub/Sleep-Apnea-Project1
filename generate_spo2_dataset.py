"""
Synthetic SpO2 Dataset Generator for SE-MSCNN v3 (Memory-Efficient)
===================================================================
Generates physiologically realistic SpO2 records matched to existing
ECG segments from the Apnea-ECG database.

MEMORY-EFFICIENT: Saves as individual .npy files in spo2_npy/ directory.
No ECG duplication — augmentation is done on-the-fly during training.

SpO2 physiology modeled:
  - Normal: baseline 95-99%, slow drift, sensor noise
  - Apnea:  desaturation 4-15% below baseline, 15-30s lag, exponential recovery
  - Per-patient characteristics (some desaturate more)
  - Artifact/false dips (~2% probability in normal segments)
"""

import os
import pickle
import numpy as np
import random
import gc
import sys

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_CACHE = os.path.join(BASE_DIR, "preprocessed_data.pkl")
SPO2_DIR = os.path.join(BASE_DIR, "spo2_npy")

# Temporal lengths matching existing branches
T_5MIN = 900   # 5-minute window
T_3MIN = 540   # 3-minute window
T_1MIN = 180   # 1-minute window


# ======================== SPO2 SIGNAL GENERATION ========================

def generate_spo2_signal(label, length, patient_params, rng):
    """
    Generate a single synthetic SpO2 signal for one segment.
    
    Args:
        label: 0.0 (Normal) or 1.0 (Apnea)
        length: number of time steps (900, 540, or 180)
        patient_params: dict with per-patient characteristics
        rng: numpy RandomState for reproducibility
    
    Returns:
        spo2: np.array of shape (length,), values in [70, 100]
    """
    baseline = patient_params['baseline']
    noise_std = patient_params['noise_std']
    drift_amp = patient_params['drift_amp']
    drift_period = patient_params['drift_period']
    desat_severity = patient_params['desat_severity']
    
    t = np.linspace(0, 1, length)
    
    # 1. Baseline + slow physiological drift
    drift = drift_amp * np.sin(2 * np.pi * t * (length / drift_period))
    spo2 = np.full(length, baseline, dtype=np.float64) + drift
    
    if label == 1.0:  # Apnea segment
        # 2. Generate desaturation events
        if length == T_5MIN:
            n_events = rng.randint(1, 5)
        elif length == T_3MIN:
            n_events = rng.randint(1, 4)
        else:
            n_events = rng.randint(1, 3)
        
        for _ in range(n_events):
            depth = rng.uniform(4, 15) * desat_severity
            onset_frac = rng.uniform(0.05, 0.75)
            lag_frac = rng.uniform(0.02, 0.08)
            recovery_speed = rng.uniform(0.03, 0.08)
            
            onset_idx = int((onset_frac + lag_frac) * length)
            onset_idx = min(onset_idx, length - 20)
            
            for i in range(onset_idx, length):
                dist = i - onset_idx
                drop_duration = int(rng.uniform(15, 30) * (length / T_5MIN))
                drop_duration = max(drop_duration, 5)
                
                if dist < drop_duration:
                    progress = dist / drop_duration
                    spo2[i] -= depth * (1 / (1 + np.exp(-10 * (progress - 0.5))))
                else:
                    recovery_dist = dist - drop_duration
                    remaining_drop = depth * np.exp(-recovery_speed * recovery_dist)
                    spo2[i] -= remaining_drop
    else:
        # 3. Normal segment — occasional artifact dips (2% chance)
        if rng.random() < 0.02:
            artifact_start = rng.randint(10, length - 30)
            artifact_depth = rng.uniform(3, 8)
            artifact_len = rng.randint(5, 15)
            for i in range(artifact_start, min(artifact_start + artifact_len, length)):
                progress = (i - artifact_start) / artifact_len
                spo2[i] -= artifact_depth * np.sin(np.pi * progress)
    
    # 4. Add sensor noise
    noise = rng.normal(0, noise_std, length)
    spo2 += noise
    
    # 5. Clamp to valid SpO2 range [70, 100]
    spo2 = np.clip(spo2, 70.0, 100.0)
    
    return spo2.astype(np.float32)


def generate_patient_params(rng, label_ratio=0.5):
    """Generate per-patient SpO2 characteristics."""
    if label_ratio > 0.5:
        baseline = rng.uniform(93.0, 97.0)
    else:
        baseline = rng.uniform(95.5, 99.0)
    
    return {
        'baseline': baseline,
        'noise_std': rng.uniform(0.3, 0.6),
        'drift_amp': rng.uniform(0.3, 1.5),
        'drift_period': rng.uniform(30, 90),
        'desat_severity': rng.uniform(0.6, 1.4),
    }


# ======================== MAIN GENERATION ========================

def generate_spo2_for_split(labels, groups, n_segments, rng):
    """
    Generate SpO2 signals for a data split (NO augmentation — original segments only).
    
    Returns:
        spo2_1: np.array (N, T_5MIN, 1)
        spo2_2: np.array (N, T_3MIN, 1)
        spo2_3: np.array (N, T_1MIN, 1)
    """
    spo2_1 = np.zeros((n_segments, T_5MIN, 1), dtype=np.float32)
    spo2_2 = np.zeros((n_segments, T_3MIN, 1), dtype=np.float32)
    spo2_3 = np.zeros((n_segments, T_1MIN, 1), dtype=np.float32)
    
    # Generate per-patient params
    if groups is not None:
        unique_subjects = list(set(groups))
        subject_params = {}
        for subj in unique_subjects:
            subj_labels = labels[[i for i, g in enumerate(groups) if g == subj]]
            apnea_ratio = np.mean(subj_labels)
            subject_params[subj] = generate_patient_params(rng, apnea_ratio)
    else:
        subject_params = None
    
    for seg_idx in range(n_segments):
        label = labels[seg_idx]
        
        if subject_params is not None and groups is not None:
            params = subject_params[groups[seg_idx]]
        else:
            params = generate_patient_params(rng, label)
        
        sig_5min = generate_spo2_signal(label, T_5MIN, params, rng)
        
        spo2_1[seg_idx, :, 0] = sig_5min
        spo2_2[seg_idx, :, 0] = sig_5min[180:720]   # center 3 minutes
        spo2_3[seg_idx, :, 0] = sig_5min[360:540]   # center 1 minute
        
        if (seg_idx + 1) % 5000 == 0:
            print(f"    Generated {seg_idx + 1}/{n_segments} SpO2 signals...")
    
    return spo2_1, spo2_2, spo2_3


def normalize_spo2(arr):
    """Normalize SpO2 to [0, 1] range: (x - 70) / 30"""
    return (arr - 70.0) / 30.0


def main():
    print("=" * 60)
    print("Synthetic SpO2 Dataset Generator (Memory-Efficient)")
    print("=" * 60)
    
    # --- Load existing data to get labels and groups ---
    print("\n[1/3] Loading existing preprocessed data for labels...")
    if not os.path.exists(PICKLE_CACHE):
        print(f"ERROR: {PICKLE_CACHE} not found. Run SE_MSCNN_v2_improved.py first.")
        sys.exit(1)
    
    with open(PICKLE_CACHE, "rb") as f:
        data = pickle.load(f)
    
    n_train = len(data['y_train'])
    n_val = len(data['y_val'])
    n_test = len(data['y_test'])
    
    print(f"  Original segments: Train={n_train}, Val={n_val}, Test={n_test}")
    print(f"  Total: {n_train + n_val + n_test}")
    print(f"  NOTE: Augmentation (9x train) will be done on-the-fly during training")
    
    groups_test = data.get('groups_test', None)
    
    rng = np.random.RandomState(SEED)
    
    # --- Generate SpO2 for each split (NO pre-augmentation) ---
    print(f"\n[2/3] Generating SpO2 signals...")
    
    print(f"\n  TRAIN ({n_train} signals)...")
    print(f"  Train apnea rate: {np.mean(data['y_train']):.2%}")
    train_spo2_1, train_spo2_2, train_spo2_3 = generate_spo2_for_split(
        data['y_train'], groups=None, n_segments=n_train, rng=rng
    )
    print(f"  Train SpO2 shape: {train_spo2_1.shape}")
    
    print(f"\n  VAL ({n_val} signals)...")
    val_spo2_1, val_spo2_2, val_spo2_3 = generate_spo2_for_split(
        data['y_val'], groups=None, n_segments=n_val, rng=rng
    )
    print(f"  Val SpO2 shape: {val_spo2_1.shape}")
    
    print(f"\n  TEST ({n_test} signals)...")
    test_spo2_1, test_spo2_2, test_spo2_3 = generate_spo2_for_split(
        data['y_test'], groups=groups_test, n_segments=n_test, rng=rng
    )
    print(f"  Test SpO2 shape: {test_spo2_1.shape}")
    
    # --- Normalize ---
    print("\nNormalizing SpO2 to [0, 1] range...")
    train_spo2_1 = normalize_spo2(train_spo2_1)
    train_spo2_2 = normalize_spo2(train_spo2_2)
    train_spo2_3 = normalize_spo2(train_spo2_3)
    val_spo2_1 = normalize_spo2(val_spo2_1)
    val_spo2_2 = normalize_spo2(val_spo2_2)
    val_spo2_3 = normalize_spo2(val_spo2_3)
    test_spo2_1 = normalize_spo2(test_spo2_1)
    test_spo2_2 = normalize_spo2(test_spo2_2)
    test_spo2_3 = normalize_spo2(test_spo2_3)
    
    # --- Sanity checks ---
    print("\n" + "=" * 40)
    print("SANITY CHECKS")
    print("=" * 40)
    
    raw_min = train_spo2_1.min() * 30 + 70
    raw_max = train_spo2_1.max() * 30 + 70
    print(f"  SpO2 range (denormalized): [{raw_min:.1f}, {raw_max:.1f}]%")
    
    apnea_mask = data['y_train'] == 1.0
    normal_mask = data['y_train'] == 0.0
    mean_apnea = (train_spo2_1[apnea_mask].mean() * 30 + 70)
    mean_normal = (train_spo2_1[normal_mask].mean() * 30 + 70)
    print(f"  Mean SpO2 (Normal): {mean_normal:.2f}%")
    print(f"  Mean SpO2 (Apnea):  {mean_apnea:.2f}%")
    print(f"  Difference: {mean_normal - mean_apnea:.2f}% (expected: 2-6%)")
    assert mean_normal > mean_apnea, "ERROR: Normal SpO2 should be higher than Apnea!"
    print("  ✓ Apnea segments have lower SpO2 (correct)")
    
    # --- Save as individual .npy files ---
    print(f"\n[3/3] Saving to {SPO2_DIR}/...")
    os.makedirs(SPO2_DIR, exist_ok=True)
    
    # SpO2 arrays
    np.save(os.path.join(SPO2_DIR, "train_spo2_1.npy"), train_spo2_1)
    np.save(os.path.join(SPO2_DIR, "train_spo2_2.npy"), train_spo2_2)
    np.save(os.path.join(SPO2_DIR, "train_spo2_3.npy"), train_spo2_3)
    np.save(os.path.join(SPO2_DIR, "val_spo2_1.npy"), val_spo2_1)
    np.save(os.path.join(SPO2_DIR, "val_spo2_2.npy"), val_spo2_2)
    np.save(os.path.join(SPO2_DIR, "val_spo2_3.npy"), val_spo2_3)
    np.save(os.path.join(SPO2_DIR, "test_spo2_1.npy"), test_spo2_1)
    np.save(os.path.join(SPO2_DIR, "test_spo2_2.npy"), test_spo2_2)
    np.save(os.path.join(SPO2_DIR, "test_spo2_3.npy"), test_spo2_3)
    
    # Labels (save copies so training script can load without pickle)
    np.save(os.path.join(SPO2_DIR, "y_train.npy"), data['y_train'])
    np.save(os.path.join(SPO2_DIR, "y_val.npy"), data['y_val'])
    np.save(os.path.join(SPO2_DIR, "y_test.npy"), data['y_test'])
    
    # Groups
    if groups_test is not None:
        np.save(os.path.join(SPO2_DIR, "groups_test.npy"), np.array(groups_test))
    
    total_size_mb = sum(
        os.path.getsize(os.path.join(SPO2_DIR, f))
        for f in os.listdir(SPO2_DIR) if f.endswith('.npy')
    ) / (1024 * 1024)
    
    print(f"  Saved! Total size: {total_size_mb:.1f} MB")
    print(f"  Files: {len(os.listdir(SPO2_DIR))}")
    
    total_records = n_train + n_val + n_test
    print(f"\n{'=' * 60}")
    print(f"TOTAL RECORDS GENERATED: {total_records:,}")
    print(f"  Train: {n_train:,} (9x augmentation on-the-fly during training)")
    print(f"  Val:   {n_val:,}")
    print(f"  Test:  {n_test:,}")
    print(f"{'=' * 60}")
    
    del data
    gc.collect()


if __name__ == "__main__":
    main()
