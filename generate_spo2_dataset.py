"""
Synthetic SpO2 Dataset Generator for SE-MSCNN v3
=================================================
Generates ~300,000 physiologically realistic SpO2 records
matched to existing ECG segments from the Apnea-ECG database.

SpO2 physiology modeled:
  - Normal: baseline 95-99%, slow drift, sensor noise
  - Apnea:  desaturation 4-15% below baseline, 15-30s lag, exponential recovery
  - Per-patient characteristics (some desaturate more)
  - Artifact/false dips (~2% probability in normal segments)

Chunk-based generation for 4GB RAM safety.
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

PICKLE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed_data.pkl")
SPO2_OUTPUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spo2_data.pkl")

# Temporal lengths matching existing branches
T_5MIN = 900   # 5-minute window
T_3MIN = 540   # 3-minute window
T_1MIN = 180   # 1-minute window

# Number of augmentation copies to generate (9x original → ~300K total)
N_AUGMENTS = 9


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
    desat_severity = patient_params['desat_severity']  # multiplier for desaturation depth
    
    t = np.linspace(0, 1, length)
    
    # 1. Baseline + slow physiological drift
    drift = drift_amp * np.sin(2 * np.pi * t * (length / drift_period))
    spo2 = np.full(length, baseline, dtype=np.float64) + drift
    
    if label == 1.0:  # Apnea segment
        # 2. Generate desaturation events
        # Number of desat events depends on window length
        if length == T_5MIN:
            n_events = rng.randint(1, 5)  # 1-4 events in 5-min window
        elif length == T_3MIN:
            n_events = rng.randint(1, 4)
        else:
            n_events = rng.randint(1, 3)
        
        for _ in range(n_events):
            # Desaturation parameters
            depth = rng.uniform(4, 15) * desat_severity   # % drop below baseline
            onset_frac = rng.uniform(0.05, 0.75)          # where in the window the drop starts
            lag_frac = rng.uniform(0.02, 0.08)             # lag (15-30s equivalent)
            recovery_speed = rng.uniform(0.03, 0.08)       # exponential recovery rate
            
            onset_idx = int((onset_frac + lag_frac) * length)
            onset_idx = min(onset_idx, length - 20)
            
            # Create desaturation envelope
            for i in range(onset_idx, length):
                dist = i - onset_idx
                # Fast drop phase (first ~15-25 steps)
                drop_duration = int(rng.uniform(15, 30) * (length / T_5MIN))
                drop_duration = max(drop_duration, 5)
                
                if dist < drop_duration:
                    # Sigmoid-like drop
                    progress = dist / drop_duration
                    spo2[i] -= depth * (1 / (1 + np.exp(-10 * (progress - 0.5))))
                else:
                    # Exponential recovery
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
    """
    Generate per-patient SpO2 characteristics.
    Patients with more apnea tend to have lower baseline SpO2.
    """
    # Patients with high apnea ratio have slightly lower baselines
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

def generate_spo2_for_split(labels, groups, n_segments, rng, augment_factor=1):
    """
    Generate SpO2 signals for a data split (train/val/test).
    
    Args:
        labels: np.array of 0/1 labels
        groups: list of subject IDs (can be None for val/test)
        n_segments: number of original segments
        rng: numpy RandomState
        augment_factor: how many copies to generate per segment
    
    Returns:
        spo2_1: np.array (N*aug, T_5MIN, 1)
        spo2_2: np.array (N*aug, T_3MIN, 1)
        spo2_3: np.array (N*aug, T_1MIN, 1)
    """
    total = n_segments * augment_factor
    
    # Pre-allocate output arrays
    spo2_1 = np.zeros((total, T_5MIN, 1), dtype=np.float32)
    spo2_2 = np.zeros((total, T_3MIN, 1), dtype=np.float32)
    spo2_3 = np.zeros((total, T_1MIN, 1), dtype=np.float32)
    
    # Generate per-patient params
    if groups is not None:
        unique_subjects = list(set(groups))
        subject_params = {}
        for subj in unique_subjects:
            subj_labels = labels[[i for i, g in enumerate(groups) if g == subj]]
            apnea_ratio = np.mean(subj_labels)
            subject_params[subj] = generate_patient_params(rng, apnea_ratio)
    else:
        # Single set of params with variation per segment
        subject_params = None
    
    out_idx = 0
    for seg_idx in range(n_segments):
        label = labels[seg_idx]
        
        if subject_params is not None and groups is not None:
            base_params = subject_params[groups[seg_idx]]
        else:
            base_params = generate_patient_params(rng, label)
        
        for aug in range(augment_factor):
            # Slightly vary params for each augmentation
            params = dict(base_params)
            if aug > 0:
                params['baseline'] += rng.uniform(-1.0, 1.0)
                params['noise_std'] = max(0.2, params['noise_std'] + rng.uniform(-0.1, 0.1))
                params['desat_severity'] *= rng.uniform(0.8, 1.2)
            
            # Generate all 3 temporal resolutions
            sig_5min = generate_spo2_signal(label, T_5MIN, params, rng)
            
            # 3-min and 1-min are center-crops of the 5-min signal (same as ECG)
            spo2_1[out_idx, :, 0] = sig_5min
            spo2_2[out_idx, :, 0] = sig_5min[180:720]   # center 3 minutes
            spo2_3[out_idx, :, 0] = sig_5min[360:540]   # center 1 minute
            
            out_idx += 1
        
        if (seg_idx + 1) % 5000 == 0:
            print(f"    Generated {out_idx}/{total} SpO2 signals...")
    
    return spo2_1[:out_idx], spo2_2[:out_idx], spo2_3[:out_idx]


def normalize_spo2(arr):
    """Normalize SpO2 to [0, 1] range: (x - 70) / 30"""
    return (arr - 70.0) / 30.0


def main():
    print("=" * 60)
    print("Synthetic SpO2 Dataset Generator")
    print("=" * 60)
    
    # --- Load existing data to get labels and groups ---
    print("\n[1/4] Loading existing preprocessed data for labels...")
    if not os.path.exists(PICKLE_CACHE):
        print(f"ERROR: {PICKLE_CACHE} not found. Run SE_MSCNN_v2_improved.py first.")
        sys.exit(1)
    
    with open(PICKLE_CACHE, "rb") as f:
        data = pickle.load(f)
    
    n_train = len(data['y_train'])
    n_val = len(data['y_val'])
    n_test = len(data['y_test'])
    
    print(f"  Original segments: Train={n_train}, Val={n_val}, Test={n_test}")
    print(f"  Total original: {n_train + n_val + n_test}")
    print(f"  Augmentation factor: {N_AUGMENTS}x (train only)")
    print(f"  Expected total: Train={n_train * N_AUGMENTS}, Val={n_val}, Test={n_test}")
    print(f"  Grand total: {n_train * N_AUGMENTS + n_val + n_test}")
    
    groups_test = data.get('groups_test', None)
    
    rng = np.random.RandomState(SEED)
    
    # --- Generate SpO2 for each split ---
    # Train: augmented 9x
    print(f"\n[2/4] Generating TRAIN SpO2 ({n_train} × {N_AUGMENTS} = {n_train * N_AUGMENTS} signals)...")
    print(f"  Train apnea rate: {np.mean(data['y_train']):.2%}")
    
    # Generate train in chunks to save RAM
    chunk_size = 5000
    train_spo2_1_chunks, train_spo2_2_chunks, train_spo2_3_chunks = [], [], []
    train_labels_aug = []
    
    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        chunk_labels = data['y_train'][start:end]
        
        s1, s2, s3 = generate_spo2_for_split(
            chunk_labels, groups=None,
            n_segments=len(chunk_labels), rng=rng,
            augment_factor=N_AUGMENTS
        )
        
        train_spo2_1_chunks.append(s1)
        train_spo2_2_chunks.append(s2)
        train_spo2_3_chunks.append(s3)
        
        # Repeat labels for augmentation
        for lbl in chunk_labels:
            train_labels_aug.extend([lbl] * N_AUGMENTS)
        
        gc.collect()
    
    train_spo2_1 = np.concatenate(train_spo2_1_chunks, axis=0)
    train_spo2_2 = np.concatenate(train_spo2_2_chunks, axis=0)
    train_spo2_3 = np.concatenate(train_spo2_3_chunks, axis=0)
    y_train_aug = np.array(train_labels_aug, dtype=np.float32)
    del train_spo2_1_chunks, train_spo2_2_chunks, train_spo2_3_chunks, train_labels_aug
    gc.collect()
    
    print(f"  Train SpO2 shape: {train_spo2_1.shape}")
    
    # Val: no augmentation
    print(f"\n[3/4] Generating VAL SpO2 ({n_val} signals)...")
    val_spo2_1, val_spo2_2, val_spo2_3 = generate_spo2_for_split(
        data['y_val'], groups=None,
        n_segments=n_val, rng=rng,
        augment_factor=1
    )
    print(f"  Val SpO2 shape: {val_spo2_1.shape}")
    
    # Test: no augmentation
    print(f"\n[3/4] Generating TEST SpO2 ({n_test} signals)...")
    test_spo2_1, test_spo2_2, test_spo2_3 = generate_spo2_for_split(
        data['y_test'], groups=groups_test,
        n_segments=n_test, rng=rng,
        augment_factor=1
    )
    print(f"  Test SpO2 shape: {test_spo2_1.shape}")
    
    # --- Normalize all SpO2 to [0,1] ---
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
    
    # Check value ranges (before normalization values should have been [70, 100])
    raw_min = train_spo2_1.min() * 30 + 70
    raw_max = train_spo2_1.max() * 30 + 70
    print(f"  SpO2 range (denormalized): [{raw_min:.1f}, {raw_max:.1f}]%")
    
    # Check apnea vs normal mean SpO2
    apnea_mask = y_train_aug == 1.0
    normal_mask = y_train_aug == 0.0
    mean_apnea = (train_spo2_1[apnea_mask].mean() * 30 + 70)
    mean_normal = (train_spo2_1[normal_mask].mean() * 30 + 70)
    print(f"  Mean SpO2 (Normal): {mean_normal:.2f}%")
    print(f"  Mean SpO2 (Apnea):  {mean_apnea:.2f}%")
    print(f"  Difference: {mean_normal - mean_apnea:.2f}% (expected: 2-6%)")
    
    assert mean_normal > mean_apnea, "ERROR: Normal SpO2 should be higher than Apnea!"
    print("  ✓ Apnea segments have lower SpO2 (correct)")
    
    # Check shapes
    print(f"\n  Train: spo2_1={train_spo2_1.shape}, labels={y_train_aug.shape}")
    print(f"  Val:   spo2_1={val_spo2_1.shape}")
    print(f"  Test:  spo2_1={test_spo2_1.shape}")
    
    # --- Also need to augment ECG data to match ---
    print("\n[4/4] Augmenting ECG data to match SpO2 train size...")
    
    # Repeat ECG train data N_AUGMENTS times
    ecg_train1 = np.repeat(data['x_train1'], N_AUGMENTS, axis=0)
    ecg_train2 = np.repeat(data['x_train2'], N_AUGMENTS, axis=0)
    ecg_train3 = np.repeat(data['x_train3'], N_AUGMENTS, axis=0)
    
    print(f"  Augmented ECG train shape: {ecg_train1.shape}")
    
    # --- Save ---
    print(f"\nSaving to {SPO2_OUTPUT}...")
    
    spo2_data = {
        # SpO2 signals
        'spo2_train1': train_spo2_1, 'spo2_train2': train_spo2_2, 'spo2_train3': train_spo2_3,
        'spo2_val1': val_spo2_1, 'spo2_val2': val_spo2_2, 'spo2_val3': val_spo2_3,
        'spo2_test1': test_spo2_1, 'spo2_test2': test_spo2_2, 'spo2_test3': test_spo2_3,
        # Augmented ECG data (train only - val/test stay original)
        'ecg_train1': ecg_train1, 'ecg_train2': ecg_train2, 'ecg_train3': ecg_train3,
        'ecg_val1': data['x_val1'], 'ecg_val2': data['x_val2'], 'ecg_val3': data['x_val3'],
        'ecg_test1': data['x_test1'], 'ecg_test2': data['x_test2'], 'ecg_test3': data['x_test3'],
        # Labels
        'y_train': y_train_aug,
        'y_val': data['y_val'],
        'y_test': data['y_test'],
        # Groups
        'groups_test': data.get('groups_test', None),
    }
    
    with open(SPO2_OUTPUT, 'wb') as f:
        pickle.dump(spo2_data, f, protocol=4)
    
    file_size_mb = os.path.getsize(SPO2_OUTPUT) / (1024 * 1024)
    print(f"  Saved! File size: {file_size_mb:.1f} MB")
    
    total_records = len(y_train_aug) + len(data['y_val']) + len(data['y_test'])
    print(f"\n{'=' * 60}")
    print(f"TOTAL RECORDS GENERATED: {total_records:,}")
    print(f"  Train: {len(y_train_aug):,} (augmented {N_AUGMENTS}x)")
    print(f"  Val:   {len(data['y_val']):,}")
    print(f"  Test:  {len(data['y_test']):,}")
    print(f"{'=' * 60}")
    
    # Cleanup
    del spo2_data, data
    gc.collect()


if __name__ == "__main__":
    main()
