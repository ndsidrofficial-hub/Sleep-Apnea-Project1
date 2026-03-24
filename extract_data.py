"""
Extract data from spo2_data.pkl into:
  1. preprocessed_data.pkl (original ECG data, de-augmented)
  2. spo2_npy/ directory (individual .npy files)

Uses gc.collect() between operations to minimize peak memory.
"""
import pickle
import numpy as np
import os
import gc
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPO2_PKL = os.path.join(BASE_DIR, "spo2_data.pkl")
PICKLE_CACHE = os.path.join(BASE_DIR, "preprocessed_data.pkl")
SPO2_DIR = os.path.join(BASE_DIR, "spo2_npy")

if not os.path.exists(SPO2_PKL):
    print(f"ERROR: {SPO2_PKL} not found")
    sys.exit(1)

print("=" * 60)
print("Extracting data from spo2_data.pkl")
print("=" * 60)
print(f"File size: {os.path.getsize(SPO2_PKL) / (1024*1024):.1f} MB")
print("Loading (this may take a moment due to file size)...")

# Force garbage collection first
gc.collect()

try:
    with open(SPO2_PKL, "rb") as f:
        data = pickle.load(f)
    print("Loaded successfully!")
except MemoryError:
    print("MemoryError! Trying with mmap-based approach...")
    # On 32GB RAM with 64-bit Python, this shouldn't happen
    # But as fallback, try loading with reduced overhead
    import mmap
    with open(SPO2_PKL, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = pickle.loads(mm)
        mm.close()
    print("Loaded via mmap!")

print(f"\nKeys: {list(data.keys())}")

# --- Extract ECG (de-augmented: take every 9th sample for train) ---
print("\n[1/2] Creating preprocessed_data.pkl...")

# The train ECG was augmented 9x via np.repeat - take every 9th to get originals
AUG_FACTOR = 9
ecg_train1_aug = data['ecg_train1']  # (121446, 900, 2)
n_original = len(ecg_train1_aug) // AUG_FACTOR

ecg_data = {
    'x_train1': ecg_train1_aug[::AUG_FACTOR][:n_original],  # De-augment
    'x_train2': data['ecg_train2'][::AUG_FACTOR][:n_original],
    'x_train3': data['ecg_train3'][::AUG_FACTOR][:n_original],
    'y_train': data['y_train'][::AUG_FACTOR][:n_original],
    'x_val1': data['ecg_val1'],
    'x_val2': data['ecg_val2'],
    'x_val3': data['ecg_val3'],
    'y_val': data['y_val'],
    'x_test1': data['ecg_test1'],
    'x_test2': data['ecg_test2'],
    'x_test3': data['ecg_test3'],
    'y_test': data['y_test'],
    'groups_test': data.get('groups_test', None),
}

print(f"  Train: {ecg_data['x_train1'].shape} (de-augmented from {ecg_train1_aug.shape})")
print(f"  Val:   {ecg_data['x_val1'].shape}")
print(f"  Test:  {ecg_data['x_test1'].shape}")
print(f"  Train labels: {ecg_data['y_train'].shape}")

with open(PICKLE_CACHE, "wb") as f:
    pickle.dump(ecg_data, f, protocol=4)

pkl_size = os.path.getsize(PICKLE_CACHE) / (1024*1024)
print(f"  Saved: {PICKLE_CACHE} ({pkl_size:.1f} MB)")
del ecg_data
gc.collect()

# --- Extract SpO2 as .npy files ---
print("\n[2/2] Creating spo2_npy/ files...")
os.makedirs(SPO2_DIR, exist_ok=True)

# SpO2 train was also augmented 9x - take every 9th for originals
spo2_train1 = data['spo2_train1'][::AUG_FACTOR][:n_original]
spo2_train2 = data['spo2_train2'][::AUG_FACTOR][:n_original]
spo2_train3 = data['spo2_train3'][::AUG_FACTOR][:n_original]

np.save(os.path.join(SPO2_DIR, "train_spo2_1.npy"), spo2_train1)
np.save(os.path.join(SPO2_DIR, "train_spo2_2.npy"), spo2_train2)
np.save(os.path.join(SPO2_DIR, "train_spo2_3.npy"), spo2_train3)
print(f"  Train: {spo2_train1.shape}")

np.save(os.path.join(SPO2_DIR, "val_spo2_1.npy"), data['spo2_val1'])
np.save(os.path.join(SPO2_DIR, "val_spo2_2.npy"), data['spo2_val2'])
np.save(os.path.join(SPO2_DIR, "val_spo2_3.npy"), data['spo2_val3'])
print(f"  Val:   {data['spo2_val1'].shape}")

np.save(os.path.join(SPO2_DIR, "test_spo2_1.npy"), data['spo2_test1'])
np.save(os.path.join(SPO2_DIR, "test_spo2_2.npy"), data['spo2_test2'])
np.save(os.path.join(SPO2_DIR, "test_spo2_3.npy"), data['spo2_test3'])
print(f"  Test:  {data['spo2_test1'].shape}")

# Labels
np.save(os.path.join(SPO2_DIR, "y_train.npy"), data['y_train'][::AUG_FACTOR][:n_original])
np.save(os.path.join(SPO2_DIR, "y_val.npy"), data['y_val'])
np.save(os.path.join(SPO2_DIR, "y_test.npy"), data['y_test'])

if data.get('groups_test', None) is not None:
    np.save(os.path.join(SPO2_DIR, "groups_test.npy"), np.array(data['groups_test']))

npy_size = sum(
    os.path.getsize(os.path.join(SPO2_DIR, f))
    for f in os.listdir(SPO2_DIR) if f.endswith('.npy')
) / (1024*1024)

print(f"\n  spo2_npy/ total: {npy_size:.1f} MB ({len(os.listdir(SPO2_DIR))} files)")

del data
gc.collect()

print(f"\n{'=' * 60}")
print("DONE! Both preprocessed_data.pkl and spo2_npy/ created.")
print(f"  preprocessed_data.pkl: {pkl_size:.1f} MB")
print(f"  spo2_npy/: {npy_size:.1f} MB")
print(f"{'=' * 60}")
