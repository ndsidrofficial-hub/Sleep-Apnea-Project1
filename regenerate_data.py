"""Quick script to regenerate preprocessed_data.pkl from raw ECG data."""
import sys
sys.path.insert(0, '.')
from SE_MSCNN_v2_improved import load_data, PICKLE_CACHE
import pickle
import os

print("Regenerating preprocessed_data.pkl from raw ECG data...")
print(f"Output: {PICKLE_CACHE}")

if os.path.exists(PICKLE_CACHE):
    print("Already exists! Skipping.")
else:
    data = load_data()
    print(f"\nSaving to {PICKLE_CACHE}...")
    with open(PICKLE_CACHE, "wb") as f:
        pickle.dump(data, f, protocol=4)
    file_size_mb = os.path.getsize(PICKLE_CACHE) / (1024 * 1024)
    print(f"Done! File size: {file_size_mb:.1f} MB")
