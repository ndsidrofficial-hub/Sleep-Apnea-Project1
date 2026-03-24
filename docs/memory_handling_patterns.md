# Memory Handling & Dataset Scaling Patterns

## The Core Problem: Monolithic Pickle MemoryError

When scaling synthetic datasets (e.g., 300K+ ECG/SpO2 records), loading a monolithic file via `pickle.load()` reliably causes `MemoryError` even on high-end hardware with 32GB RAM. The cause is that Pickle instantiates the **entire object tree simultaneously** during deserialization — Python object pointer metadata can dwarf raw binary data size.

```python
# spo2_data.pkl on disk: 2.6 GB
# RAM consumed during load: >12 GB → CRASH
with open(SPO2_CACHE, "rb") as f:
    data = pickle.load(f)  # MemoryError
```

---

## Solution 1: NPY Sharding (Current Architecture)

Instead of a single monolithic pickle, split the dataset by **modality and split** into individual `.npy` files saved in `spo2_npy/`.

```
spo2_npy/
  train_spo2_1.npy   # (N_train, T, 1) — 5-min SpO2
  train_spo2_2.npy   # (N_train, T, 1) — 3-min SpO2
  train_spo2_3.npy   # (N_train, T, 1) — 1-min SpO2
  val_spo2_1.npy
  val_spo2_2.npy
  val_spo2_3.npy
  test_spo2_1.npy
  test_spo2_2.npy
  test_spo2_3.npy
```

Loading only what you need:
```python
'spo2_train1': np.load(os.path.join(SPO2_DIR, "train_spo2_1.npy")),
```

**Result:** Peak memory drops from ~12 GB → ~600 MB.

---

## Solution 2: Dynamic Lazy Augmentation in Dataset (`LazyApneaSpO2Dataset`)

Pre-augmenting all data (e.g., 9x → store 9 × 13K = 117K samples) bloats disk and RAM usage. The lazy approach stores **only original signals** and applies random augmentations inside `__getitem__` at training time.

```python
class LazyApneaSpO2Dataset(Dataset):
    def __init__(self, x1, x2, x3, spo2_1, spo2_2, spo2_3, y,
                 augment=False, aug_factor=1):
        self.n_real = len(y)
        self.aug_factor = aug_factor
        # Store only originals — no pre-duplication

    def __len__(self):
        # Virtually expand: returns N_real * aug_factor
        return self.n_real * self.aug_factor

    def __getitem__(self, idx):
        real_idx = idx % self.n_real  # map virtual → real

        x1 = torch.from_numpy(self.x1[real_idx].T.copy()).float()

        # Only augment virtual copies (copy 0 = original passthrough)
        if self.augment and (idx // self.n_real) > 0:
            noise_level = random.uniform(0.01, 0.05)
            x1 = x1 + torch.randn_like(x1) * noise_level
            scale = random.uniform(0.85, 1.15)
            x1 = x1 * scale
            shift = random.randint(-15, 15)
            x1 = torch.roll(x1, shift, dims=-1)

        return x1, ..., torch.tensor(int(self.y[real_idx]), dtype=torch.long)
```

**Memory footprint comparison:**

| Approach | Stored Samples | Disk Size | Peak RAM |
|---|---|---|---|
| Pre-augmented (9x) | 117,000 | ~3.6 GB | ~12 GB |
| **Lazy (dynamic)** | **13,000** | **~420 MB** | **<1 GB** |

---

## Solution 3: Chunked Generation with Explicit GC

When generating large synthetic datasets, avoid accumulating all data in a single list. Process and flush in fixed-size chunks, calling `gc.collect()` between each to free intermediate Python objects.

```python
CHUNK_SIZE = 5000
all_chunks = []

for start in range(0, total_n, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, total_n)
    chunk = generate_spo2_for_indices(range(start, end))
    all_chunks.append(chunk)
    gc.collect()  # release intermediate objects immediately

final_array = np.concatenate(all_chunks)
```

---

## Solution 4: Monolithic → Sharded Migration (Recovery Protocol)

If a monolithic pickle already exists and generation is expensive to redo, extract and re-shard it **in-place** without regenerating from scratch.

```python
# extract_data.py pattern
import pickle, numpy as np, gc, os

print("Loading monolithic cache (may be slow)...")
with open("spo2_data.pkl", "rb") as f:
    data = pickle.load(f)

# If data was pre-augmented 9x, de-augment by strided slicing
original_train = data['x_train'][::9]   # keep only 1-in-9

# Save sharded .npy files
os.makedirs("spo2_npy", exist_ok=True)
np.save("spo2_npy/train_spo2_1.npy", original_train)

# Immediately delete large arrays to cap peak memory
del data
gc.collect()
```

---

## Solution 5: HDF5 for Full On-Demand Lazy Loading

For truly massive datasets (10M+ segments), replace `.npy` files with HDF5, which allows reading **individual rows** directly from disk without loading the full array.

```python
import h5py
from torch.utils.data import Dataset

class H5ApneaDataset(Dataset):
    def __init__(self, h5_path, split="train"):
        self.f = h5py.File(h5_path, 'r')
        self.x1 = self.f[f"{split}/ecg_1"]
        self.y  = self.f[f"{split}/labels"]

    def __getitem__(self, idx):
        # Only reads a single row from disk — zero extra RAM
        return torch.tensor(self.x1[idx]), int(self.y[idx])

    def __len__(self):
        return len(self.y)
```

---

## Quick Reference: When to Use What

| Scenario | Pattern |
|---|---|
| Dataset fits in RAM (<4 GB) | Standard `pickle.load` + in-memory `Dataset` |
| Dataset 4–20 GB, GPU training | **NPY sharding + `LazyApneaSpO2Dataset`** ✅ current |
| Dataset >20 GB | HDF5 with `h5py` lazy-loading |
| One-time recovery from monolithic pkl | `extract_data.py` chunked migration |
| Augmentation needed (any size) | Always use dynamic `__getitem__` augmentation |
