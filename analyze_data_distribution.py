import pickle
import numpy as np

# Load the data
with open('code_baseline/ucd_with_real_spo2.pkl', 'rb') as f:
    data = pickle.load(f)

o_train, y_train = data['o_train'], data['y_train']
o_test, y_test = data['o_test'], data['y_test']

# Split train into train and val (70/30)
from sklearn.model_selection import train_test_split
train_indices, val_indices, y_train_split, y_val_split = train_test_split(
    range(len(y_train)), y_train, test_size=0.3, random_state=42, stratify=y_train
)

print("=" * 60)
print("DATA DISTRIBUTION ANALYSIS")
print("=" * 60)

print("\n📊 TRAINING SET:")
train_unique, train_counts = np.unique(y_train_split, return_counts=True)
for label, count in zip(train_unique, train_counts):
    pct = 100 * count / len(y_train_split)
    print(f"  Class {label}: {count:4d} ({pct:5.1f}%)")
print(f"  Total: {len(y_train_split)}")

print("\n📊 VALIDATION SET:")
val_unique, val_counts = np.unique(y_val_split, return_counts=True)
for label, count in zip(val_unique, val_counts):
    pct = 100 * count / len(y_val_split)
    print(f"  Class {label}: {count:4d} ({pct:5.1f}%)")
print(f"  Total: {len(y_val_split)}")

print("\n📊 TEST SET:")
test_unique, test_counts = np.unique(y_test, return_counts=True)
for label, count in zip(test_unique, test_counts):
    pct = 100 * count / len(y_test)
    print(f"  Class {label}: {count:4d} ({pct:5.1f}%)")
print(f"  Total: {len(y_test)}")

# Calculate class weights
class_weight = {}
for label in train_unique:
    count = train_counts[label]
    weight = len(y_train_split) / (len(train_unique) * count)
    class_weight[label] = weight
    print(f"\n  Class {label} weight: {weight:.4f}")

print("\n" + "=" * 60)
print(f"Class weight dict: {class_weight}")
print("=" * 60)
