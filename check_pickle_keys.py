import pickle

PICKLE_PATH = r"C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0\apnea-ecg.pkl"

print("Loading pickle to check keys...")
with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f)

print("\nAll keys in the pickle file:")
print(list(data.keys()))

print("\nData types/shapes for quick check:")
for key in data:
    value = data[key]
    print(f"{key}: type={type(value)}, shape/len={getattr(value, 'shape', len(value)) if hasattr(value, '__len__') else 'N/A'}")