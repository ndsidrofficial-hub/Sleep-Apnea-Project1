import mne

# Change this line to point to ONE real .edf file you downloaded
EDF_FILE_PATH = r"C:\Users\siddh\Downloads\sleep-edf-database-expanded-1.0.0\sleep-telemetry\ST7161J0-PSG.edf"
print(f"Loading EDF file: {EDF_FILE_PATH}")
raw = mne.io.read_raw_edf(EDF_FILE_PATH, preload=False, verbose=False)

print("\nAll channels found in this file:")
print(raw.ch_names)

print("\nLook for any channel with 'SpO2', 'SaO2', 'Oxygen', 'Oximetry', 'Pulse Ox' or similar.")