import wfdb

# Change to ONE subject from your folder (e.g., ucddb003)
SUBJECT_ID = "ucddb003"

# Main PSG file (usually the one with 'lifecard' or no suffix)
RECORD_PATH = r"C:\Users\siddh\Downloads\ucd-sleep-apnea-1.0.0\{}_lifecard".format(SUBJECT_ID)  # adjust suffix if needed
# Or try: r"C:\Users\siddh\Downloads\ucd-sleep-apnea-1.0.0\{}_PSG".format(SUBJECT_ID)

print("Loading record...")
record = wfdb.rdrecord(RECORD_PATH)

print("\nChannels in this record:")
print(record.sig_name)

print("\nSampling rates:", record.fs)
print("Length (samples):", record.sig_len)
print("\nLook for SpO2-related channels (e.g., 'SpO2', 'SaO2', 'Oxygen', 'Pulse Ox', 'Oximetry')")