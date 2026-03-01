import numpy as np
import pandas as pd
import os

def create_sample_ecg(filename, is_apnea=False):
    # Generate 1 minute of fake ECG data at 100Hz (6000 samples)
    t = np.linspace(0, 60, 6000)
    
    # Baseline wander
    baseline = 0.5 * np.sin(2 * np.pi * 0.05 * t)
    
    # Heart rate (normal ~60bpm, apnea might have bradycardia/tachycardia)
    hr = 70 if not is_apnea else 50 + 30 * np.sin(2 * np.pi * 0.02 * t)
    
    # Generate peaks
    peaks = np.zeros(6000)
    phase = 0
    for i in range(6000):
        phase += hr[i] / 60 / 100 if isinstance(hr, np.ndarray) else hr / 60 / 100
        if phase >= 1.0:
            peaks[i] = 1.0
            phase -= 1.0
            
    # Simple QRS
    signal = baseline
    for i in np.where(peaks == 1)[0]:
        if i > 5 and i < 5995:
            signal[i-2:i+3] += np.array([-0.2, 0.5, 1.5, -0.4, -0.1])
            
    # Add noise
    signal += np.random.normal(0, 0.05, 6000)
    
    # Save
    df = pd.DataFrame({'Time_s': t, 'ECG_mV': signal})
    df.to_csv(filename, index=False)
    print(f"Created {filename}")

if __name__ == '__main__':
    create_sample_ecg('sample_patient_1_normal.csv', is_apnea=False)
    create_sample_ecg('sample_patient_2_apnea.csv', is_apnea=True)
    create_sample_ecg('sample_patient_3_borderline.csv', is_apnea=True)
