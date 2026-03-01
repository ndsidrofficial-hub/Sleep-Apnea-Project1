"""
UCD Dataset Preprocessing with Real SPO2 Data
==============================================

Extracts ECG and SpO2 data from the UCD Sleep Apnea Database.
Outputs pickle file compatible with SE-MSCNN_with_SPO2 model.

Dataset: St. Vincent's University Hospital / UCD Sleep Apnea Database
Source:  https://physionet.org/content/ucddb/1.0.0/

For each subject:
  - ECG  : read from <subj>_lifecard.edf  (128 Hz → resampled to 100 Hz)
  - SpO2 : read from <subj>.rec            (128 Hz PSG → downsampled to 8 Hz)
  - Labels: derived from <subj>_respevt.txt (apnea/hypopnea event times)

Usage:
    python preprocess_ucd_real_spo2.py
"""

import os
import sys
import re
import pickle
import shutil
import numpy as np
import mne
from scipy.signal import medfilt, resample
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
import biosppy.signals.tools as st
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ====================== CONFIG ======================
UCD_DIR = r"C:\Users\siddh\Downloads\st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0"
OUT_PKL = os.path.join(os.path.dirname(__file__), "ucd_with_real_spo2.pkl")

ECG_FS = 128  # Lifecard EDF sampling rate
SPO2_FS_IN = 128  # SpO2 stored at PSG rate (128 Hz) in .rec
SPO2_FS_OUT = 8  # Downsample SpO2 to 8 Hz for processing
TARGET_FS = 100  # Resample ECG to match Apnea-ECG baseline
EPOCH_S = 60  # 1-minute epoch
BEFORE = 2  # context minutes before centre
AFTER = 2  # context minutes after centre
WIN_S = (BEFORE + 1 + AFTER) * EPOCH_S  # 300 seconds

# RRI interpolation grid (same as baseline)
IR = 3
TM = np.arange(0, WIN_S, step=1.0 / IR)  # 900 points
HR_MIN, HR_MAX = 20, 300

# Define which subjects to use
# You can expand these lists with more subjects
TRAIN_IDS = ["ucddb002", "ucddb003", "ucddb005", "ucddb007", "ucddb008", "ucddb009"]
TEST_IDS = ["ucddb014", "ucddb015"]

scaler = lambda arr: (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def read_edf_as_edf(rec_path):
    """Read a .rec file by temporarily copying to .edf extension (MNE workaround)."""
    tmp_edf = rec_path.replace(".rec", "_mne_tmp.edf")
    shutil.copy2(rec_path, tmp_edf)
    try:
        raw = mne.io.read_raw_edf(tmp_edf, preload=True, verbose=False)
    finally:
        if os.path.exists(tmp_edf):
            os.remove(tmp_edf)
    return raw


def parse_respevt(txt_path):
    """
    Parse _respevt.txt for apnea/hypopnea events.
    
    Lines format: HH:MM:SS  EVENT_TYPE  Duration ...
    Example: 00:29:13  HYP-C             16       89.9    4.1
    """
    events = []
    try:
        with open(txt_path, "r", errors="replace") as f:
            for line in f:
                # Match HH:MM:SS at line start followed by event type
                m = re.match(r"^\s*(\d{2}):(\d{2}):(\d{2})\s+(APNEA|HYP|CSA|MSA|OSA)", line, re.IGNORECASE)
                if m:
                    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    t_start = hh * 3600 + mm * 60 + ss
                    
                    # Duration is typically the first number after event type
                    parts = line.split()
                    dur = 0
                    for p in parts[2:]:
                        try:
                            dur = int(p)
                            break
                        except ValueError:
                            continue
                    
                    events.append((t_start, t_start + dur))
    except FileNotFoundError:
        pass
    
    return events


def epoch_has_event(epoch_start_s, epoch_end_s, events):
    """Check if any event overlaps with the epoch time window."""
    for ev_s, ev_e in events:
        if ev_s < epoch_end_s and ev_e > epoch_start_s:
            return True
    return False


def process_subject(subj_id):
    """
    Process one UCD subject extracting ECG + real SPO2 data.
    
    Returns:
        (X_list, y_list, groups_list)
        X_list[i] = ((rri_tm, rri_sig), (ampl_tm, ampl_sig), spo2_segment)
    """
    base = os.path.join(UCD_DIR, subj_id)
    rec_f = base + ".rec"
    edf_f = base + "_lifecard.edf"
    evt_f = base + "_respevt.txt"

    # Check all files exist
    for p in (rec_f, edf_f, evt_f):
        if not os.path.exists(p):
            print(f"  [{subj_id}] Missing file: {p}")
            return [], [], []

    # ─────── Load respiratory events ───────
    events = parse_respevt(evt_f)

    # ─────── Load SpO2 from PSG .rec file ───────
    try:
        psg_raw = read_edf_as_edf(rec_f)
        # Find SpO2 channel (usually "SpO2" or "SaO2")
        spo2_idx = [i for i, ch in enumerate(psg_raw.ch_names) 
                    if "spo2" in ch.lower() or "sao2" in ch.lower()]
        if not spo2_idx:
            print(f"  [{subj_id}] No SpO2 channel found in PSG")
            return [], [], []
        
        spo2_raw_128 = psg_raw.get_data(picks=[spo2_idx[0]])[0]  # at 128 Hz
        psg_fs = int(psg_raw.info["sfreq"])
    except Exception as e:
        print(f"  [{subj_id}] SpO2 load error: {e}")
        return [], [], []

    # ─────── Load ECG from _lifecard.edf ───────
    try:
        ecg_raw = mne.io.read_raw_edf(edf_f, preload=True, verbose=False)
        ecg_signal = ecg_raw.get_data(picks=[0])[0]  # first channel at 128 Hz
    except Exception as e:
        print(f"  [{subj_id}] ECG load error: {e}")
        return [], [], []

    ecg_duration_s = len(ecg_signal) / ECG_FS

    # ─────── Resample ECG 128→100 Hz ───────
    n_target = int(len(ecg_signal) * TARGET_FS / ECG_FS)
    ecg_signal = resample(ecg_signal, n_target)
    fs = TARGET_FS
    sample = fs * EPOCH_S  # 6000 samples per minute at 100 Hz

    # ─────── Resample SpO2 128 Hz → 8 Hz ───────
    n_spo2_out = int(len(spo2_raw_128) * SPO2_FS_OUT / psg_fs)
    spo2_8hz = resample(spo2_raw_128, n_spo2_out)
    spo2_samples_per_epoch = SPO2_FS_OUT * EPOCH_S  # 480 samples per minute
    spo2_win_samples = SPO2_FS_OUT * WIN_S  # 2400 samples per 5-min window

    psg_duration_s = len(spo2_raw_128) / psg_fs
    safe_duration_s = min(ecg_duration_s, psg_duration_s)
    n_epochs = int(safe_duration_s // EPOCH_S)

    X_list, y_list, grp_list = [], [], []

    # ─────── Extract 5-minute windows ───────
    for j in tqdm(range(n_epochs), desc=f"{subj_id}", leave=False):
        # Skip epochs near boundaries (need before + after context)
        if j < BEFORE or (j + 1 + AFTER) > n_epochs:
            continue

        # ─── ECG window [j-before : j+1+after] ──────────────────────
        ecg_start = int((j - BEFORE) * sample)
        ecg_end = int((j + 1 + AFTER) * sample)
        if ecg_end > len(ecg_signal):
            continue
        seg = ecg_signal[ecg_start:ecg_end]

        # Bandpass filter (3-45 Hz)
        try:
            seg_f, _, _ = st.filter_signal(seg, ftype='FIR', band='bandpass',
                                           order=int(0.3 * fs),
                                           frequency=[3, 45], sampling_rate=fs)
        except Exception:
            continue

        # R-peak detection
        try:
            rpeaks, = hamilton_segmenter(seg_f, sampling_rate=fs)
            rpeaks, = correct_rpeaks(seg_f, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)
        except Exception:
            continue

        if len(rpeaks) < 2:
            continue

        # Quality check: reasonable number of beats
        n_beats_per_min = len(rpeaks) / (1 + BEFORE + AFTER)
        if n_beats_per_min < 40 or n_beats_per_min > 200:
            continue

        # Extract RRI and amplitude
        rri_tm = rpeaks[1:] / float(fs)
        rri_signal = np.diff(rpeaks) / float(fs)
        rri_signal = medfilt(rri_signal, kernel_size=3)
        ampl_tm = rpeaks / float(fs)
        ampl_sig = seg_f[rpeaks]

        # Check heart rate is physiologically reasonable
        hr = 60.0 / rri_signal
        if not np.all((hr >= HR_MIN) & (hr <= HR_MAX)):
            continue

        # ─── SpO2 window ──────────────────────────────────────────────
        spo2_start = int((j - BEFORE) * spo2_samples_per_epoch)
        spo2_end = spo2_start + spo2_win_samples
        if spo2_end > len(spo2_8hz):
            continue
        spo2_seg = spo2_8hz[spo2_start:spo2_end].astype(np.float32)

        # Remove physiologically impossible SpO2 values
        spo2_seg = np.clip(spo2_seg, 50, 100)

        # ─── Labeling ─────────────────────────────────────────────────
        epoch_start_s = j * EPOCH_S
        epoch_end_s = (j + 1) * EPOCH_S
        label = 1.0 if epoch_has_event(epoch_start_s, epoch_end_s, events) else 0.0

        X_list.append(((rri_tm, rri_signal), (ampl_tm, ampl_sig), spo2_seg))
        y_list.append(label)
        grp_list.append(subj_id)

    return X_list, y_list, grp_list


def main():
    """Main preprocessing pipeline for UCD dataset with real SPO2."""
    print("=" * 70)
    print("UCD Sleep Apnea Database Preprocessing")
    print("Dataset: Real ECG + Real SPO2")
    print("=" * 70)

    # Check dataset exists
    if not os.path.exists(UCD_DIR):
        print(f"\nERROR: UCD database not found at:")
        print(f"  {UCD_DIR}")
        print("\nPlease download from:")
        print("  https://physionet.org/content/ucddb/1.0.0/")
        return None

    o_train, y_train, groups_train = [], [], []
    o_test, y_test, groups_test = [], [], []

    # Process training subjects
    print(f"\n📖 Processing {len(TRAIN_IDS)} TRAINING subjects:")
    for sid in TRAIN_IDS:
        X, y, g = process_subject(sid)
        o_train.extend(X)
        y_train.extend(y)
        groups_train.extend(g)
        
        if y:
            n_apnea = int(sum(y))
            n_normal = len(y) - n_apnea
            print(f"  ✅ {sid}: {len(X)} epochs (apnea={n_apnea}, normal={n_normal})")
        else:
            print(f"  ⚠️  {sid}: No data extracted")

    # Process test subjects
    print(f"\n📖 Processing {len(TEST_IDS)} TEST subjects:")
    for sid in TEST_IDS:
        X, y, g = process_subject(sid)
        o_test.extend(X)
        y_test.extend(y)
        groups_test.extend(g)
        
        if y:
            n_apnea = int(sum(y))
            n_normal = len(y) - n_apnea
            print(f"  ✅ {sid}: {len(X)} epochs (apnea={n_apnea}, normal={n_normal})")
        else:
            print(f"  ⚠️  {sid}: No data extracted")

    if not o_train or not o_test:
        print("\nERROR: No data was extracted. Please check:")
        print("  - UCD database path is correct")
        print("  - Subject IDs exist in the dataset")
        print("  - Required files: _lifecard.edf, .rec, _respevt.txt")
        return None

    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 DATASET SUMMARY")
    print("=" * 70)
    n_train_apnea = int(sum(y_train))
    n_train_normal = len(y_train) - n_train_apnea
    n_test_apnea = int(sum(y_test))
    n_test_normal = len(y_test) - n_test_apnea
    
    print(f"\n📈 Training Set: {len(o_train)} epochs")
    print(f"   Apnea:  {n_train_apnea} ({100*n_train_apnea/len(y_train):.1f}%)")
    print(f"   Normal: {n_train_normal} ({100*n_train_normal/len(y_train):.1f}%)")
    
    print(f"\n📈 Test Set: {len(o_test)} epochs")
    print(f"   Apnea:  {n_test_apnea} ({100*n_test_apnea/len(y_test):.1f}%)")
    print(f"   Normal: {n_test_normal} ({100*n_test_normal/len(y_test):.1f}%)")

    # Save dataset
    ucd_data = {
        'o_train': o_train,
        'y_train': y_train,
        'groups_train': groups_train,
        'o_test': o_test,
        'y_test': y_test,
        'groups_test': groups_test,
    }

    print(f"\n💾 Saving to: {OUT_PKL}")
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(ucd_data, f, protocol=2)
    
    size_mb = os.path.getsize(OUT_PKL) / (1024**2)
    print(f"✅ File size: {size_mb:.1f} MB")
    
    print(f"\n✅ UCD dataset with REAL SPO2 ready for training!")
    return OUT_PKL


if __name__ == "__main__":
    pkl = main()
    if pkl:
        print(f"\n📝 Next step: Update SE-MSCNN_with_SPO2.py to use:")
        print(f"   pickle_path = '{pkl}'")
