"""
UCD Sleep Apnea Database Preprocessing (DEMO MODE)
===================================================
Dataset: St. Vincent's University Hospital / UCD Sleep Apnea Database (PhysioNet)
Source:  https://physionet.org/content/ucddb/1.0.0/

For each subject:
  - ECG  : read from <subj>_lifecard.edf  (128 Hz, 3-channel Holter → use ch 0)
  - SpO2 : read from <subj>.rec            (128 Hz PSG, channel index 6 = "SpO2")
  - Labels: derived from <subj>_respevt.txt (per-event apnea/hypopnea timestamps)

Segment strategy (mirrors Apnea-ECG baseline):
  - Unit: 1-minute epochs
  - Window: centre minute ± 2 min = 5-min context window (before=2, after=2)
  - Label: 1 (apnea) if ANY apnea/hypopnea event overlaps the centre minute, else 0

Output pkl keys:
  o_train / o_test  : list of ((rri_tm, rri), (ampl_tm, ampl), spo2_segment)
  y_train / y_test  : list of int labels (0/1)
  groups_train / groups_test : list of subject names

Training subjects: ucddb002, ucddb003, ucddb005  (DEMO subset)
Test subjects:     ucddb014                      (DEMO subset)
"""

import os, sys, re, shutil, pickle, tempfile, warnings
from pathlib import Path

import numpy as np
import mne
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
import biosppy.signals.tools as st
from scipy.signal import medfilt, resample
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ───────────────────────── CONFIG ───────────────────────────────────────────
UCD_DIR  = r"C:\Users\siddh\Downloads\st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0"
OUT_PKL  = os.path.join(os.path.dirname(__file__), "ucd_apnea_demo.pkl")

ECG_FS    = 128        # Lifecard EDF sampling rate
SPO2_FS_IN  = 128      # SpO2 stored at PSG rate (128 Hz) in .rec
SPO2_FS_OUT = 8        # Downsample SpO2 to 8 Hz for feature representation
TARGET_FS = 100        # Resample ECG to match Apnea-ECG baseline
EPOCH_S  = 60          # 1-minute epoch
BEFORE   = 2           # context minutes before centre
AFTER    = 2           # context minutes after  centre
WIN_S    = (BEFORE + 1 + AFTER) * EPOCH_S   # 300 s total window

# RRI interpolation grid (same as baseline: ir=3 → 900 points over 300s)
IR       = 3
TM       = np.arange(0, WIN_S, step=1.0 / IR)   # 900 points

HR_MIN, HR_MAX = 20, 300

# DEMO MODE: Reduced subset for quick validation
TRAIN_IDS = ["ucddb002", "ucddb003", "ucddb005"]
TEST_IDS  = ["ucddb014"]

# ───────────────────────── HELPERS ──────────────────────────────────────────
scaler = lambda arr: (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def parse_respevt(txt_path):
    """
    Parse _respevt.txt → list of (start_sec, end_sec) tuples for apnea/hypopnea events.
    Lines look like:
        00:29:13  HYP-C             16       89.9    4.1  ...
    Time = start; duration (s) = column 3.
    """
    events = []
    with open(txt_path, "r", errors="replace") as f:
        for line in f:
            # Match HH:MM:SS at line start followed by event type
            m = re.match(r"^\s*(\d{2}):(\d{2}):(\d{2})\s+(APNEA|HYP|CSA|MSA|OSA)", line, re.IGNORECASE)
            if m:
                hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
                t_start = hh * 3600 + mm * 60 + ss
                # Duration is the first integer after the event type column
                parts = line.split()
                dur = 0
                for p in parts[2:]:
                    try:
                        dur = int(p)
                        break
                    except ValueError:
                        continue
                events.append((t_start, t_start + dur))
    return events


def epoch_has_event(epoch_start_s, epoch_end_s, events):
    """Return True if any event overlaps with [epoch_start_s, epoch_end_s)."""
    for (ev_s, ev_e) in events:
        if ev_s < epoch_end_s and ev_e > epoch_start_s:
            return True
    return False


def read_edf_as_edf(rec_path):
    """
    Read a .rec file (which is valid EDF) by temporarily copying to .edf extension.
    Returns MNE RawEDF object.
    """
    tmp_edf = rec_path.replace(".rec", "_mne_tmp.edf")
    shutil.copy2(rec_path, tmp_edf)
    try:
        raw = mne.io.read_raw_edf(tmp_edf, preload=True, verbose=False)
    finally:
        if os.path.exists(tmp_edf):
            os.remove(tmp_edf)
    return raw


def process_subject(subj_id):
    """
    Process one UCD subject.
    Returns: (X_list, y_list, groups_list) where
      X_list[i] = ((rri_tm, rri), (ampl_tm, ampl), spo2_seg)
    """
    base   = os.path.join(UCD_DIR, subj_id)
    rec_f  = base + ".rec"
    edf_f  = base + "_lifecard.edf"
    evt_f  = base + "_respevt.txt"

    for p in (rec_f, edf_f, evt_f):
        if not os.path.exists(p):
            print(f"  [{subj_id}] Missing file: {p} — skipping")
            return [], [], []

    # ── 1. Load respiratory events ──────────────────────────────────────────
    events = parse_respevt(evt_f)

    # ── 2. Load SpO2 from PSG .rec ──────────────────────────────────────────
    psg_raw = read_edf_as_edf(rec_f)
    try:
        spo2_idx = [i for i, ch in enumerate(psg_raw.ch_names) if "spo2" in ch.lower() or "sao2" in ch.lower()]
        if not spo2_idx:
            print(f"  [{subj_id}] No SpO2 channel found — skipping")
            return [], [], []
        spo2_raw_128 = psg_raw.get_data(picks=[spo2_idx[0]])[0]  # 128 Hz
        psg_fs = psg_raw.info["sfreq"]                           # 128
    except Exception as e:
        print(f"  [{subj_id}] SpO2 load error: {e}")
        return [], [], []

    # ── 3. Load ECG from _lifecard.edf ──────────────────────────────────────
    ecg_raw = mne.io.read_raw_edf(edf_f, preload=True, verbose=False)
    ecg_signal = ecg_raw.get_data(picks=[0])[0]   # first channel (128 Hz)
    ecg_duration_s = len(ecg_signal) / ECG_FS

    # Resample ECG 128→100 Hz
    n_target = int(len(ecg_signal) * TARGET_FS / ECG_FS)
    ecg_signal = resample(ecg_signal, n_target)
    fs = TARGET_FS

    sample = fs * EPOCH_S   # 6000 samples per minute at 100 Hz

    # ── 4. Resample SpO2 128 Hz → 8 Hz ─────────────────────────────────────
    n_spo2_out = int(len(spo2_raw_128) * SPO2_FS_OUT / psg_fs)
    spo2_8hz = resample(spo2_raw_128, n_spo2_out)
    spo2_samples_per_epoch = SPO2_FS_OUT * EPOCH_S   # 480 samples per minute
    spo2_win_samples = SPO2_FS_OUT * WIN_S            # 2400 samples per 5-min window

    # Use shorter recording duration (ECG or SpO2) for epoch count
    psg_duration_s = len(spo2_raw_128) / psg_fs

    # Total epochs: limited by shortest recording
    safe_duration_s = min(ecg_duration_s, psg_duration_s)
    n_epochs = int(safe_duration_s // EPOCH_S)

    X_list, y_list, grp_list = [], [], []

    for j in tqdm(range(n_epochs), desc=subj_id, file=sys.stdout, leave=False):
        # Centre epoch boundaries
        if j < BEFORE or (j + 1 + AFTER) > n_epochs:
            continue

        # ── ECG window [j-before : j+1+after] ───────────────────────────
        ecg_start = int((j - BEFORE) * sample)
        ecg_end   = int((j + 1 + AFTER) * sample)
        if ecg_end > len(ecg_signal):
            continue
        seg = ecg_signal[ecg_start:ecg_end]

        # Bandpass filter
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

        n_beats_per_min = len(rpeaks) / (1 + BEFORE + AFTER)
        if n_beats_per_min < 40 or n_beats_per_min > 200:
            continue

        # RRI and amplitude
        rri_tm     = rpeaks[1:] / float(fs)
        rri_signal = np.diff(rpeaks) / float(fs)
        rri_signal = medfilt(rri_signal, kernel_size=3)
        ampl_tm    = rpeaks / float(fs)
        ampl_sig   = seg_f[rpeaks]

        hr = 60.0 / rri_signal
        if not np.all((hr >= HR_MIN) & (hr <= HR_MAX)):
            continue

        # ── SpO2 window ───────────────────────────────────────────────────
        spo2_start = int((j - BEFORE) * spo2_samples_per_epoch)
        spo2_end   = spo2_start + spo2_win_samples
        if spo2_end > len(spo2_8hz):
            continue
        spo2_seg = spo2_8hz[spo2_start:spo2_end].astype(np.float32)

        # Remove physiologically impossible SpO2 values
        spo2_seg = np.clip(spo2_seg, 50, 100)

        # ── Label ─────────────────────────────────────────────────────────
        epoch_start_s = j * EPOCH_S
        epoch_end_s   = (j + 1) * EPOCH_S
        label = 1.0 if epoch_has_event(epoch_start_s, epoch_end_s, events) else 0.0

        X_list.append(((rri_tm, rri_signal), (ampl_tm, ampl_sig), spo2_seg))
        y_list.append(label)
        grp_list.append(subj_id)

    return X_list, y_list, grp_list


# ───────────────────────── MAIN ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("UCD Sleep Apnea Database Preprocessing (DEMO)")
    print("=" * 65)

    o_train, y_train, groups_train = [], [], []
    o_test,  y_test,  groups_test  = [], [], []

    print(f"\nProcessing {len(TRAIN_IDS)} TRAIN subjects: {TRAIN_IDS}")
    for sid in TRAIN_IDS:
        X, y, g = process_subject(sid)
        o_train.extend(X); y_train.extend(y); groups_train.extend(g)
        print(f"  {sid}: {len(X)} epochs  (apnea={sum(y)}  normal={len(y)-sum(y)})")

    print(f"\nProcessing {len(TEST_IDS)} TEST subjects: {TEST_IDS}")
    for sid in TEST_IDS:
        X, y, g = process_subject(sid)
        o_test.extend(X); y_test.extend(y); groups_test.extend(g)
        print(f"  {sid}: {len(X)} epochs  (apnea={sum(y)}  normal={len(y)-sum(y)})")

    print(f"\nTotal train: {len(o_train)} epochs  (apnea={int(sum(y_train))})")
    print(f"Total test : {len(o_test)} epochs   (apnea={int(sum(y_test))})")

    apnea_ecg_ucd = dict(
        o_train=o_train, y_train=y_train, groups_train=groups_train,
        o_test=o_test,   y_test=y_test,   groups_test=groups_test,
    )
    with open(OUT_PKL, "wb") as f:
        pickle.dump(apnea_ecg_ucd, f, protocol=2)

    print(f"\nSaved → {OUT_PKL}")
    print("Done!")
