import os
import mne
import numpy as np
from pathlib import Path

DATA_PATH = '/home/gaurav/Programming/bci/datasets/gist/Won2022_BIDS'
won_path = Path(DATA_PATH)
CHANNELS_8 = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'O1', 'O2']

def preprocess_continuous_run(raw):
    raw.filter(0.5, 30.0, fir_design='firwin', verbose=False)
    raw.resample(128, np_eval='slow', verbose=False)
    return raw.get_data(), 128

def extract_trials_from_annotations(file_path):
    try:
        raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        orig_fs = raw.info['sfreq']
        
        # We calculate trial transitions when stimulus event interval > 1.5s
        trial_starts = []
        curr_trial = 0
        last_s_idx = -99999
        
        event_count = 0
        for ev in events:
            s_idx, ev_type = ev[0], ev[2]
            if ev_type not in event_dict.values():
                continue
            event_count += 1
            if (s_idx - last_s_idx) > (1.5 * orig_fs):
                curr_trial += 1
                trial_starts.append(s_idx)
            last_s_idx = s_idx
            
        return curr_trial, event_count
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return 0, 0

print("Parsing sub-001 P300 files to extract trials & event statistics...")
subj_path = won_path / 'sub-001' / 'eeg'
train_files = sorted(list(subj_path.glob('*task-*P300*train*.set')))
test_files = sorted(list(subj_path.glob('*task-*P300*test*.set')))

print("\n--- Train Files ---")
for f in train_files:
    trials, events = extract_trials_from_annotations(f)
    print(f"{f.name}: calculated trials = {trials}, total flash events = {events}")

print("\n--- Test Files ---")
for f in test_files:
    trials, events = extract_trials_from_annotations(f)
    print(f"{f.name}: calculated trials = {trials}, total flash events = {events}")
