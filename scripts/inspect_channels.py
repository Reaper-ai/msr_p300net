import mne
from pathlib import Path

DATA_PATH = '/home/gaurav/Programming/bci/datasets/gist/Won2022_BIDS'
f_path = Path(DATA_PATH) / 'sub-001' / 'eeg' / 'sub-001_task-P300trainrun1_run-6_eeg.set'

raw = mne.io.read_raw_eeglab(str(f_path), preload=False, verbose=False)
print("Available channels in dataset:")
print(raw.ch_names)
print(f"Total number of channels: {len(raw.ch_names)}")
