import mne
from pathlib import Path

DATA_PATH = '/home/gaurav/Programming/bci/datasets/gist/Won2022_BIDS'
f_path = Path(DATA_PATH) / 'sub-001' / 'eeg' / 'sub-001_task-P300trainrun1_run-6_eeg.set'

raw = mne.io.read_raw_eeglab(str(f_path), preload=False, verbose=False)
events, event_dict = mne.events_from_annotations(raw, verbose=False)
print("Event dict keys and values:")
print(event_dict)

# Print unique event types present in raw events
print("Unique event types in events array:")
print(set(events[:, 2]))
