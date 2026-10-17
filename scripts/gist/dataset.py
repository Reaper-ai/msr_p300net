import math
from pathlib import Path
import mne
import numpy as np
from scipy.signal import resample
import torch

import config

class SafeEEGAugmenter(torch.nn.Module):
    def __init__(self, p=0.5, noise_std=0.05, scale_range=(0.9, 1.1), max_shift=2, drop_prob=0.1):
        super().__init__()
        self.p = p
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.max_shift = max_shift
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training:
            return x
        b, ch, t = x.shape
        x_aug = x.clone()

        if torch.rand(1).item() < self.p:
            scales = torch.empty(b, 1, 1, device=x.device).uniform_(*self.scale_range)
            x_aug = x_aug * scales
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise
        if torch.rand(1).item() < self.p:
            mask = (torch.rand(b, ch, 1, device=x.device) > self.drop_prob).float()
            x_aug = x_aug * mask
        if torch.rand(1).item() < self.p:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
            if shift != 0:
                x_aug = torch.roll(x_aug, shifts=shift, dims=-1)
                if shift > 0:
                    x_aug[..., :shift] = 0
                else:
                    x_aug[..., shift:] = 0
        return x_aug

def preprocess_continuous_run(raw):
    raw.filter(0.5, 30.0, fir_design='firwin', verbose=False)
    raw.resample(config.FS, verbose=False)
    return raw.get_data(), config.FS

def extract_rsvp_data(file_list, base_seq_id=0, channels_to_use=config.CHANNELS_8):
    X_list, y_list, seq_ids = [], [], []
    curr_seq = base_seq_id

    for f in file_list:
        try:
            raw = mne.io.read_raw_eeglab(str(f), preload=True, verbose=False)
            available = {c.lower(): c for c in raw.ch_names}
            selected = [available[c.lower()] for c in channels_to_use if c.lower() in available]
            if len(selected) != len(channels_to_use):
                continue
            raw.pick(selected)

            filtered_data, orig_fs = preprocess_continuous_run(raw)
            events, event_dict = mne.events_from_annotations(raw, verbose=False)

            tgt_code = [v for k, v in event_dict.items() if str(k).strip() == '1'][0]
            pre_s = int(abs(config.T_MIN) * orig_fs)
            post_s = int(config.T_MAX * orig_fs)
            last_s_idx = -99999

            valid_events = [ev for ev in events if ev[2] in event_dict.values()]
            target_indices = set(i for i, ev in enumerate(valid_events) if ev[2] == tgt_code)

            for i, ev in enumerate(valid_events):
                s_idx, ev_type = ev[0], ev[2]

                if (s_idx - last_s_idx) > (1.5 * orig_fs):
                    curr_seq += 1
                last_s_idx = s_idx

                if ev_type != tgt_code and any(abs(i - t_idx) <= 2 for t_idx in target_indices):
                    continue

                start = s_idx - pre_s
                end = s_idx + post_s
                if start < 0 or end > filtered_data.shape[1]:
                    continue

                epoch = filtered_data[:, start:end]
                epoch -= np.mean(epoch[:, :pre_s], axis=1, keepdims=True)

                X_list.append(epoch)
                y_list.append(1 if ev_type == tgt_code else 0)
                seq_ids.append(curr_seq)
        except Exception as e:
            print(f"Error parsing RSVP file {f.name}: {e}")
            pass

    if not X_list:
        return None, None, None, curr_seq

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    s = np.array(seq_ids, dtype=np.int64)

    if X.shape[2] != config.N_SAMPLES:
        X = resample(X, config.N_SAMPLES, axis=2).astype(np.float32)
    return X, y, s, curr_seq

def extract_p300_data(file_list, base_trial_id=0, channels_to_use=config.CHANNELS_8):
    X_list, y_list, trial_ids = [], [], []
    curr_trial = base_trial_id

    for f in file_list:
        try:
            raw = mne.io.read_raw_eeglab(str(f), preload=True, verbose=False)
            available = {c.lower(): c for c in raw.ch_names}
            selected = [available[c.lower()] for c in channels_to_use if c.lower() in available]
            if len(selected) != len(channels_to_use):
                continue
            raw.pick(selected)

            filtered_data, orig_fs = preprocess_continuous_run(raw)
            events, event_dict = mne.events_from_annotations(raw, verbose=False)

            tgt_code = [v for k, v in event_dict.items() if str(k).strip() == '1'][0]
            pre_s = int(abs(config.T_MIN) * orig_fs)
            post_s = int(config.T_MAX * orig_fs)
            last_s_idx = -99999

            for ev in events:
                s_idx, ev_type = ev[0], ev[2]
                if ev_type not in event_dict.values():
                    continue

                if (s_idx - last_s_idx) > (1.5 * orig_fs):
                    curr_trial += 1
                last_s_idx = s_idx

                start = s_idx - pre_s
                end = s_idx + post_s
                if start < 0 or end > filtered_data.shape[1]:
                    continue

                epoch = filtered_data[:, start:end]
                epoch -= np.mean(epoch[:, :pre_s], axis=1, keepdims=True)

                X_list.append(epoch)
                y_list.append(1 if ev_type == tgt_code else 0)
                trial_ids.append(curr_trial)
        except Exception as e:
            print(f"Error parsing P300 file {f.name}: {e}")
            pass

    if not X_list:
        return None, None, None, curr_trial

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    t = np.array(trial_ids, dtype=np.int64)

    if X.shape[2] != config.N_SAMPLES:
        X = resample(X, config.N_SAMPLES, axis=2).astype(np.float32)
    return X, y, t, curr_trial

def load_rsvp_splits(subj, channels_to_use=config.CHANNELS_8):
    subj_path = Path(config.DATA_PATH) / f'sub-{subj:03d}' / 'eeg'
    if not subj_path.exists():
        return None
    files = sorted(list(subj_path.glob('*task-*RSVP*_eeg.set')))
    if not files:
        return None
    rsvp_data = extract_rsvp_data(files, base_seq_id=0, channels_to_use=channels_to_use)
    if rsvp_data is None or rsvp_data[0] is None:
        return None
    X, y, s, _ = rsvp_data

    u_seqs = np.unique(s)
    n_cal = max(2, int(0.50 * len(u_seqs)))
    cal_seqs, te_seqs = u_seqs[:n_cal], u_seqs[n_cal:]
    n_tr = max(1, int(0.80 * len(cal_seqs)))

    X_tr, y_tr, s_tr = X[np.isin(s, cal_seqs[:n_tr])], y[np.isin(s, cal_seqs[:n_tr])], s[np.isin(s, cal_seqs[:n_tr])]
    X_va, y_va, s_va = X[np.isin(s, cal_seqs[n_tr:])], y[np.isin(s, cal_seqs[n_tr:])], s[np.isin(s, cal_seqs[n_tr:])]
    X_te, y_te, s_te = X[np.isin(s, te_seqs)], y[np.isin(s, te_seqs)], s[np.isin(s, te_seqs)]

    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std = X_tr.std(axis=(0, 2), keepdims=True)
    return (
        ((X_tr - mean) / (std + 1e-8), y_tr, s_tr),
        ((X_va - mean) / (std + 1e-8), y_va, s_va),
        ((X_te - mean) / (std + 1e-8), y_te, s_te)
    )

def load_p300_splits(subj, channels_to_use=config.CHANNELS_8):
    subj_path = Path(config.DATA_PATH) / f'sub-{subj:03d}' / 'eeg'
    if not subj_path.exists():
        return None
    cal_files = sorted(list(subj_path.glob('*task-*P300*train*run[0-9]*_eeg.set')))[:2]
    if not cal_files:
        cal_files = sorted(list(subj_path.glob('*task-*P300*train*_eeg.set')))[:2]
    te_files = sorted(list(subj_path.glob('*task-*P300*test*_eeg.set')))[:4]

    if not cal_files or not te_files:
        return None

    cal_data = extract_p300_data(cal_files, base_trial_id=0, channels_to_use=channels_to_use)
    if cal_data is None or cal_data[0] is None:
        return None
    X_cal, y_cal, t_cal, last_t = cal_data

    te_data = extract_p300_data(te_files, base_trial_id=last_t + 1, channels_to_use=channels_to_use)
    if te_data is None or te_data[0] is None:
        return None
    X_te, y_te, t_te, _ = te_data

    u_trials = np.unique(t_cal)
    n_tr = max(1, int(0.80 * len(u_trials)))
    tr_trials, va_trials = u_trials[:n_tr], u_trials[n_tr:]

    X_tr, y_tr, t_tr = X_cal[np.isin(t_cal, tr_trials)], y_cal[np.isin(t_cal, tr_trials)], t_cal[np.isin(t_cal, tr_trials)]
    X_va, y_va, t_va = X_cal[np.isin(t_cal, va_trials)], y_cal[np.isin(t_cal, va_trials)], t_cal[np.isin(t_cal, va_trials)]

    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std = X_tr.std(axis=(0, 2), keepdims=True)
    return (
        ((X_tr - mean) / (std + 1e-8), y_tr, t_tr),
        ((X_va - mean) / (std + 1e-8), y_va, t_va),
        ((X_te - mean) / (std + 1e-8), y_te, t_te)
    )

class FastTensorDataLoader:
    def __init__(self, X, y, meta, batch_size=64, shuffle=False):
        self.X = torch.tensor(X, dtype=torch.float32).to(config.DEVICE)
        self.y = torch.tensor(y, dtype=torch.long).to(config.DEVICE)
        self.meta = torch.tensor(meta, dtype=torch.long).to(config.DEVICE)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.idx = 0
        return self
        
    def __next__(self):
        if self.idx >= len(self.X):
            raise StopIteration
        batch_idx = self.indices[self.idx : self.idx + self.batch_size]
        self.idx += self.batch_size
        return self.X[batch_idx], self.y[batch_idx], self.meta[batch_idx]
        
    def __len__(self):
        return int(math.ceil(len(self.X) / self.batch_size))
