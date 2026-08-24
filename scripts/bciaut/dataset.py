import os
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample
from scipy.io import loadmat

# Ensure local imports work by adding directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def load_session_data(session_path, phase="Train"):
    """
    Loads EEG epochs, targets, events, and runs per block for a single subject session.
    """
    dir_phase = "Train" if phase in ["Train", "Val"] else "Test"
    mat_file = session_path / dir_phase / f"{dir_phase.lower()}Data.mat"
    target_file = session_path / dir_phase / f"{dir_phase.lower()}Targets.txt"
    event_file = session_path / dir_phase / f"{dir_phase.lower()}Events.txt"
    
    if not mat_file.exists() or not target_file.exists() or not event_file.exists():
        return None, None, None, None
        
    try:
        mat_data = loadmat(str(mat_file))
    except Exception as e:
        print(f"Error loading {mat_file}: {e}")
        return None, None, None, None
        
    # Extract keys containing the 3D EEG data
    data_keys = [k for k in mat_data if not k.startswith("_") and mat_data[k].ndim == 3]
    if not data_keys: 
        return None, None, None, None
        
    epochs = mat_data[data_keys[0]].astype(np.float32)
    targets = np.loadtxt(str(target_file), dtype=int).ravel()
    events = np.loadtxt(str(event_file), dtype=int).ravel()
    
    runs_per_block = 10
    rpb_file = session_path / dir_phase / "runs_per_block.txt"
    if rpb_file.exists():
        runs_per_block = int(np.loadtxt(str(rpb_file)))
        
    n_trials = min(epochs.shape[2], len(targets), len(events))
    if n_trials == 0: 
        return None, None, None, None
        
    epochs = epochs[:, :, :n_trials]
    targets = targets[:n_trials]
    events = events[:n_trials]
    
    # Transpose epochs from (channels, time, trials) to (trials, channels, time)
    X = epochs.transpose(2, 0, 1)
    return X, targets, events, runs_per_block

class BCIAUTDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading and preprocessing BCIAUT EEG data.
    """
    def __init__(self, data_dir, subjects, sessions, phase="Train", scalers=None):
        self.data_dir = Path(data_dir)
        self.subjects = subjects
        self.sessions = sessions
        self.phase = phase
        self.scalers = scalers if scalers is not None else {}
        self.X, self.y, self.subj_ids, self.group_ids, self.session_ids = self._load_all_data()
        
    def _load_all_data(self):
        X_list, y_list, subj_list, group_list, sess_list = [], [], [], [], []
        global_block_counter = 0
        
        for subj in self.subjects:
            subj_X_list, subj_y_list, subj_s_list, subj_g_list, subj_ses_list = [], [], [], [], []
            for sess in self.sessions:
                sess_dir = self.data_dir / f"SBJ{subj:02d}" / f"S{sess:02d}"
                if sess_dir.exists():
                    X, y, events, runs_per_block = load_session_data(sess_dir, self.phase)
                    if X is not None and len(y) > 0:
                        n_trials = len(y)
                        subj_X_list.append(X)
                        subj_y_list.append(y)
                        subj_s_list.append(np.full(n_trials, subj-1, dtype=np.int64))
                        subj_ses_list.append(np.full(n_trials, sess, dtype=np.int64))
                        
                        # Compute unique block IDs
                        trials_per_block = 8 * runs_per_block
                        block_ids = np.arange(n_trials) // trials_per_block + global_block_counter
                        if len(block_ids) > 0: 
                            global_block_counter = block_ids[-1] + 1
                        subj_g_list.append((block_ids * 100) + events)
                        
            if not subj_X_list: 
                continue
                
            subj_X = np.concatenate(subj_X_list, axis=0)
            subj_y = np.concatenate(subj_y_list, axis=0)
            
            # Crop to window containing P300 response
            subj_X = subj_X[:, :, 25:300]
            # Downsample to 128 Hz [140 samples for -200 to 1000 ms window]
            subj_X = resample(subj_X, 140, axis=2)
            
            # Normalize signals per subject using training stats
            if self.phase == "Train" and subj not in self.scalers:
                mean = subj_X.mean(axis=(0,2), keepdims=True)
                std = subj_X.std(axis=(0,2), keepdims=True)
                self.scalers[subj] = {"mean": mean, "std": std}
                
            if subj in self.scalers:
                subj_X -= self.scalers[subj]["mean"]
                subj_X /= (self.scalers[subj]["std"] + 1e-8)
                
            X_list.append(subj_X)
            y_list.append(subj_y)
            subj_list.extend(subj_s_list)
            sess_list.extend(subj_ses_list)
            group_list.extend(subj_g_list)
            
        if len(X_list) == 0:
            raise RuntimeError(f"Dataset empty for phase={self.phase}")
            
        return (
            torch.tensor(np.concatenate(X_list, axis=0), dtype=torch.float32),
            torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.long),
            torch.tensor(np.concatenate(subj_list, axis=0), dtype=torch.long),
            torch.tensor(np.concatenate(group_list, axis=0), dtype=torch.long),
            torch.tensor(np.concatenate(sess_list, axis=0), dtype=torch.long)
        )
        
    def __len__(self): 
        return len(self.y)
        
    def __getitem__(self, i): 
        return self.X[i], self.y[i], self.subj_ids[i], self.group_ids[i], self.session_ids[i]

class FoldDataset(torch.utils.data.Dataset):
    """
    Subdataset specifically for cross-validation folds.
    """
    def __init__(self, x, y, s, g, ses):
        self.X, self.y = x, y
        self.subj_ids, self.group_ids, self.session_ids = s, g, ses
        
    def __len__(self): 
        return len(self.y)
        
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx], self.subj_ids[idx], self.group_ids[idx], self.session_ids[idx]

class FastTensorDataLoader:
    """
    Optimized dataloader that moves all data to the accelerator device once.
    """
    def __init__(self, dataset, batch_size, shuffle=False, device=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device if device is not None else config.DEVICE

        # Pre-load dataset tensors to the specified device
        self.dataset.X = self.dataset.X.to(self.device)
        self.dataset.y = self.dataset.y.to(self.device)
        self.dataset.subj_ids = self.dataset.subj_ids.to(self.device)
        self.dataset.group_ids = self.dataset.group_ids.to(self.device)
        self.dataset.session_ids = self.dataset.session_ids.to(self.device)

        self.n_samples = len(self.dataset.y)

    def __iter__(self):
        self.indices = torch.randperm(self.n_samples) if self.shuffle else torch.arange(self.n_samples)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration
        idx = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch_x = self.dataset.X[idx]
        batch_y = self.dataset.y[idx]
        batch_s = self.dataset.subj_ids[idx]
        batch_g = self.dataset.group_ids[idx]
        batch_ses = self.dataset.session_ids[idx]
        self.current_idx += self.batch_size
        return batch_x, batch_y, batch_s, batch_g, batch_ses

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

def get_augmented_subject_subset(fold_dataset, batch_size, noise_std=0.05):
    """
    Applies real-time GPU-based data augmentation for P300 target trials.
    Augmentation techniques include scaling, random temporal shifts, channel dropping, and additive noise.
    """
    X_sub = fold_dataset.X.to(config.DEVICE)
    y_sub = fold_dataset.y.to(config.DEVICE)
    s_sub = fold_dataset.subj_ids.to(config.DEVICE)
    g_sub = fold_dataset.group_ids.to(config.DEVICE)
    ses_sub = fold_dataset.session_ids.to(config.DEVICE)

    target_mask = (y_sub == 1)
    X_targets = X_sub[target_mask].clone()
    y_targets = y_sub[target_mask]
    s_targets = s_sub[target_mask]
    g_targets = g_sub[target_mask]
    ses_targets = ses_sub[target_mask]

    if len(y_targets) > 0:
        # 1. Random amplitude scaling
        scales = torch.empty(X_targets.shape[0], 1, 1, device=config.DEVICE).uniform_(0.8, 1.2)
        X_targets.mul_(scales)

        # 2. Random temporal shifts
        shifts = torch.randint(-15, 16, (X_targets.shape[0],), device=config.DEVICE)
        for i, shift in enumerate(shifts):
            X_targets[i] = torch.roll(X_targets[i], shifts=shift.item(), dims=-1)

        # 3. Channel dropout
        channels_to_drop = torch.randint(0, X_targets.shape[1], (X_targets.shape[0],), device=config.DEVICE)
        for i, ch in enumerate(channels_to_drop):
            X_targets[i, ch, :] = 0.0

        # 4. Additive Gaussian noise
        X_targets.add_(torch.randn_like(X_targets) * noise_std)

        # Concatenate original and augmented data
        X_final = torch.cat([X_sub, X_targets], dim=0)
        y_final = torch.cat([y_sub, y_targets], dim=0)
        s_final = torch.cat([s_sub, s_targets], dim=0)
        g_final = torch.cat([g_sub, g_targets], dim=0)
        ses_final = torch.cat([ses_sub, ses_targets], dim=0)
    else:
        X_final, y_final, s_final, g_final, ses_final = X_sub, y_sub, s_sub, g_sub, ses_sub

    aug_ds = FoldDataset(X_final, y_final, s_final, g_final, ses_final)
    return FastTensorDataLoader(aug_ds, batch_size=batch_size, shuffle=True, device=config.DEVICE)
