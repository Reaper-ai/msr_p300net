import os
import sys
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Ensure configuration is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def extract_subject_data(hdf5_path, subj_id):
    """
    Extracts pre-filtered features and labels for a single subject from GIB-UVA HDF5 database.
    """
    with h5py.File(hdf5_path, 'r') as f:
        global_subjects = f['subjects'][:].ravel()
        subj_mask = (global_subjects == subj_id)
        
        if not np.any(subj_mask): 
            return None
        
        # Features are transposed to [N, Channels=8, Time=128]
        X = f['features'][subj_mask].transpose(0, 2, 1).astype(np.float32)
        y = f['erp_labels'][subj_mask].astype(np.int64).ravel()
        trials = f['trials'][subj_mask].astype(np.int64).ravel()
        codes = f['codes'][subj_mask].astype(np.int64).ravel()
        runs = f['run_indexes'][subj_mask].astype(np.int64).ravel()
        dbs = f['database_ids'][subj_mask].astype(np.int64).ravel()
        
        char_ids = (dbs * 10000) + (runs * 100) + trials
        
    return X, y, char_ids, codes, runs, dbs

class SimpleTensorDataset(Dataset):
    """
    Custom Dataset class representing GIB-UVA epoch features, labels, and metadata.
    """
    def __init__(self, x, y, s, char_ids, codes):
        self.X = x
        self.y = y
        self.subj_ids = s
        self.char_ids = char_ids
        self.codes = codes
        
    def __len__(self): 
        return len(self.y)
        
    def __getitem__(self, i): 
        return self.X[i], self.y[i], self.subj_ids[i], self.char_ids[i], self.codes[i]

class FastTensorDataLoader:
    """
    Fast Tensor Loader that moves GIB-UVA data to device memory.
    """
    def __init__(self, dataset, batch_size, shuffle=False, device=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device if device is not None else config.DEVICE

        self.dataset.X = self.dataset.X.to(self.device)
        self.dataset.y = self.dataset.y.to(self.device)
        self.dataset.subj_ids = self.dataset.subj_ids.to(self.device)
        self.dataset.char_ids = self.dataset.char_ids.to(self.device)
        self.dataset.codes = self.dataset.codes.to(self.device)
        self.n_samples = len(self.dataset.y)

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.n_samples, device=self.device)
        else:
            self.indices = torch.arange(self.n_samples, device=self.device)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.n_samples: 
            raise StopIteration
        idx = self.indices[self.current_idx : self.current_idx + self.batch_size]
        batch_x = self.dataset.X[idx]
        batch_y = self.dataset.y[idx]
        batch_s = self.dataset.subj_ids[idx]
        batch_char = self.dataset.char_ids[idx]
        batch_c = self.dataset.codes[idx]
        self.current_idx += self.batch_size
        return batch_x, batch_y, batch_s, batch_char, batch_c

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
