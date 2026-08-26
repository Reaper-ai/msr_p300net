# eeginception (maybe incorct datset spliting) - 1 run
import os
import math
import random
import ctypes
import h5py
import gc
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader

def malloc_trim():
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ==========================================
# 1. CONFIGURATION, METADATA & SEEDING
# ==========================================
DATA_PATH = '/kaggle/input/datasets/esantamaria/gibuva-erpbci-dataset/GIB-UVA ERP-BCI.hdf5'
SUBJECT_IDS = list(range(73))  # 73 subjects (0 to 72)
SEEDS = [1234]
EPOCHS = 1000
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metadata Mapping from demographics table
SUBJECT_METADATA = {
    # Dataset 0 (22 Healthy)
    **{i: {'dataset': 0, 'condition': 'Healthy'} for i in range(0, 22)},
    # Dataset 1 (10 Healthy: 22-31, 16 Pathology: 32-47)
    **{i: {'dataset': 1, 'condition': 'Healthy'} for i in range(22, 32)},
    **{i: {'dataset': 1, 'condition': 'Unhealthy'} for i in range(32, 48)},
    # Dataset 2 (10 Healthy: 48-57, 15 Multiple Sclerosis: 58-72)
    **{i: {'dataset': 2, 'condition': 'Healthy'} for i in range(48, 58)},
    **{i: {'dataset': 2, 'condition': 'Unhealthy'} for i in range(58, 73)}
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. GIB-UVA DATA EXTRACTION & DATASET
# ==========================================
def extract_subject_data(hdf5_path, subj_id):
    """Extracts pre-filtered data and metadata for a single subject."""
    with h5py.File(hdf5_path, 'r') as f:
        global_subjects = f['subjects'][:].ravel()
        subj_mask = (global_subjects == subj_id)
        
        if not np.any(subj_mask): return None
        
        X = f['features'][subj_mask].transpose(0, 2, 1).astype(np.float32)  # (N, Channels, Time)
        y = f['erp_labels'][subj_mask].astype(np.int64).ravel()
        trials = f['trials'][subj_mask].astype(np.int64).ravel()
        codes = f['codes'][subj_mask].astype(np.int64).ravel()
        runs = f['run_indexes'][subj_mask].astype(np.int64).ravel()
        dbs = f['database_ids'][subj_mask].astype(np.int64).ravel()
        
        char_ids = (dbs * 10000) + (runs * 100) + trials
        
    return X, y, char_ids, codes, runs, dbs

class SimpleTensorDataset(Dataset):
    def __init__(self, x, y, s, char_ids, codes):
        self.X, self.y, self.subj_ids, self.char_ids, self.codes = x, y, s, char_ids, codes
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.subj_ids[i], self.char_ids[i], self.codes[i]

# ==========================================
# 3. PYTORCH EEG-INCEPTION ARCHITECTURE
# ==========================================
class EEGInceptionPyTorch(nn.Module):
    def __init__(self, num_channels=8, time_samples=128, filters_per_branch=8, 
                 dropout_rate=0.25, n_classes=2):
        super().__init__()
        
        # Inception Module 1
        self.conv_t64 = nn.Conv2d(1, filters_per_branch, (1, 64), padding=(0, 32), bias=False)
        self.conv_t32 = nn.Conv2d(1, filters_per_branch, (1, 32), padding=(0, 16), bias=False)
        self.conv_t16 = nn.Conv2d(1, filters_per_branch, (1, 16), padding=(0, 8), bias=False)
        
        self.bn1 = nn.BatchNorm2d(filters_per_branch * 3)
        total_b1_filters = filters_per_branch * 3
        
        self.conv_spatial = nn.Conv2d(
            total_b1_filters, total_b1_filters * 2, 
            kernel_size=(num_channels, 1), 
            groups=total_b1_filters, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(total_b1_filters * 2)
        self.pool1 = nn.AvgPool2d((4, 1))
        self.drop1 = nn.Dropout2d(dropout_rate)
        
        # Inception Module 2
        spatial_out_channels = total_b1_filters * 2
        self.conv_b2_16 = nn.Conv2d(spatial_out_channels, filters_per_branch, (1, 16), padding=(0, 8), bias=False)
        self.conv_b2_8  = nn.Conv2d(spatial_out_channels, filters_per_branch, (1, 8), padding=(0, 4), bias=False)
        self.conv_b2_4  = nn.Conv2d(spatial_out_channels, filters_per_branch, (1, 4), padding=(0, 2), bias=False)
        
        self.bn3 = nn.BatchNorm2d(filters_per_branch * 3)
        self.pool2 = nn.AvgPool2d((2, 1))
        self.drop2 = nn.Dropout2d(dropout_rate)
        
        # Output Module
        b2_out_channels = filters_per_branch * 3
        self.conv_out1 = nn.Conv2d(b2_out_channels, b2_out_channels // 2, (8, 1), padding=(4, 0), bias=False)
        self.bn4 = nn.BatchNorm2d(b2_out_channels // 2)
        self.pool3 = nn.AvgPool2d((2, 1))
        self.drop3 = nn.Dropout2d(dropout_rate)
        
        self.conv_out2 = nn.Conv2d(b2_out_channels // 2, b2_out_channels // 4, (4, 1), padding=(2, 0), bias=False)
        self.bn5 = nn.BatchNorm2d(b2_out_channels // 4)
        self.pool4 = nn.AvgPool2d((2, 1))
        self.drop4 = nn.Dropout2d(dropout_rate)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, time_samples)
            out = self.forward_features(dummy.permute(0, 1, 3, 2))
            flat_size = out.reshape(1, -1).shape[1]
            
        self.fc = nn.Linear(flat_size, n_classes)

    def forward_features(self, x):
        t64 = self.conv_t64(x)
        t32 = self.conv_t32(x)
        t16 = self.conv_t16(x)
        
        b1_concat = torch.cat([t64, t32, t16], dim=1)
        out = F.elu(self.bn1(b1_concat))
        
        out = self.conv_spatial(out)
        out = F.elu(self.bn2(out))
        out = self.pool1(out)
        out = self.drop1(out)
        
        b2_16 = self.conv_b2_16(out)
        b2_8  = self.conv_b2_8(out)
        b2_4  = self.conv_b2_4(out)
        
        b2_concat = torch.cat([b2_16, b2_8, b2_4], dim=1)
        out = F.elu(self.bn3(b2_concat))
        out = self.pool2(out)
        out = self.drop2(out)
        
        out = self.conv_out1(out)
        out = F.elu(self.bn4(out))
        out = self.pool3(out)
        out = self.drop3(out)
        
        out = self.conv_out2(out)
        out = F.elu(self.bn5(out))
        out = self.pool4(out)
        out = self.drop4(out)
        return out

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        features = self.forward_features(x)
        flattened = features.reshape(features.size(0), -1)
        return self.fc(flattened)

# ==========================================
# 4. SAMPLER & DATA LOADER
# ==========================================
class BalancedBatchSampler:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_indices = torch.where(dataset.y == 1)[0].tolist()
        self.neg_indices = torch.where(dataset.y == 0)[0].tolist()
        self.n_pos = len(self.pos_indices)
        self.n_neg = len(self.neg_indices)
        self.n_batches = max(1, (2 * min(self.n_pos, self.n_neg)) // batch_size)
    
    def __iter__(self):
        pos_shuffled = random.sample(self.pos_indices, self.n_pos)
        neg_shuffled = random.sample(self.neg_indices, self.n_neg)
        batch_indices = []
        n_pairs = min(self.n_pos, self.n_neg)
        for i in range(n_pairs):
            batch_indices.append(pos_shuffled[i % self.n_pos])
            batch_indices.append(neg_shuffled[i % self.n_neg])
        random.shuffle(batch_indices)
        for i in range(0, len(batch_indices), self.batch_size):
            yield batch_indices[i:i+self.batch_size]
    
    def __len__(self):
        return self.n_batches

class FastTensorDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(dataset.y)

    def __iter__(self):
        self.indices = torch.randperm(self.n_samples) if self.shuffle else torch.arange(self.n_samples)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.n_samples: raise StopIteration
        idx = self.indices[self.current_idx : self.current_idx + self.batch_size]
        batch_x = self.dataset.X[idx].to(device, non_blocking=True)
        batch_y = self.dataset.y[idx].to(device, non_blocking=True)
        batch_s = self.dataset.subj_ids[idx].to(device, non_blocking=True)
        batch_char = self.dataset.char_ids[idx].to(device, non_blocking=True)
        batch_c = self.dataset.codes[idx].to(device, non_blocking=True)
        self.current_idx += self.batch_size
        return batch_x, batch_y, batch_s, batch_char, batch_c

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

# ==========================================
# 5. METRICS & EVALUATION
# ==========================================
def evaluate_metrics(model, loader, device, max_seq=10, n_classes=36):
    model.eval()
    block_probs = defaultdict(lambda: defaultdict(list))
    block_targets = {}
    
    with torch.no_grad():
        for batch in loader:
            data = batch[0].to(device)
            target = batch[1].cpu().numpy().ravel()
            char_ids = batch[3].cpu().numpy().ravel()
            codes = batch[4].cpu().numpy().ravel()
            
            with torch.amp.autocast(device_type=device.type):
                probs = F.softmax(model(data), dim=1).cpu().numpy()[:, 1]
            
            for prob, tgt, cid, code in zip(probs, target, char_ids, codes):
                cid, code = int(cid), int(code)
                block_probs[cid][code].append(prob)
                if int(tgt) == 1:
                    if cid not in block_targets: block_targets[cid] = {}
                    if code <= 6:
                        block_targets[cid]['row'] = code
                    else:
                        block_targets[cid]['col'] = code

    y_true, y_pred = [], []
    for cid, gold in block_targets.items():
        if 'row' not in gold or 'col' not in gold: continue
        
        avg_probs = {c: np.mean(p_list) for c, p_list in block_probs[cid].items()}
        row_codes = {k: v for k, v in avg_probs.items() if k <= 6}
        col_codes = {k: v for k, v in avg_probs.items() if k >= 7}
        
        if not row_codes or not col_codes: continue
            
        pred_row = max(row_codes, key=row_codes.get)
        pred_col = max(col_codes, key=col_codes.get)
        
        true_obj = (gold['row'] - 1) * 6 + (gold['col'] - 7)
        pred_obj = (pred_row - 1) * 6 + (pred_col - 7)
        
        y_true.append(true_obj)
        y_pred.append(pred_obj)

    overall_acc = accuracy_score(y_true, y_pred) * 100.0 if y_true else 0.0
    overall_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) if y_true else 0.0
    
    itr_results = []
    for seq_limit in range(1, max_seq + 1):
        correct, total = 0, len(block_targets)
        for cid, gold in block_targets.items():
            if 'row' not in gold or 'col' not in gold: 
                total -= 1
                continue
                
            avg_probs = {c: np.mean(p_list[:seq_limit]) for c, p_list in block_probs[cid].items() if len(p_list) > 0}
            row_codes = {k: v for k, v in avg_probs.items() if k <= 6}
            col_codes = {k: v for k, v in avg_probs.items() if k >= 7}
            
            if not row_codes or not col_codes: continue
            
            pred_r = max(row_codes, key=row_codes.get)
            pred_c = max(col_codes, key=col_codes.get)
            
            if pred_r == gold['row'] and pred_c == gold['col']:
                correct += 1
                
        acc = correct / total if total > 0 else 0.0
        
        if acc <= (1.0 / n_classes): itr = 0.0
        elif acc >= 1.0: itr = math.log2(n_classes)
        else: itr = (math.log2(n_classes) + acc * math.log2(acc) + (1.0 - acc) * math.log2((1.0 - acc) / (n_classes - 1)))
        
        itr_results.append({'Sequences': seq_limit, 'Accuracy (%)': acc * 100.0, 'Bits/Selection': itr})

    return overall_acc, overall_f1, pd.DataFrame(itr_results)

# ==========================================
# 6. EXPERIMENTAL RUN LOOP
# ==========================================
all_run_records = []
itr_run_records = []

for seed in SEEDS:
    print(f"\n================ Running Seed: {seed} ================")
    set_seed(seed)
    
    for subj_id in SUBJECT_IDS:
        data = extract_subject_data(DATA_PATH, subj_id)
        if data is None: continue
        
        X_sub, y_sub, char_ids_sub, codes_sub, runs_sub, dbs_sub = data
        
        # Dataset ID for calibration configuration
        db_id = int(dbs_sub[0]) if len(dbs_sub) > 0 else SUBJECT_METADATA[subj_id]['dataset']
        condition = SUBJECT_METADATA[subj_id]['condition']
        
        unique_runs = np.unique(runs_sub)
        unique_runs.sort()
        
        # Datasets 0 and 1: 2 calibration runs; Dataset 2: 1 calibration run
        n_calib_runs = 2 if db_id in [0, 1] else 1
        
        if len(unique_runs) <= n_calib_runs:
            # Skip if subject doesn't have remaining test runs
            continue
            
        calib_runs = unique_runs[:n_calib_runs]
        
        calib_mask = np.isin(runs_sub, calib_runs)
        test_mask = ~calib_mask
        
        # Split calibration run characters 80/20 for Train/Val
        calib_chars = np.unique(char_ids_sub[calib_mask])
        if len(calib_chars) < 5: continue
        
        np.random.shuffle(calib_chars)
        split_train = int(0.8 * len(calib_chars))
        train_chars = calib_chars[:split_train]
        val_chars = calib_chars[split_train:]
        
        mask_tr = np.isin(char_ids_sub, train_chars)
        mask_va = np.isin(char_ids_sub, val_chars)
        mask_te = test_mask
        
        X_tr, y_tr, cid_tr, c_tr = X_sub[mask_tr], y_sub[mask_tr], char_ids_sub[mask_tr], codes_sub[mask_tr]
        X_va, y_va, cid_va, c_va = X_sub[mask_va], y_sub[mask_va], char_ids_sub[mask_va], codes_sub[mask_va]
        X_te, y_te, cid_te, c_te = X_sub[mask_te], y_sub[mask_te], char_ids_sub[mask_te], codes_sub[mask_te]
        
        if len(y_tr) == 0 or len(y_te) == 0: continue
        
        # Subject-wise Z-Score using train statistics
        mean_tr = X_tr.mean(axis=(0, 2), keepdims=True)
        std_tr = X_tr.std(axis=(0, 2), keepdims=True)
        
        X_tr = (X_tr - mean_tr) / (std_tr + 1e-8)
        X_va = (X_va - mean_tr) / (std_tr + 1e-8)
        X_te = (X_te - mean_tr) / (std_tr + 1e-8)
        
        s_tr = np.full(len(y_tr), subj_id, dtype=np.int64)
        s_va = np.full(len(y_va), subj_id, dtype=np.int64)
        s_te = np.full(len(y_te), subj_id, dtype=np.int64)
        
        X_tr_t = torch.tensor(X_tr).unsqueeze(1)
        X_va_t = torch.tensor(X_va).unsqueeze(1)
        X_te_t = torch.tensor(X_te).unsqueeze(1)
        
        tr_ds = SimpleTensorDataset(X_tr_t, torch.tensor(y_tr), torch.tensor(s_tr), torch.tensor(cid_tr), torch.tensor(c_tr))
        va_ds = SimpleTensorDataset(X_va_t, torch.tensor(y_va), torch.tensor(s_va), torch.tensor(cid_va), torch.tensor(c_va))
        te_ds = SimpleTensorDataset(X_te_t, torch.tensor(y_te), torch.tensor(s_te), torch.tensor(cid_te), torch.tensor(c_te))
        
        train_sampler = BalancedBatchSampler(tr_ds, BATCH_SIZE)
        train_loader = DataLoader(tr_ds, batch_sampler=train_sampler, num_workers=0)
        val_loader = FastTensorDataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = FastTensorDataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = EEGInceptionPyTorch(num_channels=8, time_samples=128, filters_per_branch=8, dropout_rate=0.25, n_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler(device=device.type)
        
        best_val_loss = float('inf')
        best_weights = None
        patience = 0
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for batch in train_loader:
                data, target = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    data, target = batch[0].to(device), batch[1].to(device)
                    with torch.amp.autocast(device_type=device.type):
                        output = model(data)
                        loss = criterion(output, target)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
            if patience > 50:
                break
                
        if best_weights is not None:
            model.load_state_dict(best_weights)
            
        obj_acc, obj_f1, itr_df = evaluate_metrics(model, test_loader, device)
        
        print(f"Seed {seed} | Subj {subj_id:02d} (DS {db_id}, {condition}) -> Acc: {obj_acc:.2f}% | F1: {obj_f1:.4f}")
        
        all_run_records.append({
            'Seed': seed,
            'Subject': f"SBJ{subj_id:02d}",
            'Subject_ID': subj_id,
            'Dataset_ID': db_id,
            'Condition': condition,
            'Obj Acc (%)': obj_acc,
            'Macro F1': obj_f1
        })
            
        for _, row in itr_df.iterrows():
            itr_run_records.append({
                'Seed': seed,
                'Subject': f"SBJ{subj_id:02d}",
                'Dataset_ID': db_id,
                'Condition': condition,
                'Sequences': row['Sequences'],
                'Accuracy (%)': row['Accuracy (%)'],
                'Bits/Selection': row['Bits/Selection']
            })
            
        del model, optimizer, scaler
        torch.cuda.empty_cache()
        gc.collect()
        malloc_trim()

# ==========================================
# 7. SUMMARY & BREAKDOWN REPORTING
# ==========================================
df_subjects = pd.DataFrame(all_run_records)
df_itr = pd.DataFrame(itr_run_records)
df_subjects.to_csv("eeg_inception_gibuva_subject_results.csv", index=False)
df_itr.to_csv("eeg_inception_gibuva_itr_results.csv", index=False)

def compute_stats(series):
    if len(series) == 0:
        return 0, 0, 0, 0, 0, 0, 0
    mean = series.mean()
    std = series.std()
    sem = std / math.sqrt(len(series))
    median = series.median()
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    ci95_low, ci95_high = np.percentile(series, [2.5, 97.5])
    return mean, std, sem, median, (q75 - q25), ci95_low, ci95_high

def print_metric_block(title, sub_df):
    acc_m, acc_s, acc_sem, acc_med, acc_iqr, acc_l, acc_h = compute_stats(sub_df['Obj Acc (%)'])
    f1_m, f1_s, f1_sem, f1_med, f1_iqr, f1_l, f1_h = compute_stats(sub_df['Macro F1'])
    print(f"\n--- {title} (N={len(sub_df)} runs/seeds) ---")
    print(f"  Accuracy: {acc_m:.2f}% ± {acc_s:.2f}% | Median: {acc_med:.2f}% (IQR: {acc_iqr:.2f}) | 95% CI: [{acc_l:.2f}%, {acc_h:.2f}%]")
    print(f"  Macro F1: {f1_m:.4f} ± {f1_s:.4f} | Median: {f1_med:.4f} (IQR: {f1_iqr:.4f}) | 95% CI: [{f1_l:.4f}, {f1_h:.4f}]")

print("\n" + "="*75)
print("              COMPREHENSIVE STATISTICAL BREAKDOWN")
print("="*75)

# 1. Overall
print_metric_block("OVERALL AGGREGATE", df_subjects)

# 2. Dataset-wise Breakdown
print("\n" + "="*75)
print("                    DATASET-WISE BREAKDOWN")
print("="*75)
for ds_id in sorted(df_subjects['Dataset_ID'].unique()):
    sub_df = df_subjects[df_subjects['Dataset_ID'] == ds_id]
    print_metric_block(f"Dataset {ds_id}", sub_df)

# 3. Condition-wise Breakdown (Healthy vs. Unhealthy)
print("\n" + "="*75)
print("                   CONDITION BREAKDOWN")
print("="*75)
for cond in ['Healthy', 'Unhealthy']:
    sub_df = df_subjects[df_subjects['Condition'] == cond]
    print_metric_block(f"Condition: {cond}", sub_df)

print("\n" + "="*75)

#HST net with correct train dataset spliting [datset 1 and 2 - 2 run datset 3 - 1 run] 
import os
import copy
import math
import random
import ctypes
import h5py
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader

def malloc_trim():
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ==========================================
# 1. CONFIGURATION & DEMOGRAPHIC METADATA
# ==========================================
DATA_PATH = '/kaggle/input/datasets/esantamaria/gibuva-erpbci-dataset/GIB-UVA ERP-BCI.hdf5'
SUBJECT_IDS = list(range(73))  # 73 subjects (0 to 72)
SEEDS = [1234]
EPOCHS = 200
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 25
MIN_DELTA = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUBJECT_METADATA = {
    # Dataset 0: Subjects 0-21 (Healthy)
    **{i: {'dataset': 0, 'condition': 'Healthy'} for i in range(0, 22)},
    # Dataset 1: Subjects 22-31 (Healthy), Subjects 32-47 (Pathology/Unhealthy)
    **{i: {'dataset': 1, 'condition': 'Healthy'} for i in range(22, 32)},
    **{i: {'dataset': 1, 'condition': 'Unhealthy'} for i in range(32, 48)},
    # Dataset 2: Subjects 48-57 (Healthy), Subjects 58-72 (Multiple Sclerosis/Unhealthy)
    **{i: {'dataset': 2, 'condition': 'Healthy'} for i in range(48, 58)},
    **{i: {'dataset': 2, 'condition': 'Unhealthy'} for i in range(58, 73)}
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. DATA LOADING & EXTRACTION
# ==========================================
def extract_subject_data(hdf5_path, subj_id):
    """Extracts pre-filtered ERP epochs and metadata for a single subject."""
    with h5py.File(hdf5_path, 'r') as f:
        global_subjects = f['subjects'][:].ravel()
        subj_mask = (global_subjects == subj_id)
        
        if not np.any(subj_mask):
            return None
        
        X = f['features'][subj_mask].transpose(0, 2, 1).astype(np.float32)  # (N, Channels=8, Time=128)
        y = f['erp_labels'][subj_mask].astype(np.int64).ravel()
        trials = f['trials'][subj_mask].astype(np.int64).ravel()
        codes = f['codes'][subj_mask].astype(np.int64).ravel()
        runs = f['run_indexes'][subj_mask].astype(np.int64).ravel()
        dbs = f['database_ids'][subj_mask].astype(np.int64).ravel()
        
        char_ids = (dbs * 10000) + (runs * 100) + trials
        
    return X, y, char_ids, codes, runs, dbs

class SimpleTensorDataset(Dataset):
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
    def __init__(self, dataset, batch_size, shuffle=False, device=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device if device is not None else DEVICE

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

# ==========================================
# 3. AUGMENTATION & ARCHITECTURE
# ==========================================
class SafeEEGAugmenter(nn.Module):
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
        # Accepts (B, C, T) or (B, 1, C, T)
        orig_dim = x.dim()
        if orig_dim == 4:
            x_aug = x.squeeze(1).clone()
        else:
            x_aug = x.clone()

        b, ch, t = x_aug.shape

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
                    
        return x_aug.unsqueeze(1) if orig_dim == 4 else x_aug

class StrictMultiScaleStemBlock(nn.Module):
    def __init__(self, in_channels=1, spatial_channels=12, depth_multiplier=2, eeg_channels=8):
        super().__init__()
        self.conv_t15 = nn.Conv2d(in_channels, spatial_channels // 3, (1, 15), padding=(0, 7), bias=False)
        self.conv_t31 = nn.Conv2d(in_channels, spatial_channels // 3, (1, 31), padding=(0, 15), bias=False)
        self.conv_t63 = nn.Conv2d(in_channels, spatial_channels // 3, (1, 63), padding=(0, 31), bias=False)
        self.bn_t = nn.BatchNorm2d(spatial_channels)
        out_spatial = spatial_channels * depth_multiplier
        self.conv_s = nn.Conv2d(spatial_channels, out_spatial, (eeg_channels, 1), groups=spatial_channels, bias=False)
        self.bn_s = nn.BatchNorm2d(out_spatial)

    def forward(self, x):
        t_concat = torch.cat([self.conv_t15(x), self.conv_t31(x), self.conv_t63(x)], dim=1)
        return F.elu(self.bn_s(self.conv_s(self.bn_t(t_concat))))

class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=2):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1_dw = nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, pad), dilation=(1, dilation), groups=channels, bias=False)
        self.conv1_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2_dw = nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, pad), dilation=(1, dilation), groups=channels, bias=False)
        self.conv2_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1_pw(self.conv1_dw(x))))
        out = self.bn2(self.conv2_pw(self.conv2_dw(out)))
        return F.elu(out + x)

class ResidualSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv_dw = nn.Conv2d(in_channels, in_channels, (1, kernel_size), padding=(0, pad), groups=in_channels, bias=False)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.skip_proj = nn.Conv2d(in_channels, out_channels, (1, 1), bias=False)

    def forward(self, x):
        out = self.conv_pw(self.conv_dw(x))
        return F.elu(self.bn(out) + self.skip_proj(x))

class RefinedMultiScaleEEGNet(nn.Module):
    def __init__(self, n_channels=8, n_samples=128):
        super().__init__()
        self.block1 = StrictMultiScaleStemBlock(in_channels=1, spatial_channels=12, depth_multiplier=2, eeg_channels=n_channels)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(0.25)
        
        self.block2 = ResidualTemporalBlock(channels=24, kernel_size=5, dilation=2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop2 = nn.Dropout(0.25)
        
        self.block3 = ResidualSeparableBlock(in_channels=24, out_channels=32, kernel_size=5)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2))
        self.drop3 = nn.Dropout(0.30)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            d_out = self.drop1(self.pool1(self.block1(dummy)))
            d_out = self.drop2(self.pool2(self.block2(d_out)))
            d_out = self.drop3(self.pool3(self.block3(d_out)))
            self.flat_dim = d_out.numel()

        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(self.flat_dim, 1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.drop1(self.pool1(self.block1(x)))
        x = self.drop2(self.pool2(self.block2(x)))
        x = self.drop3(self.pool3(self.block3(x)))
        return self.classifier(torch.flatten(x, 1))

def apply_max_norm(model):
    with torch.no_grad():
        w1 = model.block1.conv_s.weight
        norm1 = w1.norm(2, dim=(1, 2, 3), keepdim=True)
        w1.mul_(torch.clamp(norm1, max=1.0) / (norm1 + 1e-8))
        
        wd = model.classifier[-1].weight
        normd = wd.norm(2, dim=1, keepdim=True)
        wd.mul_(torch.clamp(normd, max=0.25) / (normd + 1e-8))

# ==========================================
# 4. LOSS FUNCTION
# ==========================================
class SmoothBalancedFocalLoss(nn.Module):
    def __init__(self, gamma=1.36, pos_weight=5.75, smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets_smooth = targets * (1.0 - self.smoothing) + (1.0 - targets) * self.smoothing
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets_smooth,
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device),
            reduction='none'
        )
        with torch.no_grad():
            probs = torch.sigmoid(inputs)
            pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        return ((1.0 - pt) ** self.gamma * bce_loss).mean()

# ==========================================
# 5. TRAINING, VALIDATION & EVALUATION
# ==========================================
def train_subject_epoch(model, loader, optimizer, scheduler, criterion, amp_scaler, augmenter):
    model.train()
    augmenter.train()
    running_loss = 0.0
    for batch in loader:
        data = augmenter(batch[0])
        target = batch[1].float().unsqueeze(1)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=DEVICE.type):
            output = model(data)
            loss = criterion(output, target)
            
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        scale_before = amp_scaler.get_scale()
        amp_scaler.step(optimizer)
        amp_scaler.update()
        apply_max_norm(model)
        
        if not (scale_before > amp_scaler.get_scale()):
            scheduler.step()
            
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate_val_loss(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            data = batch[0]
            target = batch[1].float().unsqueeze(1)
            with torch.amp.autocast(device_type=DEVICE.type):
                output = model(data)
                loss = criterion(output, target)
            val_loss += loss.item()
    return val_loss / max(1, len(loader))

def evaluate_comprehensive_metrics_rcp(model, loader, max_seq=10, n_classes=36):
    model.eval()
    block_scores = defaultdict(lambda: defaultdict(list))
    block_targets = {}
    
    with torch.no_grad():
        for batch in loader:
            data = batch[0]
            target = batch[1].cpu().numpy().ravel()
            char_ids = batch[3].cpu().numpy().ravel()
            codes = batch[4].cpu().numpy().ravel()
            
            with torch.amp.autocast(device_type=DEVICE.type):
                # Raw single-logit output per stimulus
                logits = model(data).squeeze(1).cpu().numpy().ravel()
            
            for logit, tgt, cid, code in zip(logits, target, char_ids, codes):
                cid, code = int(cid), int(code)
                block_scores[cid][code].append(logit)
                if int(tgt) == 1:
                    if cid not in block_targets:
                        block_targets[cid] = {}
                    if code <= 6:
                        block_targets[cid]['row'] = code
                    else:
                        block_targets[cid]['col'] = code

    y_true, y_pred = [], []
    for cid, gold in block_targets.items():
        if 'row' not in gold or 'col' not in gold:
            continue
        
        avg_scores = {c: np.mean(s_list) for c, s_list in block_scores[cid].items()}
        row_codes = {k: v for k, v in avg_scores.items() if k <= 6}
        col_codes = {k: v for k, v in avg_scores.items() if k >= 7}
        
        if not row_codes or not col_codes:
            continue
            
        pred_row = max(row_codes, key=row_codes.get)
        pred_col = max(col_codes, key=col_codes.get)
        
        true_obj = (gold['row'] - 1) * 6 + (gold['col'] - 7)
        pred_obj = (pred_row - 1) * 6 + (pred_col - 7)
        
        y_true.append(true_obj)
        y_pred.append(pred_obj)

    overall_acc = accuracy_score(y_true, y_pred) * 100.0 if y_true else 0.0
    overall_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) if y_true else 0.0
    
    itr_results = []
    for seq_limit in range(1, max_seq + 1):
        correct, total = 0, len(block_targets)
        for cid, gold in block_targets.items():
            if 'row' not in gold or 'col' not in gold:
                total -= 1
                continue
                
            avg_scores = {c: np.mean(s_list[:seq_limit]) for c, s_list in block_scores[cid].items() if len(s_list) > 0}
            row_codes = {k: v for k, v in avg_scores.items() if k <= 6}
            col_codes = {k: v for k, v in avg_scores.items() if k >= 7}
            
            if not row_codes or not col_codes:
                continue
            
            pred_r = max(row_codes, key=row_codes.get)
            pred_c = max(col_codes, key=col_codes.get)
            
            if pred_r == gold['row'] and pred_c == gold['col']:
                correct += 1
                
        acc = correct / total if total > 0 else 0.0
        
        if acc <= (1.0 / n_classes):
            itr = 0.0
        elif acc >= 1.0:
            itr = math.log2(n_classes)
        else:
            itr = (math.log2(n_classes) + acc * math.log2(acc) + (1.0 - acc) * math.log2((1.0 - acc) / (n_classes - 1)))
            
        itr_results.append({'Sequences': seq_limit, 'Accuracy (%)': acc * 100.0, 'Bits/Selection': itr})

    return overall_acc, overall_f1, pd.DataFrame(itr_results)

# ==========================================
# 6. MAIN MULTI-SEED EXPERIMENTAL LOOP
# ==========================================
augmenter = SafeEEGAugmenter(p=0.5, noise_std=0.05).to(DEVICE)
all_run_records = []
itr_run_records = []

for seed in SEEDS:
    print(f"\n==================================================")
    print(f"               RUNNING SEED: {seed}")
    print(f"==================================================")
    set_seed(seed)
    
    for subj_id in SUBJECT_IDS:
        data = extract_subject_data(DATA_PATH, subj_id)
        if data is None:
            continue
        
        X_sub, y_sub, char_ids_sub, codes_sub, runs_sub, dbs_sub = data
        
        db_id = int(dbs_sub[0]) if len(dbs_sub) > 0 else SUBJECT_METADATA[subj_id]['dataset']
        condition = SUBJECT_METADATA[subj_id]['condition']
        
        unique_runs = np.unique(runs_sub)
        unique_runs.sort()
        
        # Calibration Runs: Datasets 0 and 1 -> 2 runs; Dataset 2 -> 1 run
        n_calib_runs = 2 if db_id in [0, 1] else 1
        
        if len(unique_runs) <= n_calib_runs:
            continue
            
        calib_runs = unique_runs[:n_calib_runs]
        calib_mask = np.isin(runs_sub, calib_runs)
        test_mask = ~calib_mask
        
        # Character-level 80/20 train/val split on calibration runs
        calib_chars = np.unique(char_ids_sub[calib_mask])
        if len(calib_chars) < 5:
            continue
        
        np.random.shuffle(calib_chars)
        split_train = int(0.8 * len(calib_chars))
        train_chars = calib_chars[:split_train]
        val_chars = calib_chars[split_train:]
        
        mask_tr = np.isin(char_ids_sub, train_chars)
        mask_va = np.isin(char_ids_sub, val_chars)
        mask_te = test_mask
        
        X_tr, y_tr, cid_tr, c_tr = X_sub[mask_tr], y_sub[mask_tr], char_ids_sub[mask_tr], codes_sub[mask_tr]
        X_va, y_va, cid_va, c_va = X_sub[mask_va], y_sub[mask_va], char_ids_sub[mask_va], codes_sub[mask_va]
        X_te, y_te, cid_te, c_te = X_sub[mask_te], y_sub[mask_te], char_ids_sub[mask_te], codes_sub[mask_te]
        
        if len(y_tr) == 0 or len(y_te) == 0:
            continue
        
        # Subject-wise Z-Score normalization based ONLY on train data
        mean_tr = X_tr.mean(axis=(0, 2), keepdims=True)
        std_tr = X_tr.std(axis=(0, 2), keepdims=True)
        
        X_tr = (X_tr - mean_tr) / (std_tr + 1e-8)
        X_va = (X_va - mean_tr) / (std_tr + 1e-8)
        X_te = (X_te - mean_tr) / (std_tr + 1e-8)
        
        s_tr = np.full(len(y_tr), subj_id, dtype=np.int64)
        s_va = np.full(len(y_va), subj_id, dtype=np.int64)
        s_te = np.full(len(y_te), subj_id, dtype=np.int64)
        
        tr_ds = SimpleTensorDataset(torch.tensor(X_tr).unsqueeze(1), torch.tensor(y_tr), torch.tensor(s_tr), torch.tensor(cid_tr), torch.tensor(c_tr))
        va_ds = SimpleTensorDataset(torch.tensor(X_va).unsqueeze(1), torch.tensor(y_va), torch.tensor(s_va), torch.tensor(cid_va), torch.tensor(c_va))
        te_ds = SimpleTensorDataset(torch.tensor(X_te).unsqueeze(1), torch.tensor(y_te), torch.tensor(s_te), torch.tensor(cid_te), torch.tensor(c_te))
        
        train_loader = FastTensorDataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, device=DEVICE)
        val_loader = FastTensorDataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, device=DEVICE)
        test_loader = FastTensorDataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, device=DEVICE)
        
        model = RefinedMultiScaleEEGNet(n_channels=8, n_samples=128).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1.5e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=4e-3, epochs=EPOCHS, 
            steps_per_epoch=max(1, len(train_loader)), pct_start=0.2
        )
        criterion = SmoothBalancedFocalLoss(gamma=1.36, pos_weight=5.75, smoothing=0.05).to(DEVICE)
        scaler = torch.amp.GradScaler(device=DEVICE.type)
        
        best_val_loss = float('inf')
        best_weights = None
        patience = 0
        
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_subject_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, augmenter)
            val_loss = evaluate_val_loss(model, val_loader, criterion)
            
            if val_loss < (best_val_loss - MIN_DELTA):
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                
            if patience >= EARLY_STOP_PATIENCE:
                break
                
        if best_weights is not None:
            model.load_state_dict(best_weights)
            
        obj_acc, obj_f1, itr_df = evaluate_comprehensive_metrics_rcp(model, test_loader)
        
        print(f"Seed {seed} | Subj {subj_id:02d} (DS{db_id}, {condition[:4]}) -> Acc: {obj_acc:6.2f}% | F1: {obj_f1:.4f}")
        
        all_run_records.append({
            'Seed': seed,
            'Subject': f"SBJ{subj_id:02d}",
            'Subject_ID': subj_id,
            'Dataset_ID': db_id,
            'Condition': condition,
            'Obj Acc (%)': obj_acc,
            'Macro F1': obj_f1
        })
            
        for _, row in itr_df.iterrows():
            itr_run_records.append({
                'Seed': seed,
                'Subject': f"SBJ{subj_id:02d}",
                'Dataset_ID': db_id,
                'Condition': condition,
                'Sequences': row['Sequences'],
                'Accuracy (%)': row['Accuracy (%)'],
                'Bits/Selection': row['Bits/Selection']
            })
            
        del model, optimizer, scheduler, criterion, scaler
        torch.cuda.empty_cache()
        gc.collect()
        malloc_trim()

# ==========================================
# 7. EXPORT & STATISTICAL BREAKDOWN
# ==========================================
df_subjects = pd.DataFrame(all_run_records)
df_itr = pd.DataFrame(itr_run_records)
df_subjects.to_csv("refined_eegnet_gibuva_subject_results.csv", index=False)
df_itr.to_csv("refined_eegnet_gibuva_itr_results.csv", index=False)

def compute_stats(series):
    if len(series) == 0:
        return 0, 0, 0, 0, 0, 0, 0
    mean = series.mean()
    std = series.std()
    sem = std / math.sqrt(len(series))
    median = series.median()
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    ci95_low, ci95_high = np.percentile(series, [2.5, 97.5])
    return mean, std, sem, median, (q75 - q25), ci95_low, ci95_high

def print_metric_block(title, sub_df):
    acc_m, acc_s, acc_sem, acc_med, acc_iqr, acc_l, acc_h = compute_stats(sub_df['Obj Acc (%)'])
    f1_m, f1_s, f1_sem, f1_med, f1_iqr, f1_l, f1_h = compute_stats(sub_df['Macro F1'])
    print(f"\n--- {title} (N={len(sub_df)} runs/seeds) ---")
    print(f"  Accuracy: {acc_m:.2f}% ± {acc_s:.2f}% | Median: {acc_med:.2f}% (IQR: {acc_iqr:.2f}) | 95% CI: [{acc_l:.2f}%, {acc_h:.2f}%]")
    print(f"  Macro F1: {f1_m:.4f} ± {f1_s:.4f} | Median: {f1_med:.4f} (IQR: {f1_iqr:.4f}) | 95% CI: [{f1_l:.4f}, {f1_h:.4f}]")

print("\n" + "="*80)
print("                   COMPREHENSIVE STATISTICAL REPORT")
print("="*80)

# Overall
print_metric_block("OVERALL AGGREGATE", df_subjects)

# Dataset-wise Breakdown
print("\n" + "="*80)
print("                       DATASET-WISE BREAKDOWN")
print("="*80)
for ds_id in sorted(df_subjects['Dataset_ID'].unique()):
    sub_df = df_subjects[df_subjects['Dataset_ID'] == ds_id]
    print_metric_block(f"Dataset {ds_id}", sub_df)

# Condition-wise Breakdown
print("\n" + "="*80)
print("                      CONDITION-WISE BREAKDOWN")
print("="*80)
for cond in ['Healthy', 'Unhealthy']:
    sub_df = df_subjects[df_subjects['Condition'] == cond]
    print_metric_block(f"Condition: {cond}", sub_df)

print("\n" + "="*80)