import os
import sys
import copy
import math
import random
import ctypes
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader

# Add parent scripts/gib directory to python path for shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from dataset import extract_subject_data, SimpleTensorDataset, FastTensorDataLoader
from evaluation import evaluate_metrics

# Local model imports
from model import P3CNET

# Setup path configuration specific to P3CNET
RESULT_DIR = os.path.join(config.RESULT_DIR, "p3cnet")
os.makedirs(RESULT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n[Global Seed] {seed}")

def malloc_trim():
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="P3CNET GIB Pipeline")
    parser.add_argument("--subjects", type=int, nargs="+", default=config.SUBJECT_IDS,
                        help="List of subjects to run (0-72)")
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS,
                        help="Seeds to run")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--patience", type=int, default=45,
                        help="Early stopping patience")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save result files (for testing/debugging)")
    return parser.parse_args()

def main():
    args = parse_args()
    subjects = args.subjects
    seeds = args.seeds
    epochs = args.epochs
    batch_size = args.batch_size
    patience_limit = args.patience
    
    print(f"Device: {config.DEVICE}")
    print(f"Dataset path: {config.DATA_PATH}")
    print(f"Subjects to process: {subjects}")
    print(f"Seeds to evaluate: {seeds}")
    
    all_run_records = []
    itr_run_records = []
    existing_pairs = set()
    csv_path = os.path.join(RESULT_DIR, "subject_results.csv")
    itr_csv_path = os.path.join(RESULT_DIR, "itr_results.csv")
    
    if not args.no_save:
        if os.path.exists(csv_path):
            try:
                df_exist = pd.read_csv(csv_path)
                all_run_records = df_exist.to_dict('records')
                for r in all_run_records:
                    if 'Seed' in r and 'Subject_ID' in r:
                        existing_pairs.add((int(r['Seed']), int(r['Subject_ID'])))
                print(f"Loaded {len(existing_pairs)} completed subject-seed pairs from: {csv_path}")
            except Exception as e:
                print(f"Warning: Could not load existing CSV: {e}")
        if os.path.exists(itr_csv_path):
            try:
                itr_run_records = pd.read_csv(itr_csv_path).to_dict('records')
            except Exception:
                pass
    
    for seed in seeds:
        print(f"\n================ Running Seed: {seed} ================")
        set_seed(seed)
        
        for subj_id in subjects:
            if not args.no_save and (seed, subj_id) in existing_pairs:
                print(f"Skipping Seed {seed} | Subj {subj_id:02d} (already computed)")
                continue
                
            data = extract_subject_data(config.DATA_PATH, subj_id)
            if data is None: 
                continue
            
            X_sub, y_sub, char_ids_sub, codes_sub, runs_sub, dbs_sub = data
            db_id = int(dbs_sub[0]) if len(dbs_sub) > 0 else config.SUBJECT_METADATA[subj_id]['dataset']
            condition = config.SUBJECT_METADATA[subj_id]['condition']
            
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
            if len(calib_chars) < 2:
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
            
            # Standard shuffling dataloader for training, fast dataloader for val/test
            train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = FastTensorDataLoader(va_ds, batch_size=batch_size, shuffle=False, device=config.DEVICE)
            test_loader = FastTensorDataLoader(te_ds, batch_size=batch_size, shuffle=False, device=config.DEVICE)
            
            model = P3CNET(num_channels=8, time_samples=128).to(config.DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
            
            class_weights = torch.tensor([1.0, 7.0], dtype=torch.float32).to(config.DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            best_val_loss = float('inf')
            best_weights = None
            patience = 0
            
            # Train and validate loop
            for epoch in range(1, epochs + 1):
                model.train()
                for batch in train_loader:
                    data, target = batch[0].to(config.DEVICE), batch[1].to(config.DEVICE)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    # Strict Max-Norm constraints on conv1 (max 1.0) and fc (max 0.25)
                    with torch.no_grad():
                        norm1 = model.conv1.weight.norm(2, dim=(1, 2, 3), keepdim=True)
                        model.conv1.weight.mul_(torch.clamp(norm1, max=1.0) / (norm1 + 1e-8))
                        norm_fc = model.fc.weight.norm(2, dim=1, keepdim=True)
                        model.fc.weight.mul_(torch.clamp(norm_fc, max=0.25) / (norm_fc + 1e-8))
                
                # Check validation loss
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, target, _, _, _ in val_loader:
                        data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                        val_loss += criterion(model(data), target).item()
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                if patience > patience_limit:
                    break
            
            if best_weights is not None:
                model.load_state_dict(best_weights)
                
            obj_acc, obj_f1, itr_df = evaluate_metrics(model, test_loader, config.DEVICE, model_type="softmax")
            print(f"Seed {seed} | Subj {subj_id:02d} (DS {db_id}, {condition[:4]}) -> Acc: {obj_acc:.2f}% | F1: {obj_f1:.4f}")
            
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
                
            del model, optimizer
            torch.cuda.empty_cache()
            malloc_trim()
            
    # Export results to CSV
    df_subjects = pd.DataFrame(all_run_records)
    df_itr = pd.DataFrame(itr_run_records)
    
    if df_subjects.empty:
        print("\nCRITICAL ERROR: No records collected.")
    else:
        if not args.no_save:
            df_subjects.to_csv(os.path.join(RESULT_DIR, "subject_results.csv"), index=False)
            df_itr.to_csv(os.path.join(RESULT_DIR, "itr_results.csv"), index=False)
            print(f"\nSaved CSV results to: {RESULT_DIR}")
        else:
            print("\n[Dry Run] Results were NOT saved to disk.")
        
        # Display Final Aggregate Statistics
        print("\n" + "="*75)
        print(" P3CNET OVERALL STATISTICS (across all seeds & subjects)")
        print("="*75)
        
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
            
        acc_m, acc_s, acc_sem, acc_med, acc_iqr, acc_l, acc_h = compute_stats(df_subjects['Obj Acc (%)'])
        f1_m, f1_s, f1_sem, f1_med, f1_iqr, f1_l, f1_h = compute_stats(df_subjects['Macro F1'])
        
        print(f"Object Accuracy: {acc_m:.2f}% ± {acc_s:.2f}% (SEM: {acc_sem:.2f}, Median: {acc_med:.2f}%, IQR: {acc_iqr:.2f}, 95% CI: [{acc_l:.2f}%, {acc_h:.2f}%])")
        print(f"Macro F1 Score:  {f1_m:.4f} ± {f1_s:.4f} (SEM: {f1_sem:.4f}, Median: {f1_med:.4f}, IQR: {f1_iqr:.4f}, 95% CI: [{f1_l:.4f}, {f1_h:.4f}])")
        print("="*75)

if __name__ == "__main__":
    main()
