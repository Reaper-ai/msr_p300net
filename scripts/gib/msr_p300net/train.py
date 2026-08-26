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

# Add parent scripts/gib directory to python path for shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from dataset import extract_subject_data, SimpleTensorDataset, FastTensorDataLoader
from evaluation import evaluate_metrics

# Local model imports
from model import MSR_P300NET, apply_max_norm

# Setup path configuration specific to MSR_P300NET
RESULT_DIR = os.path.join(config.RESULT_DIR, "msr_p300net")
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

class SafeEEGAugmenter(nn.Module):
    """
    Data Augmentation for EEG signals.
    """
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

class SmoothBalancedFocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing for Handling Binary Imbalances.
    Original baseline parameters: gamma=1.36, pos_weight=5.75, smoothing=0.05.
    """
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

def train_subject_epoch(model, loader, optimizer, scheduler, criterion, amp_scaler, augmenter):
    model.train()
    augmenter.train()
    running_loss = 0.0
    for batch in loader:
        data = augmenter(batch[0])
        target = batch[1].float().unsqueeze(1)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=config.DEVICE.type):
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
            with torch.amp.autocast(device_type=config.DEVICE.type):
                output = model(data)
                loss = criterion(output, target)
            val_loss += loss.item()
    return val_loss / max(1, len(loader))

def parse_args():
    parser = argparse.ArgumentParser(description="MSR_P300NET Training on GIB Dataset")
    parser.add_argument("--subjects", type=int, nargs="+", default=config.SUBJECT_IDS,
                        help="List of subjects to run (0-72)")
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS,
                        help="Seeds to run")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save models or result files (for testing/debugging)")
    return parser.parse_args()

def main():
    args = parse_args()
    subjects = args.subjects
    seeds = args.seeds
    epochs = args.epochs
    batch_size = args.batch_size
    patience_limit = args.patience
    min_delta = 1e-4
    
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
                
    augmenter = SafeEEGAugmenter(p=0.5, noise_std=0.05).to(config.DEVICE)
    
    # Load existing interpretability results if they exist to avoid overwriting other subjects/seeds
    import json
    json_path = os.path.join(RESULT_DIR, "interpretability_results.json")
    interpretability_results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                interpretability_results = json.load(f)
            print(f"Loaded existing interpretability results from: {json_path}")
        except Exception as e:
            print(f"Warning: Could not load existing JSON file: {e}")
            
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
            
            train_loader = FastTensorDataLoader(tr_ds, batch_size=batch_size, shuffle=True, device=config.DEVICE)
            val_loader = FastTensorDataLoader(va_ds, batch_size=batch_size, shuffle=False, device=config.DEVICE)
            test_loader = FastTensorDataLoader(te_ds, batch_size=batch_size, shuffle=False, device=config.DEVICE)
            
            model = MSR_P300NET(n_channels=8, n_samples=128).to(config.DEVICE)
            
            # Original values: lr=2e-3, weight_decay=1.5e-3, max_lr=4e-3
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0015)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=5e-3, epochs=epochs, 
                steps_per_epoch=max(1, len(train_loader)), pct_start=0.2
            )
            # Original values: gamma=1.36, pos_weight=5.75, smoothing=0.05
            criterion = SmoothBalancedFocalLoss(gamma=2, pos_weight=5, smoothing=0.05).to(config.DEVICE)
            scaler = torch.amp.GradScaler(device=config.DEVICE.type)
            
            best_val_loss = float('inf')
            best_weights = None
            patience = 0
            
            for epoch in range(1, epochs + 1):
                train_loss = train_subject_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, augmenter)
                val_loss = evaluate_val_loss(model, val_loader, criterion)
                
                if val_loss < (best_val_loss - min_delta):
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                    
                if patience >= patience_limit:
                    break
                    
            if best_weights is not None:
                model.load_state_dict(best_weights)
                # Save best model to models/gib/ folder
                if not args.no_save:
                    model_name = f"subj_{subj_id:02d}_seed_{seed}_best.pth"
                    model_save_path = os.path.join(config.MODEL_DIR, model_name)
                    torch.save(best_weights, model_save_path)
                    print(f"  → Saved best model to: {model_save_path}")
                else:
                    print("  → [Dry Run] Model checkpoint was NOT saved to disk.")
                
            obj_acc, obj_f1, itr_df = evaluate_metrics(model, test_loader, config.DEVICE, model_type="logit")
            print(f"Seed {seed} | Subj {subj_id:02d} (DS {db_id}, {condition[:4]}) -> Acc: {obj_acc:6.2f}% | F1: {obj_f1:.4f}")
            
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
                
            # --- EXTRACT INTERPRETABILITY DATA ---
            print("  → Extracting spatial weights, temporal attention, and gradient saliency...")
            
            # 1. Spatial Weights (from block1.conv_s.weight)
            spatial_w = model.block1.conv_s.weight.detach().abs().mean(dim=(0, 1, 3)).cpu().numpy().tolist()
            
            # 2. Temporal Attention Weights (MSR_P300NET has no attention layer, return empty list)
            temporal_att = []
            
            # 3. Input Gradient Saliency (averaged over target trials)
            target_test_mask = (y_te == 1)
            if target_test_mask.any():
                model.eval()
                X_te_targets = torch.tensor(X_te[target_test_mask]).unsqueeze(1).clone().detach().to(config.DEVICE)
                X_te_targets.requires_grad = True
                with torch.enable_grad():
                    with torch.amp.autocast(device_type=config.DEVICE.type):
                        logits = model(X_te_targets)
                    logits.backward(torch.ones_like(logits))
                grad_saliency = X_te_targets.grad.detach().abs().mean(dim=0).cpu().numpy().tolist()
            else:
                grad_saliency = []
                
            # Record interpretability results
            subj_key = f"SBJ{subj_id:02d}"
            seed_key = f"seed_{seed}"
            if subj_key not in interpretability_results:
                interpretability_results[subj_key] = {}
            interpretability_results[subj_key][seed_key] = {
                "spatial_weights": spatial_w,
                "temporal_attention": temporal_att,
                "gradient_saliency": grad_saliency
            }
            
            del model, optimizer, scheduler, criterion, scaler
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
            
            # Save interpretability results to JSON
            with open(json_path, "w") as f:
                json.dump(interpretability_results, f, indent=4)
                
            print(f"\nSaved CSV and JSON interpretability results to: {RESULT_DIR}")
        else:
            print("\n[Dry Run] Results and interpretability maps were NOT saved to disk.")
        
        # Display Final Aggregate Statistics
        print("\n" + "="*75)
        print(" MSR_P300NET OVERALL STATISTICS (across all seeds & subjects)")
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
