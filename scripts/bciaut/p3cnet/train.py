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
from sklearn.metrics import f1_score, accuracy_score

# Add parent folder scripts/bciaut/ to import shared resources
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from dataset import BCIAUTDataset, FoldDataset, FastTensorDataLoader
from model import P3CNET

# Setup specific target directory for P3CNET baseline results
RESULT_DIR = os.path.join(config.ROOT_DIR, "results/bciaut/p3cnet")
os.makedirs(RESULT_DIR, exist_ok=True)

# Default seeds for comparison
P3CNET_SEEDS = [42, 1234, 777]

def set_seed(seed):
    """
    Sets random seeds for reproducibility.
    """
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
    """
    Cleans up memory to prevent leakage on Linux.
    """
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def evaluate_metrics(model, loader, device, max_seq=10, n_classes=8):
    """
    Evaluates metrics (Overall accuracy, Macro F1, Session-wise scores, and ITR) for P3CNET.
    Queries model outputs directly since the model forward pass ends with a Softmax layer.
    """
    model.eval()
    block_probs = defaultdict(lambda: defaultdict(list))
    block_targets = {}
    block_sessions = {}
    
    with torch.no_grad():
        for batch in loader:
            data = batch[0].to(device)
            target = batch[1].cpu().numpy().ravel()
            group_ids = batch[3].cpu().numpy().ravel()
            sess_ids = batch[4].cpu().numpy().ravel()
            
            # Model output has shape [batch, 2] and already incorporates softmax
            probs = model(data)[:, 1].cpu().numpy()
            
            for prob, tgt, gid, sess in zip(probs, target, group_ids, sess_ids):
                gid = int(gid)
                block_id, object_id = gid // 100, gid % 100
                block_probs[block_id][object_id].append(prob)
                block_sessions[block_id] = int(sess)
                if int(tgt) == 1:
                    block_targets[block_id] = object_id

    session_preds = defaultdict(list)
    session_trues = defaultdict(list)
    y_true, y_pred = [], []
    
    for block_id, true_obj in block_targets.items():
        sess = block_sessions[block_id]
        avg_probs = {obj: np.mean(p_list) for obj, p_list in block_probs[block_id].items()}
        pred_obj = max(avg_probs, key=avg_probs.get)
        
        y_true.append(true_obj)
        y_pred.append(pred_obj)
        session_trues[sess].append(true_obj)
        session_preds[sess].append(pred_obj)

    overall_acc = accuracy_score(y_true, y_pred) * 100.0
    overall_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    session_metrics = {}
    for sess in session_trues:
        s_acc = accuracy_score(session_trues[sess], session_preds[sess]) * 100.0
        s_f1 = f1_score(session_trues[sess], session_preds[sess], average='macro', zero_division=0)
        session_metrics[sess] = {'Accuracy (%)': s_acc, 'Macro F1': s_f1}

    itr_results = []
    for seq_limit in range(1, max_seq + 1):
        correct, total = 0, len(block_targets)
        for block_id, true_obj in block_targets.items():
            avg_probs = {obj: np.mean(p_list[:seq_limit]) for obj, p_list in block_probs[block_id].items() if len(p_list) > 0}
            if avg_probs and max(avg_probs, key=avg_probs.get) == true_obj:
                correct += 1
        acc = correct / total if total > 0 else 0.0
        
        if acc <= (1.0 / n_classes):
            itr = 0.0
        elif acc >= 1.0:
            itr = math.log2(n_classes)
        else:
            itr = (math.log2(n_classes) + acc * math.log2(acc) + (1.0 - acc) * math.log2((1.0 - acc) / (n_classes - 1)))
        itr_results.append({'Sequences': seq_limit, 'Accuracy (%)': acc * 100.0, 'Bits/Selection': itr})

    return overall_acc, overall_f1, session_metrics, pd.DataFrame(itr_results)

def parse_args():
    parser = argparse.ArgumentParser(description="P3CNET Baseline Training & Evaluation Pipeline")
    parser.add_argument("--subjects", type=int, nargs="+", default=config.SUBJECT_IDS,
                        help="List of subjects to run (1-15)")
    parser.add_argument("--seeds", type=int, nargs="+", default=P3CNET_SEEDS,
                        help="Random seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Max training epochs per subject/seed")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training/inference")
    parser.add_argument("--patience", type=int, default=45,
                        help="Early stopping patience epochs based on validation loss")
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
    session_run_records = []
    itr_run_records = []
    
    # Load training and testing datasets across all subjects
    print("Loading datasets across all subjects...")
    train_dataset = BCIAUTDataset(config.DATA_PATH, subjects, range(1, 8), phase="Train")
    test_dataset = BCIAUTDataset(config.DATA_PATH, subjects, range(4, 8), phase="Test",
                                  scalers=train_dataset.scalers)
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError("Loaded dataset is empty.")
        
    for seed in seeds:
        print(f"\n================ P3CNET Running Seed: {seed} ================")
        set_seed(seed)
        
        for subj_id in subjects:
            mask_tr = (train_dataset.subj_ids == (subj_id - 1))
            mask_te = (test_dataset.subj_ids == (subj_id - 1))
            
            if not mask_tr.any() or not mask_te.any():
                continue
                
            X_tr, y_tr = train_dataset.X[mask_tr], train_dataset.y[mask_tr]
            s_tr, g_tr, ses_tr = train_dataset.subj_ids[mask_tr], train_dataset.group_ids[mask_tr], train_dataset.session_ids[mask_tr]
            
            X_te, y_te = test_dataset.X[mask_te], test_dataset.y[mask_te]
            s_te, g_te, ses_te = test_dataset.subj_ids[mask_te], test_dataset.group_ids[mask_te], test_dataset.session_ids[mask_te]
            
            # Extract stratified indices for train/validation splits
            pos_idx = torch.where(y_tr == 1)[0].numpy()
            neg_idx = torch.where(y_tr == 0)[0].numpy()
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
            
            n_pos_val = max(1, int(0.2 * len(pos_idx)))
            n_neg_val = max(1, int(0.2 * len(neg_idx)))
            
            val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
            train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])
            
            # Setup tensor datasets
            tr_ds = FoldDataset(X_tr[train_idx], y_tr[train_idx], s_tr[train_idx], g_tr[train_idx], ses_tr[train_idx])
            va_ds = FoldDataset(X_tr[val_idx], y_tr[val_idx], s_tr[val_idx], g_tr[val_idx], ses_tr[val_idx])
            te_ds = FoldDataset(X_te, y_te, s_te, g_te, ses_te)
            
            # Setup dataloaders (Standard shuffling dataloader for training, fast dataloader for val/test)
            train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = FastTensorDataLoader(va_ds, batch_size=batch_size, shuffle=False, device=config.DEVICE)
            test_loader = FastTensorDataLoader(te_ds, batch_size=batch_size, shuffle=False, device=config.DEVICE)
            
            # Instantiate model, optimizer, loss function with 7:1 class weights
            model = P3CNET(num_channels=8, time_samples=140).to(config.DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
            
            class_weights = torch.tensor([1.0, 7.0], dtype=torch.float32).to(config.DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            best_val_loss = float('inf')
            best_weights = None
            patience = 0
            
            # Train and validate loop
            for epoch in range(1, epochs + 1):
                model.train()
                for data, target, _, _, _ in train_loader:
                    data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                    optimizer.zero_grad()
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
                
                # Check for improvements
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience > patience_limit:
                    break
            
            # Load best checkpoint weights
            if best_weights is not None:
                model.load_state_dict({k: v.to(config.DEVICE) for k, v in best_weights.items()})
                
            # Evaluate on independent test set
            obj_acc, obj_f1, sess_metrics, itr_df = evaluate_metrics(model, test_loader, config.DEVICE)
            print(f"Seed {seed} | Subject {subj_id:02d} -> Acc: {obj_acc:.2f}% | F1: {obj_f1:.4f}")
            
            # Record scores (No model saving or saliency calculations)
            all_run_records.append({
                'Seed': seed,
                'Subject': f"SBJ{subj_id:02d}",
                'Obj Acc (%)': obj_acc,
                'Macro F1': obj_f1
            })
            
            for sess, metrics in sess_metrics.items():
                session_run_records.append({
                    'Seed': seed,
                    'Subject': f"SBJ{subj_id:02d}",
                    'Session': sess,
                    'Obj Acc (%)': metrics['Accuracy (%)'],
                    'Macro F1': metrics['Macro F1']
                })
                
            for _, row in itr_df.iterrows():
                itr_run_records.append({
                    'Seed': seed,
                    'Subject': f"SBJ{subj_id:02d}",
                    'Sequences': row['Sequences'],
                    'Accuracy (%)': row['Accuracy (%)'],
                    'Bits/Selection': row['Bits/Selection']
                })
                
            # Cleanup GPU memory
            del model, optimizer
            torch.cuda.empty_cache()
            malloc_trim()
            
    # Export results to CSV
    df_subjects = pd.DataFrame(all_run_records)
    df_sessions = pd.DataFrame(session_run_records)
    df_itr = pd.DataFrame(itr_run_records)
    
    if df_subjects.empty:
        print("\nCRITICAL ERROR: No records collected.")
    else:
        if not args.no_save:
            df_subjects.to_csv(os.path.join(RESULT_DIR, "subject_results.csv"), index=False)
            df_sessions.to_csv(os.path.join(RESULT_DIR, "session_results.csv"), index=False)
            df_itr.to_csv(os.path.join(RESULT_DIR, "itr_results.csv"), index=False)
            print(f"\nSaved CSV results to: {RESULT_DIR}")
        else:
            print("\n[Dry Run] Results were NOT saved to disk.")
        
        # Display Final Aggregate Statistics
        print("\n" + "="*70)
        print(" P3CNET FINAL AGGREGATE STATISTICS (ACROSS ALL SEEDS & SUBJECTS)")
        print("="*70)
        
        def compute_stats(series):
            mean = series.mean()
            std = series.std()
            sem = std / math.sqrt(len(series)) if len(series) > 0 else 0
            median = series.median()
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            ci95_low, ci95_high = np.percentile(series, [2.5, 97.5])
            return mean, std, sem, median, (q75 - q25), ci95_low, ci95_high
            
        acc_mean, acc_std, acc_sem, acc_med, acc_iqr, acc_ci_l, acc_ci_h = compute_stats(df_subjects['Obj Acc (%)'])
        f1_mean, f1_std, f1_sem, f1_med, f1_iqr, f1_ci_l, f1_ci_h = compute_stats(df_subjects['Macro F1'])
        
        print(f"Object Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}% (SEM: {acc_sem:.2f}, Median: {acc_med:.2f}%, IQR: {acc_iqr:.2f}, 95% CI: [{acc_ci_l:.2f}%, {acc_ci_h:.2f}%])")
        print(f"Macro F1 Score:  {f1_mean:.4f} ± {f1_std:.4f} (SEM: {f1_sem:.4f}, Median: {f1_med:.4f}, IQR: {f1_iqr:.4f}, 95% CI: [{f1_ci_l:.4f}, {f1_ci_h:.4f}])")
        print("="*70)

if __name__ == "__main__":
    main()
