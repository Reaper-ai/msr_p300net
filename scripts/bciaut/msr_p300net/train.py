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
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# Add the parent script directory to Python path to allow config, dataset, and evaluation imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from dataset import BCIAUTDataset, FoldDataset, FastTensorDataLoader, get_augmented_subject_subset
from evaluation import evaluate_comprehensive_metrics

# Import model-specific components locally
from model import MSR_P300NET
from training import SmoothBalancedFocalLoss, train_subject_epoch, evaluate_val_loss

def set_seed(seed):
    """
    Sets random seeds for reproducibility across random, numpy, and PyTorch.
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
    Trims memory using libc call (Linux only) to avoid memory accumulation.
    """
    try:
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="BCIAUT-P300 Dataset Training & Evaluation Pipeline")
    parser.add_argument("--subjects", type=int, nargs="+", default=config.SUBJECT_IDS,
                        help="List of subjects to run (1-15)")
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS,
                        help="Random seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Max training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for training/inference")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience based on validation loss")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save models or result files (for testing/debugging)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Overwrite configuration using CLI arguments
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
    existing_pairs = set()
    csv_path = os.path.join(config.RESULT_DIR, "subject_results.csv")
    sess_csv_path = os.path.join(config.RESULT_DIR, "session_results.csv")
    itr_csv_path = os.path.join(config.RESULT_DIR, "itr_results.csv")
    
    # Load existing interpretability results if they exist to avoid overwriting other subjects/seeds
    import json
    json_path = os.path.join(config.RESULT_DIR, "interpretability_results.json")
    interpretability_results = {}
    
    if not args.no_save:
        if os.path.exists(csv_path):
            try:
                df_exist = pd.read_csv(csv_path)
                all_run_records = df_exist.to_dict('records')
                for r in all_run_records:
                    # Note: Subject in BCIAUT CSV is formatted as "SBJxx", e.g. SBJ01
                    if 'Seed' in r and 'Subject' in r:
                        subj_str = r['Subject']
                        subj_num = int(subj_str.replace("SBJ", ""))
                        existing_pairs.add((int(r['Seed']), subj_num))
                print(f"Loaded {len(existing_pairs)} completed subject-seed pairs from: {csv_path}")
            except Exception as e:
                print(f"Warning: Could not load existing CSV: {e}")
        if os.path.exists(sess_csv_path):
            try:
                session_run_records = pd.read_csv(sess_csv_path).to_dict('records')
            except Exception:
                pass
        if os.path.exists(itr_csv_path):
            try:
                itr_run_records = pd.read_csv(itr_csv_path).to_dict('records')
            except Exception:
                pass
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    interpretability_results = json.load(f)
                print(f"Loaded existing interpretability results from: {json_path}")
            except Exception as e:
                print(f"Warning: Could not load existing JSON file: {e}")
    
    for seed in seeds:
        print(f"\n{'='*75}\n SEED: {seed}\n{'='*75}")
        set_seed(seed)
        
        # Load train and test datasets across all specified subjects
        print("Loading datasets...")
        train_dataset = BCIAUTDataset(config.DATA_PATH, subjects, range(1, 8), phase="Train")
        test_dataset = BCIAUTDataset(config.DATA_PATH, subjects, range(1, 8), phase="Test",
                                      scalers=train_dataset.scalers)
                                      
        print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            raise RuntimeError("Loaded dataset is empty. Check DATA_PATH and session layout.")
            
        for subj_id in subjects:
            if not args.no_save and (seed, subj_id) in existing_pairs:
                print(f"Skipping Seed {seed} | Subj {subj_id:02d} (already computed)")
                continue
                
            print(f"\n--- Subject SBJ{subj_id:02d} ---")
            
            mask_tr = (train_dataset.subj_ids == (subj_id - 1))
            mask_te = (test_dataset.subj_ids == (subj_id - 1))
            
            # Make sure subject has both training and testing data
            if not mask_tr.any() or not mask_te.any():
                print(f"  No data for subject {subj_id}, skipping.")
                continue
                
            # Get training pool for the subject
            X_pool = train_dataset.X[mask_tr]
            y_pool = train_dataset.y[mask_tr]
            s_pool = train_dataset.subj_ids[mask_tr]
            g_pool = train_dataset.group_ids[mask_tr]
            ses_pool = train_dataset.session_ids[mask_tr]
            
            # Identify unique blocks for splitting (so all samples in a block stay in the same split)
            pool_blocks = (g_pool.cpu().numpy() // 100)
            unique_blocks = np.unique(pool_blocks)
            
            # 5-fold Cross-Validation setup
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            
            best_overall_val_loss = float('inf')
            best_fold_idx = -1
            best_fold_weights = None
            best_fold_val_indices = None
            
            # Track validation metrics for information purposes
            for fold_idx, (train_blk_idx, val_blk_idx) in enumerate(kf.split(unique_blocks)):
                train_blocks = unique_blocks[train_blk_idx]
                val_blocks = unique_blocks[val_blk_idx]
                
                tr_indices = np.isin(pool_blocks, train_blocks)
                va_indices = np.isin(pool_blocks, val_blocks)
                
                # Split fold dataset
                fold_tr_ds = FoldDataset(X_pool[tr_indices], y_pool[tr_indices],
                                         s_pool[tr_indices], g_pool[tr_indices], ses_pool[tr_indices])
                fold_va_ds = FoldDataset(X_pool[va_indices], y_pool[va_indices],
                                         s_pool[va_indices], g_pool[va_indices], ses_pool[va_indices])
                
                # Setup Dataloaders for this fold
                train_loader = get_augmented_subject_subset(fold_tr_ds, batch_size=batch_size)
                val_loader = FastTensorDataLoader(fold_va_ds, batch_size=batch_size, shuffle=False, device=config.DEVICE)
                
                # Build network, loss criterion, optimizer & scheduler
                model = MSR_P300NET(num_channels=8, input_time_samples=140).to(config.DEVICE)
                optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, epochs=epochs,
                                                          steps_per_epoch=len(train_loader), pct_start=0.3)
                criterion = SmoothBalancedFocalLoss(gamma=2.5, pos_weight=config.STATIC_POS_WEIGHT, smoothing=0.05).to(config.DEVICE)
                scaler = torch.amp.GradScaler(device=config.DEVICE.type)
                
                best_val_loss = float('inf')
                best_weights = None
                patience = 0
                
                # Training Loop
                for epoch in range(1, epochs + 1):
                    train_subject_epoch(model, train_loader, optimizer, scheduler, criterion, scaler)
                    val_loss = evaluate_val_loss(model, val_loader, criterion)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = copy.deepcopy(model.state_dict())
                        patience = 0
                    else:
                        patience += 1
                        
                    if patience >= patience_limit:
                        break
                        
                # Check if this fold performed the best on validation loss
                if best_val_loss < best_overall_val_loss:
                    best_overall_val_loss = best_val_loss
                    best_fold_idx = fold_idx
                    best_fold_weights = best_weights
                    best_fold_val_indices = va_indices
                    
                print(f"    Fold {fold_idx+1}/5 - Best Val Loss: {best_val_loss:.4f}")
                
                # Free GPU memory
                del model, optimizer, scheduler, criterion, scaler
                torch.cuda.empty_cache()
                malloc_trim()
                
            print(f"  → Subject SBJ{subj_id:02d} Best Fold: {best_fold_idx+1}/5 (Val Loss: {best_overall_val_loss:.4f})")
            
            # Load the best overall model weights from the best cross-validation fold
            best_model = MSR_P300NET(num_channels=8, input_time_samples=140).to(config.DEVICE)
            best_model.load_state_dict(best_fold_weights)
            
            # Save the best model to disk
            if not args.no_save:
                model_name = f"subj_{subj_id:02d}_seed_{seed}_best.pth"
                model_save_path = os.path.join(config.MODEL_DIR, model_name)
                torch.save(best_fold_weights, model_save_path)
                print(f"  → Saved best model to: {model_save_path}")
            else:
                print("  → [Dry Run] Model checkpoint was NOT saved to disk.")
            
            # Recreate validation loader for the best fold to perform logistic calibration
            best_fold_va_ds = FoldDataset(X_pool[best_fold_val_indices], y_pool[best_fold_val_indices],
                                          s_pool[best_fold_val_indices], g_pool[best_fold_val_indices], ses_pool[best_fold_val_indices])
            best_fold_va_loader = FastTensorDataLoader(best_fold_va_ds, batch_size=batch_size, shuffle=False, device=config.DEVICE)
            
            # Get validation logits for calibration
            best_model.eval()
            val_logits, val_y = [], []
            with torch.no_grad():
                for batch in best_fold_va_loader:
                    data_v, target_v = batch[0], batch[1].cpu().numpy()
                    with torch.amp.autocast(device_type=config.DEVICE.type):
                        logits = best_model(data_v).squeeze(1).cpu().numpy()
                    val_logits.append(logits)
                    val_y.append(target_v)
            val_logits_all = np.concatenate(val_logits).reshape(-1, 1)
            val_y_all = np.concatenate(val_y)
            
            # Fit Logistic Regression calibrator
            calibrator = LogisticRegression()
            if len(np.unique(val_y_all)) > 1:
                calibrator.fit(val_logits_all, val_y_all)
            else:
                calibrator.coef_ = np.array([[1.0]])
                calibrator.intercept_ = np.array([0.0])
                calibrator.classes_ = np.array([0, 1])
                
            # Load and run evaluation on the independent test dataset
            sub_te_ds = FoldDataset(test_dataset.X[mask_te], test_dataset.y[mask_te],
                                    test_dataset.subj_ids[mask_te], test_dataset.group_ids[mask_te],
                                    test_dataset.session_ids[mask_te])
            test_loader = FastTensorDataLoader(sub_te_ds, batch_size=batch_size, shuffle=False, device=config.DEVICE)
            
            acc, f1, sess_metrics, itr_df = evaluate_comprehensive_metrics(best_model, calibrator, test_loader)
            
            # Record scores
            all_run_records.append({
                'Method': '5-Fold CV Best Fold', 
                'Seed': seed, 
                'Subject': f"SBJ{subj_id:02d}",
                'Obj Acc (%)': acc, 
                'Macro F1': f1
            })
            
            for sess, metrics in sess_metrics.items():
                session_run_records.append({
                    'Method': '5-Fold CV Best Fold', 
                    'Seed': seed, 
                    'Subject': f"SBJ{subj_id:02d}",
                    'Session': sess, 
                    'Obj Acc (%)': metrics['Accuracy (%)'],
                    'Macro F1': metrics['Macro F1']
                })
                
            for _, row in itr_df.iterrows():
                itr_run_records.append({
                    'Method': '5-Fold CV Best Fold', 
                    'Seed': seed, 
                    'Subject': f"SBJ{subj_id:02d}",
                    'Sequences': row['Sequences'], 
                    'Accuracy (%)': row['Accuracy (%)'],
                    'Bits/Selection': row['Bits/Selection']
                })
                
            print(f"  → Test Results - Acc: {acc:.2f}%  F1: {f1:.4f}")
            
            # --- EXTRACT INTERPRETABILITY DATA ---
            print("  → Extracting spatial weights, temporal attention, and gradient saliency...")
            # 1. Spatial Weights (shape [8])
            spatial_w = best_model.block1.conv_s.weight.detach().abs().mean(dim=(0, 1, 3)).cpu().numpy().tolist()
            
            # 2. Temporal Attention Weights (MSR_P300NET has no attention layer, return empty list)
            temporal_att = []
            
            # 3. Input Gradient Saliency (shape [8, 140])
            target_test_mask = (test_dataset.y[mask_te] == 1)
            if target_test_mask.any():
                best_model.eval()
                X_te_targets = test_dataset.X[mask_te][target_test_mask].clone().detach().to(config.DEVICE)
                X_te_targets.requires_grad = True
                with torch.enable_grad():
                    with torch.amp.autocast(device_type=config.DEVICE.type):
                        logits = best_model(X_te_targets)
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

            # Free memory
            del best_model, calibrator, test_loader
            torch.cuda.empty_cache()
            malloc_trim()
            
    # Save all results to CSV files
    df_subjects = pd.DataFrame(all_run_records)
    df_sessions = pd.DataFrame(session_run_records)
    df_itr = pd.DataFrame(itr_run_records)
    
    if df_subjects.empty:
        print("\nCRITICAL ERROR: No subjects were processed or no records saved.")
    else:
        if not args.no_save:
            df_subjects.to_csv(os.path.join(config.RESULT_DIR, "subject_results.csv"), index=False)
            df_sessions.to_csv(os.path.join(config.RESULT_DIR, "session_results.csv"), index=False)
            df_itr.to_csv(os.path.join(config.RESULT_DIR, "itr_results.csv"), index=False)
            
            # Save interpretability results to JSON
            with open(json_path, "w") as f:
                json.dump(interpretability_results, f, indent=4)
            print(f"\nSaved CSV and JSON interpretability results to: {config.RESULT_DIR}")
        else:
            print("\n[Dry Run] Results and interpretability maps were NOT saved to disk.")
        
        # Display Final Aggregate Statistics
        print("\n" + "="*75)
        print(" FINAL AGGREGATE STATISTICS (across all seeds & subjects)")
        print("="*75)
        
        def compute_stats(series):
            mean, std = series.mean(), series.std()
            sem = std / math.sqrt(len(series)) if len(series) > 0 else 0
            median = series.median()
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            ci_low, ci_high = np.percentile(series, [2.5, 97.5])
            return mean, std, sem, median, (q75-q25), ci_low, ci_high
            
        acc_m, acc_s, acc_se, acc_med, acc_iqr, acc_ci_l, acc_ci_h = compute_stats(df_subjects['Obj Acc (%)'])
        print(f"Object Accuracy: {acc_m:.2f}% ± {acc_s:.2f}% (SEM: {acc_se:.2f}, Median: {acc_med:.2f}%, IQR: {acc_iqr:.2f}, 95% CI: [{acc_ci_l:.2f}%, {acc_ci_h:.2f}%])")
        f1_m, f1_s, f1_se, f1_med, f1_iqr, f1_ci_l, f1_ci_h = compute_stats(df_subjects['Macro F1'])
        print(f"Macro F1 Score:  {f1_m:.4f} ± {f1_s:.4f} (SEM: {f1_se:.4f}, Median: {f1_med:.4f}, IQR: {f1_iqr:.4f}, 95% CI: [{f1_ci_l:.4f}, {f1_ci_h:.4f}])")

if __name__ == "__main__":
    main()
