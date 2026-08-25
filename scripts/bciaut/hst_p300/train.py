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

# Add the parent script directory to Python path to allow config, dataset, and evaluation imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from dataset import BCIAUTDataset, FoldDataset, FastTensorDataLoader

# Updated evaluate functions
from evaluation import evaluate_comprehensive_metrics, evaluate_comprehensive_metrics_from_logits

# Import model-specific components locally
from model import HST_P300Net, SafeEEGAugmenter
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
                        help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for training/inference")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience based on validation loss")
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
    
    # Check if it is a test run (e.g. debug run with low epoch count)
    is_test_run = (epochs < 10)
    suffix = "_test" if is_test_run else ""
    
    # File paths for consolidated results (2 sets of 3 files)
    sub_best_path = os.path.join(config.RESULT_DIR, f"best_fold_subject_results{suffix}.csv")
    ses_best_path = os.path.join(config.RESULT_DIR, f"best_fold_session_results{suffix}.csv")
    itr_best_path = os.path.join(config.RESULT_DIR, f"best_fold_itr_results{suffix}.csv")
    sub_ens_path = os.path.join(config.RESULT_DIR, f"ensemble_subject_results{suffix}.csv")
    ses_ens_path = os.path.join(config.RESULT_DIR, f"ensemble_session_results{suffix}.csv")
    itr_ens_path = os.path.join(config.RESULT_DIR, f"ensemble_itr_results{suffix}.csv")
    
    # Load existing consolidated results if they exist to support resume functionality
    print("Checking for existing consolidated results to resume...")
    if os.path.exists(sub_best_path) and os.path.exists(sub_ens_path):
        try:
            df_sub_best = pd.read_csv(sub_best_path)
            df_sub_ens = pd.read_csv(sub_ens_path)
            all_run_records.extend(df_sub_best.to_dict('records'))
            all_run_records.extend(df_sub_ens.to_dict('records'))
            print("Loaded existing subject results.")
        except Exception as e:
            print(f"Warning: Could not load subject results: {e}")
            
    if os.path.exists(ses_best_path) and os.path.exists(ses_ens_path):
        try:
            df_ses_best = pd.read_csv(ses_best_path)
            df_ses_ens = pd.read_csv(ses_ens_path)
            session_run_records.extend(df_ses_best.to_dict('records'))
            session_run_records.extend(df_ses_ens.to_dict('records'))
        except: pass
        
    if os.path.exists(itr_best_path) and os.path.exists(itr_ens_path):
        try:
            df_itr_best = pd.read_csv(itr_best_path)
            df_itr_ens = pd.read_csv(itr_ens_path)
            itr_run_records.extend(df_itr_best.to_dict('records'))
            itr_run_records.extend(df_itr_ens.to_dict('records'))
        except: pass
            
    # Load existing interpretability results if they exist to avoid overwriting other subjects/seeds
    import json
    json_path = os.path.join(config.RESULT_DIR, "interpretability_results.json")
    interpretability_results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                interpretability_results = json.load(f)
            print(f"Loaded existing interpretability results from: {json_path}")
        except Exception as e:
            print(f"Warning: Could not load existing JSON file: {e}")
            
    # Load train and test datasets across all specified subjects once to save time and RAM
    print("Loading datasets...")
    train_dataset = BCIAUTDataset(config.DATA_PATH, subjects, range(1, 8), phase="Train")
    test_dataset = BCIAUTDataset(config.DATA_PATH, subjects, range(4, 8), phase="Test",
                                  scalers=train_dataset.scalers)
                                  
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError("Loaded dataset is empty. Check DATA_PATH and session layout.")
    
    for seed in seeds:
        print(f"\n{'='*75}\n SEED: {seed}\n{'='*75}")
        set_seed(seed)
        
        for subj_id in subjects:
            subj_key = f"SBJ{subj_id:02d}"
            seed_key = f"seed_{seed}"
            
            # Check if this subject & seed has already been computed in consolidated files
            completed_best = any(r['Subject'] == subj_key and r['Seed'] == seed and r['Method'] == 'Best Fold' for r in all_run_records)
            completed_ens = any(r['Subject'] == subj_key and r['Seed'] == seed and r['Method'] == 'Normal Ensemble' for r in all_run_records)
            if completed_best and completed_ens:
                print(f"\n--- Subject SBJ{subj_id:02d} (already computed for seed {seed}), skipping. ---")
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
            
            # Slice independent test set
            sub_te_ds = FoldDataset(test_dataset.X[mask_te], test_dataset.y[mask_te],
                                    test_dataset.subj_ids[mask_te], test_dataset.group_ids[mask_te],
                                    test_dataset.session_ids[mask_te])
            test_loader = FastTensorDataLoader(sub_te_ds, batch_size=batch_size, shuffle=False, device=config.DEVICE)
            
            test_y = sub_te_ds.y.cpu().numpy()
            test_g = sub_te_ds.group_ids.cpu().numpy()
            test_ses = sub_te_ds.session_ids.cpu().numpy()
            
            # 5-fold Cross-Validation setup
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            
            best_overall_val_loss = float('inf')
            best_fold_idx = -1
            best_fold_weights = None
            
            all_fold_test_logits = []
            val_losses = []
            
            # Track validation metrics across 5 folds
            for fold_idx, (train_blk_idx, val_blk_idx) in enumerate(kf.split(unique_blocks)):
                train_blocks = unique_blocks[train_blk_idx]
                val_blocks = unique_blocks[val_blk_idx]
                
                tr_indices = np.isin(pool_blocks, train_blocks)
                va_indices = np.isin(pool_blocks, val_blocks)
                
                # Split dataset for this fold
                fold_tr_ds = FoldDataset(X_pool[tr_indices], y_pool[tr_indices],
                                         s_pool[tr_indices], g_pool[tr_indices], ses_pool[tr_indices])
                fold_va_ds = FoldDataset(X_pool[va_indices], y_pool[va_indices],
                                         s_pool[va_indices], g_pool[va_indices], ses_pool[va_indices])
                
                # Setup Dataloaders
                train_loader = FastTensorDataLoader(fold_tr_ds, batch_size=batch_size, shuffle=True, device=config.DEVICE)
                val_loader = FastTensorDataLoader(fold_va_ds, batch_size=batch_size, shuffle=False, device=config.DEVICE)
                
                # Build network, loss criterion, optimizer & scheduler
                model = HST_P300Net(num_channels=8, input_time_samples=140).to(config.DEVICE)
                augmenter = SafeEEGAugmenter(p=0.5, noise_std=0.05).to(config.DEVICE)
                optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1.5e-3)
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, epochs=epochs,
                                                          steps_per_epoch=len(train_loader), pct_start=0.2)
                criterion = SmoothBalancedFocalLoss(gamma=1.36, pos_weight=5.75, smoothing=0.05).to(config.DEVICE)
                scaler = torch.amp.GradScaler(device=config.DEVICE.type)
                
                best_val_loss = float('inf')
                best_weights = None
                patience = 0
                
                # Training Loop
                for epoch in range(1, epochs + 1):
                    train_subject_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, augmenter=augmenter)
                    val_loss = evaluate_val_loss(model, val_loader, criterion)
                    
                    if val_loss < (best_val_loss - 1e-4):
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
                    
                val_losses.append(best_val_loss)
                print(f"    Fold {fold_idx+1}/5 - Best Val Loss: {best_val_loss:.4f}")
                
                # Run test set through this fold's best weights
                model.load_state_dict(best_weights)
                model.eval()
                fold_logits = []
                with torch.no_grad():
                    for batch in test_loader:
                        data_t = batch[0]
                        with torch.amp.autocast(device_type=config.DEVICE.type):
                            logits_t = model(data_t).squeeze(1).cpu().numpy()
                        fold_logits.append(logits_t)
                all_fold_test_logits.append(np.concatenate(fold_logits))
                
                # Free memory explicitly per fold
                del model, augmenter, optimizer, scheduler, criterion, scaler
                del train_loader, val_loader
                del fold_tr_ds, fold_va_ds
                if 'best_weights' in locals():
                    del best_weights
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                malloc_trim()
            
            print(f"  → Subject SBJ{subj_id:02d} Best Fold: {best_fold_idx+1}/5 (Val Loss: {best_overall_val_loss:.4f})")
            
            # Save the single overall best model weights to disk (only saving 1 instead of all 5 to save space)
            model_name = f"subj_{subj_id:02d}_seed_{seed}_best.pth"
            model_save_path = os.path.join(config.MODEL_DIR, model_name)
            torch.save(best_fold_weights, model_save_path)
            print(f"  → Saved best single fold model to: {model_save_path}")
            
            # --- EVALUATE ENSEMBLES ---
            all_fold_test_logits = np.array(all_fold_test_logits) # [5, num_test_trials]
            
            # 1. Normal (Simple Average) Ensemble
            normal_ensemble_logits = np.mean(all_fold_test_logits, axis=0)
            
            # 2. Best Single Fold
            best_fold_logits = all_fold_test_logits[best_fold_idx]
            
            # Evaluate and save results for both strategies
            methods_logits = {
                'Best Fold': best_fold_logits,
                'Normal Ensemble': normal_ensemble_logits
            }
            
            for method_name, logits in methods_logits.items():
                acc, f1, sess_metrics, itr_df = evaluate_comprehensive_metrics_from_logits(
                    logits, test_y, test_g, test_ses
                )
                
                rec = {
                    'Method': method_name, 
                    'Seed': seed, 
                    'Subject': f"SBJ{subj_id:02d}",
                    'Obj Acc (%)': acc, 
                    'Macro F1': f1
                }
                all_run_records.append(rec)
                
                for sess, metrics in sess_metrics.items():
                    session_run_records.append({
                        'Method': method_name, 
                        'Seed': seed, 
                        'Subject': f"SBJ{subj_id:02d}",
                        'Session': sess, 
                        'Obj Acc (%)': metrics['Accuracy (%)'],
                        'Macro F1': metrics['Macro F1']
                    })
                    
                for _, row in itr_df.iterrows():
                    itr_run_records.append({
                        'Method': method_name, 
                        'Seed': seed, 
                        'Subject': f"SBJ{subj_id:02d}",
                        'Sequences': row['Sequences'], 
                        'Accuracy (%)': row['Accuracy (%)'],
                        'Bits/Selection': row['Bits/Selection']
                    })
                
                print(f"  → {method_name} - Acc: {acc:.2f}%  F1: {f1:.4f}")
                
            # Save consolidated results immediately to results folder (overwriting the 6 files) to prevent data loss on crash
            df_subjects_temp = pd.DataFrame(all_run_records)
            df_sessions_temp = pd.DataFrame(session_run_records)
            df_itr_temp = pd.DataFrame(itr_run_records)
            
            # Save Best Fold
            df_subjects_temp[df_subjects_temp['Method'] == 'Best Fold'].to_csv(sub_best_path, index=False)
            df_sessions_temp[df_sessions_temp['Method'] == 'Best Fold'].to_csv(ses_best_path, index=False)
            df_itr_temp[df_itr_temp['Method'] == 'Best Fold'].to_csv(itr_best_path, index=False)
            
            # Save Normal Ensemble
            df_subjects_temp[df_subjects_temp['Method'] == 'Normal Ensemble'].to_csv(sub_ens_path, index=False)
            df_sessions_temp[df_sessions_temp['Method'] == 'Normal Ensemble'].to_csv(ses_ens_path, index=False)
            df_itr_temp[df_itr_temp['Method'] == 'Normal Ensemble'].to_csv(itr_ens_path, index=False)
            
            print(f"  → Saved consolidated{' TEST' if is_test_run else ''} results immediately for {subj_key} seed {seed}.")
            
            # --- EXTRACT INTERPRETABILITY DATA ---
            print("  → Extracting spatial weights, temporal attention, and gradient saliency from best fold...")
            best_model = HST_P300Net(num_channels=8, input_time_samples=140).to(config.DEVICE)
            best_model.load_state_dict(best_fold_weights)
            
            # 1. Spatial Weights (shape [8])
            spatial_w = best_model.block1.conv_s.weight.detach().abs().mean(dim=(0, 1, 3)).cpu().numpy().tolist()
            
            # 2. Temporal Attention Weights (No longer present in the updated architecture)
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
            interpretability_results[subj_key] = {}
            interpretability_results[subj_key][seed_key] = {
                "spatial_weights": spatial_w,
                "temporal_attention": temporal_att,
                "gradient_saliency": grad_saliency
            }
            with open(json_path, "w") as f:
                json.dump(interpretability_results, f, indent=4)

            # Free memory explicitly to prevent memory leaks / OOM / VS Code crashes
            del best_model, test_loader
            del X_pool, y_pool, s_pool, g_pool, ses_pool
            del sub_te_ds
            if 'best_fold_weights' in locals():
                del best_fold_weights
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            malloc_trim()
            
    # Save all aggregate results to main CSV files at the end into 2 sets of 3 files (total 6 files)
    df_subjects = pd.DataFrame(all_run_records)
    df_sessions = pd.DataFrame(session_run_records)
    df_itr = pd.DataFrame(itr_run_records)
    
    if df_subjects.empty:
        print("\nCRITICAL ERROR: No subjects were processed or no records saved.")
    else:
        # Save Best Fold
        df_subjects[df_subjects['Method'] == 'Best Fold'].to_csv(sub_best_path, index=False)
        df_sessions[df_sessions['Method'] == 'Best Fold'].to_csv(ses_best_path, index=False)
        df_itr[df_itr['Method'] == 'Best Fold'].to_csv(itr_best_path, index=False)
        
        # Save Normal Ensemble
        df_subjects[df_subjects['Method'] == 'Normal Ensemble'].to_csv(sub_ens_path, index=False)
        df_sessions[df_sessions['Method'] == 'Normal Ensemble'].to_csv(ses_ens_path, index=False)
        df_itr[df_itr['Method'] == 'Normal Ensemble'].to_csv(itr_ens_path, index=False)
        
        print(f"\nFinal consolidated{' TEST' if is_test_run else ''} CSV results (6 files) saved to: {config.RESULT_DIR}")
        
        # Display Final Aggregate Statistics
        for method_name in ['Best Fold', 'Normal Ensemble']:
            print("\n" + "="*75)
            print(f" FINAL AGGREGATE STATISTICS: {method_name}")
            print("="*75)
            df_m = df_subjects[df_subjects['Method'] == method_name]
            if df_m.empty:
                print("No records for this method.")
                continue
                
            def compute_stats(series):
                mean, std = series.mean(), series.std()
                sem = std / math.sqrt(len(series)) if len(series) > 0 else 0
                median = series.median()
                q25, q75 = series.quantile(0.25), series.quantile(0.75)
                ci_low, ci_high = np.percentile(series, [2.5, 97.5])
                return mean, std, sem, median, (q75-q25), ci_low, ci_high
                
            acc_m, acc_s, acc_se, acc_med, acc_iqr, acc_ci_l, acc_ci_h = compute_stats(df_m['Obj Acc (%)'])
            print(f"Object Accuracy: {acc_m:.2f}% ± {acc_s:.2f}% (SEM: {acc_se:.2f}, Median: {acc_med:.2f}%, IQR: {acc_iqr:.2f}, 95% CI: [{acc_ci_l:.2f}%, {acc_ci_h:.2f}%])")
            f1_m, f1_s, f1_se, f1_med, f1_iqr, f1_ci_l, f1_ci_h = compute_stats(df_m['Macro F1'])
            print(f"Macro F1 Score:  {f1_m:.4f} ± {f1_s:.4f} (SEM: {f1_se:.4f}, Median: {f1_med:.4f}, IQR: {f1_iqr:.4f}, 95% CI: [{f1_ci_l:.4f}, {f1_ci_h:.4f}])")

if __name__ == "__main__":
    main()
