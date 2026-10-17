import os
import sys
import copy
import ctypes
import gc
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Add scripts/gist directory to python path for modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from dataset import load_rsvp_splits, load_p300_splits, FastTensorDataLoader, SafeEEGAugmenter
import ctypes
from evaluation import evaluate_rsvp_metrics, evaluate_p300_metrics, ParametricFocalLoss
from model import EEGNetv4, apply_eegnet_constraints

def malloc_trim():
    try:
        ctypes.cdll.LoadLibrary('libc.so.6').malloc_trim(0)
    except Exception:
        pass

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_augmented_model(tr_ldr, va_ldr, n_channels, epochs_limit):
    model = EEGNetv4(in_chans=n_channels, n_classes=1, input_window_samples=config.N_SAMPLES).to(config.DEVICE)
    augmenter = SafeEEGAugmenter(p=0.5, noise_std=0.05, scale_range=(0.9, 1.1), max_shift=2, drop_prob=0.1).to(config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.MAX_LR / 25.0, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.MAX_LR, epochs=epochs_limit, 
        steps_per_epoch=len(tr_ldr), pct_start=config.PCT_START
    )
    criterion = ParametricFocalLoss(gamma=config.GAMMA, pos_weight=config.POS_WEIGHT, smoothing=config.SMOOTHING).to(config.DEVICE)
    scaler = torch.amp.GradScaler(device=config.DEVICE.type)

    best_loss, best_w = float('inf'), None
    epochs_no_improve = 0
    stopped_ep = epochs_limit

    for epoch in range(1, epochs_limit + 1):
        model.train()
        augmenter.train()

        for X_b, y_b, _ in tr_ldr:
            optimizer.zero_grad(set_to_none=True)
            X_aug = augmenter(X_b)
            with torch.amp.autocast(device_type=config.DEVICE.type):
                output = model(X_aug)
                loss = criterion(output, y_b.float().unsqueeze(1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            apply_eegnet_constraints(model)
            scheduler.step()

        model.eval()
        augmenter.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b, _ in va_ldr:
                with torch.amp.autocast(device_type=config.DEVICE.type):
                    val_loss += criterion(model(X_b), y_b.float().unsqueeze(1)).item()
        val_loss /= max(1, len(va_ldr))

        if val_loss < best_loss:
            best_loss = val_loss
            best_w = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE:
            stopped_ep = epoch
            break

    if best_w:
        model.load_state_dict(best_w)
    return model, stopped_ep

def parse_args():
    parser = argparse.ArgumentParser(description="Won2022 EEGNetv4 GIST Benchmark Runner")
    parser.add_argument("--mode", type=str, default="both", choices=["rsvp", "p300", "both"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    parser.add_argument("--subjects", type=int, nargs="+", default=config.ALL_SUBJECTS)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results and force rerun")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Won2022 GIST EEGNetv4 Benchmark Pipeline:")
    print(f"  Mode:      {args.mode.upper()}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Subjects:  {args.subjects}")
    print(f"  No Save:   {args.no_save}")
    print(f"  Overwrite: {args.overwrite}")
    print(f"  Device:    {config.DEVICE}")

    # RSVP Task Execution
    if args.mode in ["rsvp", "both"]:
        print(f"\n{'='*85}\n RUNNING RSVP COHORT BENCHMARK (EEGNETV4)\n{'='*85}")
        csv_path = os.path.join(config.OUTPUT_DIR, "eegnet", "rsvp_cohort_results_eegnet.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        if not args.no_save and os.path.exists(csv_path):
            try:
                df_exist = pd.read_csv(csv_path)
                if args.overwrite:
                    subjects_to_drop = [f'S{s:02d}' for s in args.subjects]
                    df_exist = df_exist[~(df_exist['Seed'].isin(args.seeds) & df_exist['Subject'].isin(subjects_to_drop))]
                    df_exist.to_csv(csv_path, index=False)
                    print(f"Selective overwrite: removed existing RSVP rows for subjects {subjects_to_drop} and seeds {args.seeds} in {csv_path}")
            except Exception as e:
                print(f"Error handling overwrite: {e}")
                
        existing_runs = set()
        if not args.no_save and os.path.exists(csv_path):
            try:
                df_exist = pd.read_csv(csv_path)
                for _, r in df_exist.iterrows():
                    existing_runs.add((int(r['Seed']), r['Subject']))
            except Exception:
                pass

        rsvp_results = []
        for seed in args.seeds:
            print(f"\n================ Running RSVP Seed: {seed} ================")
            set_seed(seed)
            for subj in args.subjects:
                subj_name = f'S{subj:02d}'
                if (seed, subj_name) in existing_runs:
                    print(f"Skipping Seed {seed} | Subject {subj_name} (already computed)")
                    continue
                
                splits = load_rsvp_splits(subj, channels_to_use=config.CHANNELS_8)
                if splits is None:
                    continue
                (X_tr, y_tr, s_tr), (X_va, y_va, s_va), (X_te, y_te, s_te) = splits
                tr_ldr = FastTensorDataLoader(X_tr, y_tr, s_tr, config.BATCH_SIZE, True)
                va_ldr = FastTensorDataLoader(X_va, y_va, s_va, config.BATCH_SIZE, False)
                te_ldr = FastTensorDataLoader(X_te, y_te, s_te, config.BATCH_SIZE, False)

                model, stopped_ep = train_augmented_model(tr_ldr, va_ldr, n_channels=8, epochs_limit=args.epochs)
                auc, bal_acc, seq_acc, f1 = evaluate_rsvp_metrics(model, te_ldr)

                run_result = {
                    'Seed': seed,
                    'Subject': subj_name,
                    'Stopped Ep': stopped_ep,
                    'AUC': auc,
                    'Single BalAcc (%)': bal_acc,
                    'Seq Top-1 Acc (%)': seq_acc,
                    'Macro F1': f1
                }
                rsvp_results.append(run_result)
                print(f" >> RSVP {subj_name} | Seed {seed} | Ep: {stopped_ep:3d} | AUC: {auc:.4f} | Single BalAcc: {bal_acc:5.1f}% | Seq Top-1 Acc: {seq_acc:5.1f}% | F1: {f1:.4f}")

                if not args.no_save:
                    pd.DataFrame([run_result]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
                    existing_runs.add((seed, subj_name))

                del model, tr_ldr, va_ldr, te_ldr, X_tr, X_va, X_te
                gc.collect()
                torch.cuda.empty_cache()
                malloc_trim()

        if not args.no_save and os.path.exists(csv_path):
            df_rsvp = pd.read_csv(csv_path)
            print(f"\n--- RSVP BENCHMARK SUMMARY (EEGNETV4) ---")
            print(f"Mean Seq Top-1 Acc: {df_rsvp['Seq Top-1 Acc (%)'].mean():5.2f}% ± {df_rsvp['Seq Top-1 Acc (%)'].std():5.2f}%")
            print(f"Mean AUC-ROC:       {df_rsvp['AUC'].mean():5.4f} ± {df_rsvp['AUC'].std():5.4f}")
            print(f"Mean Single BalAcc: {df_rsvp['Single BalAcc (%)'].mean():5.2f}% ± {df_rsvp['Single BalAcc (%)'].std():5.2f}%")
            print(f"Mean Macro F1:      {df_rsvp['Macro F1'].mean():5.4f} ± {df_rsvp['Macro F1'].std():5.4f}")
            print("-" * 85)

    # P300 Task Execution
    if args.mode in ["p300", "both"]:
        print(f"\n{'='*85}\n RUNNING P300 SPELLER COHORT BENCHMARK (EEGNETV4)\n{'='*85}")
        csv_path = os.path.join(config.OUTPUT_DIR, "eegnet", "p300_cohort_results_eegnet.csv")
        itr_csv_path = os.path.join(config.OUTPUT_DIR, "eegnet", "p300_itr_results_eegnet.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        if not args.no_save and args.overwrite:
            subjects_to_drop = [f'S{s:02d}' for s in args.subjects]
            if os.path.exists(csv_path):
                try:
                    df_exist = pd.read_csv(csv_path)
                    df_exist = df_exist[~(df_exist['Seed'].isin(args.seeds) & df_exist['Subject'].isin(subjects_to_drop))]
                    df_exist.to_csv(csv_path, index=False)
                    print(f"Selective overwrite: removed existing P300 rows for subjects {subjects_to_drop} and seeds {args.seeds} in {csv_path}")
                except Exception as e:
                    print(f"Error handling overwrite for P300 CSV: {e}")
            if os.path.exists(itr_csv_path):
                try:
                    df_exist = pd.read_csv(itr_csv_path)
                    df_exist = df_exist[~(df_exist['Seed'].isin(args.seeds) & df_exist['Subject'].isin(subjects_to_drop))]
                    df_exist.to_csv(itr_csv_path, index=False)
                    print(f"Selective overwrite: removed existing P300 ITR rows in {itr_csv_path}")
                except Exception as e:
                    print(f"Error handling overwrite for P300 ITR CSV: {e}")

        existing_p300 = set()
        if not args.no_save and os.path.exists(csv_path):
            try:
                df_exist = pd.read_csv(csv_path)
                for _, r in df_exist.iterrows():
                    existing_p300.add((int(r['Seed']), r['Subject']))
            except Exception:
                pass

        p300_results = []
        for seed in args.seeds:
            print(f"\n================ Running P300 Seed: {seed} ================")
            set_seed(seed)
            for subj in args.subjects:
                subj_name = f'S{subj:02d}'
                if (seed, subj_name) in existing_p300:
                    print(f"Skipping Seed {seed} | Subject {subj_name} (already computed)")
                    continue

                splits = load_p300_splits(subj, channels_to_use=config.CHANNELS_8)
                if splits is None:
                    continue
                (X_tr, y_tr, t_tr), (X_va, y_va, t_va), (X_te, y_te, t_te) = splits
                tr_ldr = FastTensorDataLoader(X_tr, y_tr, t_tr, config.BATCH_SIZE, True)
                va_ldr = FastTensorDataLoader(X_va, y_va, t_va, config.BATCH_SIZE, False)
                te_ldr = FastTensorDataLoader(X_te, y_te, t_te, config.BATCH_SIZE, False)

                model, stopped_ep = train_augmented_model(tr_ldr, va_ldr, n_channels=8, epochs_limit=args.epochs)
                trial_auc, trial_bacc, pairwise_acc, f1, seq_accs, seq_itrs = evaluate_p300_metrics(model, te_ldr)

                run_result = {
                    'Seed': seed,
                    'Subject': subj_name,
                    'Channels': 8,
                    'Stopped Ep': stopped_ep,
                    'Trial BalAcc (%)': trial_bacc,
                    'Pairwise Acc (%)': pairwise_acc,
                    'Trial AUC': trial_auc,
                    'Macro F1': f1
                }
                p300_results.append(run_result)

                # Save sequence-wise ITR
                itr_list = []
                for s_idx, (s_acc, s_itr) in enumerate(zip(seq_accs, seq_itrs)):
                    itr_list.append({
                        'Seed': seed,
                        'Subject': subj_name,
                        'Channels': 8,
                        'Sequence': s_idx + 1,
                        'Accuracy (%)': s_acc,
                        'ITR (bits/select)': s_itr
                    })

                print(f" >> P300 {subj_name} | Seed {seed} | Ep: {stopped_ep:3d} | Trial BalAcc: {trial_bacc:5.2f}% | Pairwise: {pairwise_acc:5.2f}% | Trial AUC: {trial_auc:5.4f} | F1: {f1:.4f}")

                if not args.no_save:
                    pd.DataFrame([run_result]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
                    pd.DataFrame(itr_list).to_csv(itr_csv_path, mode='a', header=not os.path.exists(itr_csv_path), index=False)
                    existing_p300.add((seed, subj_name))

                del model, tr_ldr, va_ldr, te_ldr, X_tr, X_va, X_te
                gc.collect()
                torch.cuda.empty_cache()
                malloc_trim()

        if not args.no_save and os.path.exists(csv_path):
            df_p300 = pd.read_csv(csv_path)
            print(f"\n--- P300 SPELLER BENCHMARK SUMMARY (EEGNETV4) ---")
            print(f"Mean Trial BalAcc: {df_p300['Trial BalAcc (%)'].mean():5.2f}% ± {df_p300['Trial BalAcc (%)'].std():5.2f}%")
            print(f"Mean Pairwise Acc: {df_p300['Pairwise Acc (%)'].mean():5.2f}% ± {df_p300['Pairwise Acc (%)'].std():5.2f}%")
            print(f"Mean Trial AUC:    {df_p300['Trial AUC'].mean():5.4f} ± {df_p300['Trial AUC'].std():5.4f}")
            print(f"Mean Macro F1:     {df_p300['Macro F1'].mean():5.4f} ± {df_p300['Macro F1'].std():5.4f}")
            print("-" * 85)

if __name__ == "__main__":
    main()
