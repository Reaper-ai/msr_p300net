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
from dataset import load_p300_splits, FastTensorDataLoader, SafeEEGAugmenter
from evaluation import evaluate_p300_metrics, ParametricFocalLoss
from model import MSR_P300NET, apply_max_norm

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
    model = MSR_P300NET(n_channels=n_channels, n_samples=config.N_SAMPLES).to(config.DEVICE)
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

            apply_max_norm(model)
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
    parser = argparse.ArgumentParser(description="Won2022 MSR_P300NET Channel Ablation Runner")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    parser.add_argument("--subjects", type=int, nargs="+", default=config.ALL_SUBJECTS)
    parser.add_argument("--channels", type=int, nargs="+", default=[4, 16, 32])
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results and force rerun")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Won2022 GIST MSR_P300NET Channel Ablation Pipeline:")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Subjects:  {args.subjects}")
    print(f"  Channels:  {args.channels}")
    print(f"  No Save:   {args.no_save}")
    print(f"  Overwrite: {args.overwrite}")
    print(f"  Device:    {config.DEVICE}")

    chan_map = {
        4: config.CHANNELS_4,
        8: config.CHANNELS_8,
        16: config.CHANNELS_16,
        32: config.CHANNELS_32
    }
    interpretability_results = {}
    json_path = os.path.join(config.OUTPUT_DIR, "msr_p300net_ablation", "msr_p300net_ablation_interpretability.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if not args.no_save and not args.overwrite and os.path.exists(json_path):
        try:
            import json
            with open(json_path, "r") as f:
                interpretability_results = json.load(f)
        except Exception:
            pass
    csv_path = os.path.join(config.OUTPUT_DIR, 'msr_p300net_ablation', 'p300_ablation_results_msr_p300net.csv')
    itr_csv_path = os.path.join(config.OUTPUT_DIR, 'msr_p300net_ablation', 'p300_ablation_itr_results_msr_p300net.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not args.no_save and args.overwrite:
        for path in [csv_path, itr_csv_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Overwriting: deleted: {path}")
                except Exception:
                    pass

    existing_p300 = set()
    if not args.no_save and os.path.exists(csv_path):
        try:
            df_exist = pd.read_csv(csv_path)
            for _, r in df_exist.iterrows():
                existing_p300.add((int(r['Seed']), r['Subject'], int(r['Channels'])))
        except Exception:
            pass

    p300_results = []
    for seed in args.seeds:
        print(f"\n================ Running Seed: {seed} ================")
        for ch_config in args.channels:
            print(f"--- Channel Config: {ch_config} channels ---")
            set_seed(seed)
            channels_to_use = chan_map.get(ch_config, config.CHANNELS_8)
            
            for subj in args.subjects:
                subj_name = f'S{subj:02d}'
                if (seed, subj_name, ch_config) in existing_p300:
                    print(f"Skipping Seed {seed} | Subject {subj_name} | Ch {ch_config} (already computed)")
                    continue

                splits = load_p300_splits(subj, channels_to_use=channels_to_use)
                if splits is None:
                    continue
                (X_tr, y_tr, t_tr), (X_va, y_va, t_va), (X_te, y_te, t_te) = splits
                tr_ldr = FastTensorDataLoader(X_tr, y_tr, t_tr, config.BATCH_SIZE, True)
                va_ldr = FastTensorDataLoader(X_va, y_va, t_va, config.BATCH_SIZE, False)
                te_ldr = FastTensorDataLoader(X_te, y_te, t_te, config.BATCH_SIZE, False)

                model, stopped_ep = train_augmented_model(tr_ldr, va_ldr, n_channels=ch_config, epochs_limit=args.epochs)
                trial_auc, trial_bacc, pairwise_acc, f1, seq_accs, seq_itrs = evaluate_p300_metrics(model, te_ldr)

                run_result = {
                    'Seed': seed,
                    'Subject': subj_name,
                    'Channels': ch_config,
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
                        'Channels': ch_config,
                        'Sequence': s_idx + 1,
                        'Accuracy (%)': s_acc,
                        'ITR (bits/select)': s_itr
                    })

                print(f" >> P300 {subj_name} | Ch: {ch_config:2d} | Seed {seed} | Ep: {stopped_ep:3d} | Trial BalAcc: {trial_bacc:5.2f}% | Pairwise: {pairwise_acc:5.2f}% | Trial AUC: {trial_auc:5.4f} | F1: {f1:.4f}")

                if not args.no_save:
                    model_dir = os.path.join(config.MODEL_DIR, "msr_p300net_ablation")
                    os.makedirs(model_dir, exist_ok=True)
                    model_save_path = os.path.join(model_dir, f"subj_{subj:02d}_channels_{ch_config}_seed_{seed}_best.pth")
                    torch.save(model.state_dict(), model_save_path)

                    # Interpretability Extraction
                    spatial_w = model.conv_s.weight.detach().abs().mean(dim=(0, 1, 3)).cpu().numpy().tolist()
                    temporal_att = []
                    target_test_mask = (y_te == 1)
                    if target_test_mask.any():
                        model.eval()
                        X_te_targets = torch.tensor(X_te[target_test_mask], dtype=torch.float32, device=config.DEVICE)
                        X_te_targets.requires_grad = True
                        with torch.enable_grad():
                            with torch.amp.autocast(device_type=config.DEVICE.type):
                                logits = model(X_te_targets)
                            logits.backward(torch.ones_like(logits))
                        grad_saliency = X_te_targets.grad.detach().abs().mean(dim=0).cpu().numpy().tolist()
                    else:
                        grad_saliency = []

                    task_key = f"p300_ch_{ch_config}"
                    subj_key = f"SBJ{subj:02d}"
                    seed_key = f"seed_{seed}"
                    if task_key not in interpretability_results:
                        interpretability_results[task_key] = {}
                    if subj_key not in interpretability_results[task_key]:
                        interpretability_results[task_key][subj_key] = {}
                    interpretability_results[task_key][subj_key][seed_key] = {
                        "spatial_weights": spatial_w,
                        "temporal_attention": temporal_att,
                        "gradient_saliency": grad_saliency
                    }

                    import json
                    with open(json_path, "w") as f:
                        json.dump(interpretability_results, f, indent=4)

                    pd.DataFrame([run_result]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
                    pd.DataFrame(itr_list).to_csv(itr_csv_path, mode='a', header=not os.path.exists(itr_csv_path), index=False)
                    existing_p300.add((seed, subj_name, ch_config))

                del model, tr_ldr, va_ldr, te_ldr, X_tr, X_va, X_te
                gc.collect()
                torch.cuda.empty_cache()
                malloc_trim()

    if not args.no_save and os.path.exists(csv_path):
        df_p300 = pd.read_csv(csv_path)
        print(f"\n--- P300 SPELLER ABLATION SUMMARY (MSR_P300NET) ---")
        print(f"Mean Trial BalAcc: {df_p300['Trial BalAcc (%)'].mean():5.2f}% ± {df_p300['Trial BalAcc (%)'].std():5.2f}%")
        print(f"Mean Pairwise Acc: {df_p300['Pairwise Acc (%)'].mean():5.2f}% ± {df_p300['Pairwise Acc (%)'].std():5.2f}%")
        print(f"Mean Trial AUC:    {df_p300['Trial AUC'].mean():5.4f} ± {df_p300['Trial AUC'].std():5.4f}")
        print(f"Mean Macro F1:     {df_p300['Macro F1'].mean():5.4f} ± {df_p300['Macro F1'].std():5.4f}")
        print("-" * 85)

if __name__ == "__main__":
    main()
