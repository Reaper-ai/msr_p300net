import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

# Add parent scripts/gib directory to python path for shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from dataset import extract_subject_data, SimpleTensorDataset, FastTensorDataLoader
from msr_p300net.model import MSR_P300NET

# Target result directory
RESULT_DIR = os.path.join(config.RESULT_DIR, "msr_p300net")
os.makedirs(RESULT_DIR, exist_ok=True)

# Spelling grid layout characters (up to 48 characters for 6x8 grid)
# Row 0: A-F, plus 2 extra slots (G, H in some maps, or placeholder)
# Since Database 1 has 8 columns (7-14), we map characters based on (row - 1)*8 + (col - 7)
# Let's map characters dynamically or just use indices 0 to 47
GRID_SIZE = 48  # 6 rows * 8 columns

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_grid_coords(idx):
    """
    Returns (row, col) grid index (0 to 5, 0 to 7) for character index (0 to 47)
    """
    return idx // 8, idx % 8

def compute_distances(true_idx, pred_idx):
    """
    Calculates vertical and horizontal grid distance between true and predicted characters.
    """
    r_t, c_t = get_grid_coords(true_idx)
    r_p, c_p = get_grid_coords(pred_idx)
    return abs(r_p - r_t), abs(c_p - c_t)

def get_error_category(true_idx, pred_idx):
    """
    Categorizes the spelling selection: correct, same-row adjacent, same-column adjacent, diagonal adjacent, or non-adjacent.
    """
    dr, dc = compute_distances(true_idx, pred_idx)
    if dr == 0 and dc == 0:
        return "correct"
    elif dr <= 1 and dc <= 1:
        if dr == 0 and dc == 1:
            return "same_row_adjacent"
        elif dr == 1 and dc == 0:
            return "same_col_adjacent"
        elif dr == 1 and dc == 1:
            return "diagonal_adjacent"
    return "non_adjacent"

def analyze_matrices(matrix):
    """
    Analyzes a 48x48 confusion matrix to count errors by adjacency categories.
    """
    total = int(matrix.sum())
    correct = int(np.diag(matrix).sum())
    errors = total - correct
    
    same_row_adj = 0
    same_col_adj = 0
    diagonal_adj = 0
    non_adj = 0
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if i == j:
                continue
            count = int(matrix[i, j])
            if count == 0:
                continue
            category = get_error_category(i, j)
            if category == "same_row_adjacent":
                same_row_adj += count
            elif category == "same_col_adjacent":
                same_col_adj += count
            elif category == "diagonal_adjacent":
                diagonal_adj += count
            else:
                non_adj += count
                
    adjacent_total = same_row_adj + same_col_adj + diagonal_adj
    return {
        "total_selections": total,
        "correct_selections": correct,
        "total_errors": errors,
        "accuracy_pct": (correct / total * 100.0) if total > 0 else 0.0,
        "same_row_adjacent": same_row_adj,
        "same_col_adjacent": same_col_adj,
        "diagonal_adjacent": diagonal_adj,
        "adjacent_total": adjacent_total,
        "non_adjacent": non_adj,
        "adjacent_error_share_pct": (adjacent_total / errors * 100.0) if errors > 0 else 0.0
    }

def main():
    print("Starting GIB P300 Confusion Matrix & Adjacent Error Analysis...\n")
    
    seeds = config.SEEDS
    subjects = config.SUBJECT_IDS
    
    results_json = {}
    
    for seed in seeds:
        print(f"================ Running Inference for Seed: {seed} ================")
        set_seed(seed)
        
        # Initialize 48x48 matrices
        global_mat = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
        healthy_mat = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
        unhealthy_mat = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
        
        for subj_id in subjects:
            # Extract demographics
            db_id = config.SUBJECT_METADATA[subj_id]['dataset']
            condition = config.SUBJECT_METADATA[subj_id]['condition']
            
            # Load checkpoints path
            model_name = f"subj_{subj_id:02d}_seed_{seed}_best.pth"
            model_path = os.path.join(config.MODEL_DIR, model_name)
            if not os.path.exists(model_path):
                # Skip if model hasn't been trained/saved
                continue
                
            data = extract_subject_data(config.DATA_PATH, subj_id)
            if data is None:
                continue
                
            X_sub, y_sub, char_ids_sub, codes_sub, runs_sub, dbs_sub = data
            unique_runs = np.unique(runs_sub)
            unique_runs.sort()
            
            n_calib_runs = 2 if db_id in [0, 1] else 1
            if len(unique_runs) <= n_calib_runs:
                continue
                
            calib_runs = unique_runs[:n_calib_runs]
            calib_mask = np.isin(runs_sub, calib_runs)
            test_mask = ~calib_mask
            
            calib_chars = np.unique(char_ids_sub[calib_mask])
            if len(calib_chars) < 2:
                continue
                
            # Must replicate deterministic train/val split for normalization
            np.random.shuffle(calib_chars)
            split_train = int(0.8 * len(calib_chars))
            train_chars = calib_chars[:split_train]
            
            mask_tr = np.isin(char_ids_sub, train_chars)
            mask_te = test_mask
            
            X_tr = X_sub[mask_tr]
            X_te, y_te, cid_te, c_te = X_sub[mask_te], y_sub[mask_te], char_ids_sub[mask_te], codes_sub[mask_te]
            
            # Normalize using same train stats
            mean_tr = X_tr.mean(axis=(0, 2), keepdims=True)
            std_tr = X_tr.std(axis=(0, 2), keepdims=True)
            X_te = (X_te - mean_tr) / (std_tr + 1e-8)
            
            s_te = np.full(len(y_te), subj_id, dtype=np.int64)
            te_ds = SimpleTensorDataset(torch.tensor(X_te).unsqueeze(1), torch.tensor(y_te), torch.tensor(s_te), torch.tensor(cid_te), torch.tensor(c_te))
            test_loader = FastTensorDataLoader(te_ds, batch_size=config.BATCH_SIZE, shuffle=False, device=config.DEVICE)
            
            # Instantiate model and load weights
            model = MSR_P300NET(n_channels=8, n_samples=128).to(config.DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
            model.eval()
            
            # Extract predictions
            block_probs = defaultdict(lambda: defaultdict(list))
            block_targets = {}
            
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch[0]
                    target = batch[1].cpu().numpy().ravel()
                    char_ids = batch[3].cpu().numpy().ravel()
                    codes = batch[4].cpu().numpy().ravel()
                    
                    with torch.amp.autocast(device_type=config.DEVICE.type):
                        probs = model(inputs).squeeze(1).cpu().numpy().ravel()
                        
                    for prob, tgt, cid, code in zip(probs, target, char_ids, codes):
                        cid, code = int(cid), int(code)
                        # Skip silence/inter-stimulus codes
                        if code == 0:
                            continue
                        block_probs[cid][code].append(prob)
                        if int(tgt) == 1:
                            if cid not in block_targets:
                                block_targets[cid] = {}
                            if code <= 6:
                                block_targets[cid]['row'] = code
                            else:
                                block_targets[cid]['col'] = code
                                
            # Make block selections
            for cid, gold in block_targets.items():
                if 'row' not in gold or 'col' not in gold:
                    continue
                # Skip invalid row or column codes
                if gold['row'] < 1 or gold['row'] > 6 or gold['col'] < 7 or gold['col'] > 14:
                    continue
                
                avg_scores = {obj: np.mean(s_list) for obj, s_list in block_probs[cid].items()}
                row_codes = {k: v for k, v in avg_scores.items() if k <= 6}
                col_codes = {k: v for k, v in avg_scores.items() if k >= 7}
                
                if not row_codes or not col_codes:
                    continue
                    
                pred_r = max(row_codes, key=row_codes.get)
                pred_c = max(col_codes, key=col_codes.get)
                
                # Verify predicted codes are within valid bounds
                if pred_r < 1 or pred_r > 6 or pred_c < 7 or pred_c > 14:
                    continue
                
                # Map row/column to 6x8 index (0 to 47)
                true_idx = (gold['row'] - 1) * 8 + (gold['col'] - 7)
                pred_idx = (pred_r - 1) * 8 + (pred_c - 7)
                
                # Accumulate counts
                global_mat[true_idx, pred_idx] += 1
                if condition == 'Healthy':
                    healthy_mat[true_idx, pred_idx] += 1
                else:
                    unhealthy_mat[true_idx, pred_idx] += 1
                    
        # Analyze results for this seed
        stats_global = analyze_matrices(global_mat)
        stats_healthy = analyze_matrices(healthy_mat)
        stats_unhealthy = analyze_matrices(unhealthy_mat)
        
        print(f"\nSEED {seed} ANALYSIS SUMMARY:")
        print(f"Cohort   | Selections | Accuracy | Total Errors | Adjacent Errors (Row / Col / Diag) | Non-Adjacent | Adjacent Share")
        print("-" * 125)
        for name, stats in [("Global", stats_global), ("Healthy", stats_healthy), ("Patient", stats_unhealthy)]:
            print(f"{name:<8} | {stats['total_selections']:10d} | {stats['accuracy_pct']:7.2f}% | {stats['total_errors']:12d} | "
                  f"{stats['adjacent_total']:15d} ({stats['same_row_adjacent']}/{stats['same_col_adjacent']}/{stats['diagonal_adjacent']}) | "
                  f"{stats['non_adjacent']:12d} | {stats['adjacent_error_share_pct']:13.2f}%")
        print("\n")
        
        # Save matrices in output dictionary
        results_json[f"seed_{seed}"] = {
            "global": global_mat.tolist(),
            "healthy": healthy_mat.tolist(),
            "unhealthy": unhealthy_mat.tolist(),
            "stats": {
                "global": stats_global,
                "healthy": stats_healthy,
                "unhealthy": stats_unhealthy
            }
        }
        
    # Write confusion matrices to JSON file
    json_path = os.path.join(RESULT_DIR, "confusion_matrices.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=4)
        
    print(f"Analysis complete. Confusion matrices and statistics saved to: {json_path}")

if __name__ == "__main__":
    main()
