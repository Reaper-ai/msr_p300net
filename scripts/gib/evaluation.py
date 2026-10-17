import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score

def evaluate_metrics(model, loader, device, model_type="softmax", max_seq=15, n_classes=36):
    """
    Standardized metric evaluation (Accuracy, Macro F1, ITR) for GIB models.
    Supports model_type="softmax" (outputs probabilities/logits of shape [B, 2])
    and model_type="logit" (outputs single binary logit of shape [B, 1]).
    """
    model.eval()
    block_scores = defaultdict(lambda: defaultdict(list))
    block_targets = {}
    
    with torch.no_grad():
        for batch in loader:
            data = batch[0].to(device)
            target = batch[1].cpu().numpy().ravel()
            char_ids = batch[3].cpu().numpy().ravel()
            codes = batch[4].cpu().numpy().ravel()
            
            with torch.amp.autocast(device_type=device.type):
                outputs = model(data)
                if model_type == "softmax":
                    # If model has Softmax output, outputs are already probabilities.
                    # Otherwise, apply softmax to logits.
                    if hasattr(model, 'softmax') or (hasattr(model, 'fc') and isinstance(outputs, torch.Tensor) and outputs.max() <= 1.0 and outputs.min() >= 0.0):
                        probs = outputs[:, 1].cpu().numpy()
                    else:
                        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                else: # "logit"
                    probs = outputs.squeeze(1).cpu().numpy().ravel()
            
            for prob, tgt, cid, code in zip(probs, target, char_ids, codes):
                cid, code = int(cid), int(code)
                block_scores[cid][code].append(prob)
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
