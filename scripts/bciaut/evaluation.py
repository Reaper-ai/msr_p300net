import os
import sys
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def evaluate_comprehensive_metrics(model, loader, max_seq=10, n_classes=8):
    """
    Computes comprehensive metrics including overall accuracy, macro F1, session-wise accuracy/F1,
    and ITR (Information Transfer Rate) across sequences using raw logit-averaging from main.py.
    """
    model.eval()
    block_scores = defaultdict(lambda: defaultdict(list))
    block_targets, block_sessions = {}, {}
    
    with torch.no_grad():
        for batch in loader:
            data = batch[0]
            target = batch[1].cpu().numpy().ravel()
            group_ids = batch[3].cpu().numpy().ravel()
            sess_ids = batch[4].cpu().numpy().ravel()
            
            with torch.amp.autocast(device_type=config.DEVICE.type):
                # Extract raw logits directly
                logits = model(data).squeeze(1).cpu().numpy().ravel()
            
            for logit, tgt, gid, sess in zip(logits, target, group_ids, sess_ids):
                gid = int(gid)
                block_id, object_id = gid // 100, gid % 100
                
                # Append raw logits for mathematically sound averaging
                block_scores[block_id][object_id].append(logit)
                    
                block_sessions[block_id] = int(sess)
                if int(tgt) == 1: 
                    block_targets[block_id] = object_id

    session_preds, session_trues = defaultdict(list), defaultdict(list)
    y_true, y_pred = [], []
    
    for block_id, true_obj in block_targets.items():
        sess = block_sessions[block_id]
        # Average the raw logits
        avg_scores = {obj: np.mean(s_list) for obj, s_list in block_scores[block_id].items()}
        # Select object with the highest average logit
        pred_obj = max(avg_scores, key=avg_scores.get)
        
        y_true.append(true_obj); y_pred.append(pred_obj)
        session_trues[sess].append(true_obj); session_preds[sess].append(pred_obj)

    overall_acc = accuracy_score(y_true, y_pred) * 100.0
    overall_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    session_metrics = {}
    for sess in session_trues:
        s_acc = accuracy_score(session_trues[sess], session_preds[sess]) * 100.0
        s_f1 = f1_score(session_trues[sess], session_preds[sess], average='macro', zero_division=0)
        session_metrics[sess] = {'Accuracy (%)': s_acc, 'Macro F1': s_f1}

    itr_results = []
    for seq_limit in range(1, max_seq+1):
        correct, total = 0, len(block_targets)
        for block_id, true_obj in block_targets.items():
            avg_scores = {obj: np.mean(s_list[:seq_limit]) for obj, s_list in block_scores[block_id].items() if len(s_list)>0}
            if avg_scores and max(avg_scores, key=avg_scores.get) == true_obj: 
                correct += 1
                
        acc = correct / total if total > 0 else 0.0
        if acc <= (1.0/n_classes): itr = 0.0
        elif acc >= 1.0: itr = math.log2(n_classes)
        else: itr = (math.log2(n_classes) + acc*math.log2(acc) + (1.0-acc)*math.log2((1.0-acc)/(n_classes-1)))
        itr_results.append({'Sequences': seq_limit, 'Accuracy (%)': acc*100.0, 'Bits/Selection': itr})

    return overall_acc, overall_f1, session_metrics, pd.DataFrame(itr_results)

def evaluate_comprehensive_metrics_from_logits(logits, targets, group_ids, sess_ids, max_seq=10, n_classes=8):
    """
    Computes comprehensive metrics (accuracy, F1, ITR) directly from pre-computed raw logits,
    targets, group IDs, and session IDs.
    """
    block_scores = defaultdict(lambda: defaultdict(list))
    block_targets, block_sessions = {}, {}
    
    for logit, tgt, gid, sess in zip(logits, targets, group_ids, sess_ids):
        gid = int(gid)
        block_id, object_id = gid // 100, gid % 100
        block_scores[block_id][object_id].append(logit)
        block_sessions[block_id] = int(sess)
        if int(tgt) == 1: 
            block_targets[block_id] = object_id

    session_preds, session_trues = defaultdict(list), defaultdict(list)
    y_true, y_pred = [], []
    
    for block_id, true_obj in block_targets.items():
        sess = block_sessions[block_id]
        # Average the raw logits
        avg_scores = {obj: np.mean(s_list) for obj, s_list in block_scores[block_id].items()}
        # Select object with the highest average logit
        pred_obj = max(avg_scores, key=avg_scores.get)
        
        y_true.append(true_obj); y_pred.append(pred_obj)
        session_trues[sess].append(true_obj); session_preds[sess].append(pred_obj)

    overall_acc = accuracy_score(y_true, y_pred) * 100.0
    overall_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    session_metrics = {}
    for sess in session_trues:
        s_acc = accuracy_score(session_trues[sess], session_preds[sess]) * 100.0
        s_f1 = f1_score(session_trues[sess], session_preds[sess], average='macro', zero_division=0)
        session_metrics[sess] = {'Accuracy (%)': s_acc, 'Macro F1': s_f1}

    itr_results = []
    for seq_limit in range(1, max_seq+1):
        correct, total = 0, len(block_targets)
        for block_id, true_obj in block_targets.items():
            avg_scores = {obj: np.mean(s_list[:seq_limit]) for obj, s_list in block_scores[block_id].items() if len(s_list)>0}
            if avg_scores and max(avg_scores, key=avg_scores.get) == true_obj: 
                correct += 1
                
        acc = correct / total if total > 0 else 0.0
        if acc <= (1.0/n_classes): itr = 0.0
        elif acc >= 1.0: itr = math.log2(n_classes)
        else: itr = (math.log2(n_classes) + acc*math.log2(acc) + (1.0-acc)*math.log2((1.0-acc)/(n_classes-1)))
        itr_results.append({'Sequences': seq_limit, 'Accuracy (%)': acc*100.0, 'Bits/Selection': itr})

    return overall_acc, overall_f1, session_metrics, pd.DataFrame(itr_results)
