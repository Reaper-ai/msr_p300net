import math
from collections import defaultdict
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import torch

import torch.nn as nn
import torch.nn.functional as F
import config

class ParametricFocalLoss(nn.Module):
    def __init__(self, gamma=config.GAMMA, pos_weight=config.POS_WEIGHT, smoothing=config.SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets_smooth = targets * (1.0 - self.smoothing) + (1.0 - targets) * self.smoothing
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets_smooth,
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device),
            reduction='none'
        )
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        return ((1.0 - pt) ** self.gamma * bce).mean()

def calculate_itr(acc_fraction, n_classes=2):
    if acc_fraction <= (1.0 / n_classes):
        return 0.0
    elif acc_fraction >= 1.0:
        return math.log2(n_classes)
    else:
        return (math.log2(n_classes) + 
                acc_fraction * math.log2(acc_fraction) + 
                (1.0 - acc_fraction) * math.log2((1.0 - acc_fraction) / (n_classes - 1)))

def evaluate_rsvp_metrics(model, loader):
    model.eval()
    probs, targets, seqs = [], [], []
    with torch.no_grad():
        for X_b, y_b, s_b in loader:
            p = torch.sigmoid(model(X_b)).squeeze(1).cpu().numpy()
            probs.extend(p)
            targets.extend(y_b.cpu().numpy())
            seqs.extend(s_b.cpu().numpy())

    probs = np.array(probs)
    targets = np.array(targets)
    try:
        auc = roc_auc_score(targets, probs)
    except Exception:
        auc = 0.5

    best_f1, best_th = 0.0, 0.5
    for th in np.linspace(0.1, 0.9, 81):
        preds = (probs >= th).astype(int)
        score = f1_score(targets, preds, average='macro', zero_division=0)
        if score > best_f1:
            best_f1, best_th = score, th
    bal_acc = balanced_accuracy_score(targets, (probs >= best_th).astype(int)) * 100.0

    seq_logits = defaultdict(list)
    seq_targets = defaultdict(list)
    for p, t, s in zip(probs, targets, seqs):
        seq_logits[s].append(p)
        seq_targets[s].append(t)

    correct, total = 0, 0
    for s, p_list in seq_logits.items():
        t_list = np.array(seq_targets[s])
        if not np.any(t_list == 1):
            continue
        total += 1
        if np.argmax(p_list) in np.where(t_list == 1)[0]:
            correct += 1
    seq_acc = (correct / total * 100.0) if total > 0 else 0.0
    return auc, bal_acc, seq_acc, best_f1

def evaluate_p300_metrics(model, loader, max_seq=15):
    model.eval()
    trial_tgt_logits = defaultdict(list)
    trial_nontgt_logits = defaultdict(list)

    with torch.no_grad():
        for X_b, y_b, t_b in loader:
            with torch.amp.autocast(device_type=config.DEVICE.type):
                logits = model(X_b).squeeze(1).cpu().numpy()
            y_np = y_b.cpu().numpy()
            t_np = t_b.cpu().numpy()
            for logit, tgt, tid in zip(logits, y_np, t_np):
                tid = int(tid)
                if int(tgt) == 1:
                    trial_tgt_logits[tid].append(float(logit))
                else:
                    trial_nontgt_logits[tid].append(float(logit))

    y_true_trial, y_scores_trial = [], []
    hits, total = 0, 0
    for tid in trial_tgt_logits.keys():
        if not trial_tgt_logits[tid] or not trial_nontgt_logits[tid]:
            continue
        m_tgt = float(np.mean(trial_tgt_logits[tid]))
        m_nontgt = float(np.mean(trial_nontgt_logits[tid]))
        if m_tgt > m_nontgt:
            hits += 1
        total += 1
        y_true_trial.extend([1, 0])
        y_scores_trial.extend([m_tgt, m_nontgt])

    pairwise_acc = (hits / total * 100.0) if total > 0 else 0.0
    try:
        trial_auc = roc_auc_score(y_true_trial, y_scores_trial)
    except Exception:
        trial_auc = 0.5

    best_f1, best_th = 0.0, 0.0
    for th in np.linspace(np.min(y_scores_trial), np.max(y_scores_trial), 200):
        preds = (np.array(y_scores_trial) >= th).astype(int)
        score = f1_score(y_true_trial, preds, average='macro', zero_division=0)
        if score > best_f1:
            best_f1, best_th = score, th
            
    best_bal_acc = balanced_accuracy_score(y_true_trial, (np.array(y_scores_trial) >= best_th).astype(int)) * 100.0

    seq_accuracies = []
    seq_itrs = []
    for S in range(1, max_seq + 1):
        s_hits, s_total = 0, 0
        for tid in trial_tgt_logits.keys():
            tgt_slice = trial_tgt_logits[tid][:S]
            nontgt_slice = trial_nontgt_logits[tid][:5*S]
            if not tgt_slice or not nontgt_slice:
                continue
            s_total += 1
            if np.mean(tgt_slice) > np.mean(nontgt_slice):
                s_hits += 1
        s_acc = (s_hits / s_total) if s_total > 0 else 0.0
        s_itr = calculate_itr(s_acc, n_classes=2)
        seq_accuracies.append(s_acc * 100.0)
        seq_itrs.append(s_itr)

    return trial_auc, best_bal_acc, pairwise_acc, best_f1, seq_accuracies, seq_itrs
