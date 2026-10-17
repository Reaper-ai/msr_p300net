# BCI Model Training Specifications & Performance Metrics

This document details the training configurations, validation strategies, hyperparameters, and evaluation metrics used for **MSR_P300NET (Ours)**, **EEGNetv4**, **EEGInception**, and **P3CNET** models across the **BCIAUT**, **GIB**, and **GIST** datasets. 

---

## 1. Performance Metric Definitions

The following mathematical formulations and descriptions define the metrics used in the evaluation and methods sections of BCI spelling tasks.

### 1.1 Objective / Spelling Accuracy (%)
Objective Accuracy evaluates the spelling success rate at the character/selection level. 
*   **Definition**: The percentage of correctly predicted target characters out of the total trials.
*   **Formulation**:
    $$\text{Objective Accuracy (\%)} = \left( \frac{\text{Correctly Decoded Characters}}{\text{Total Characters Spelled}} \right) \times 100$$
*   **BCI Decoding Context**:
    *   **Single-Character Speller (BCIAUT)**: The user focuses on a target character choice among $N=8$ flashing positions. The model calculates the target probability for each choice. The decoded character is:
        $$\hat{c} = \arg\max_{c \in \{1, \dots, 8\}} \sum_{s=1}^{S} P(c | \text{Trial}_{c, s})$$
        where $S$ is the number of sequence repetitions.
    *   **Matrix Speller (GIST, GIB)**: The user focuses on a target character in a matrix grid (e.g., $6 \times 6$). Rows and columns flash. Let $P(r)$ be the aggregated target probability for row $r$, and $P(c)$ for column $c$. The decoded character coordinates are:
        $$\hat{r} = \arg\max_{r} \sum_{s=1}^{S} P(\text{Row}_r), \quad \hat{c} = \arg\max_{c} \sum_{s=1}^{S} P(\text{Col}_c)$$
        The predicted character is the intersection of $(\hat{r}, \hat{c})$.

### 1.2 Balanced Accuracy (%)
Used for single-trial target classification (target vs. non-target) to prevent biased evaluation due to extreme class imbalances.
*   **Definition**: The arithmetic mean of sensitivity (True Positive Rate, TPR) and specificity (True Negative Rate, TNR).
*   **Formulation**:
    $$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2} = \frac{\text{TPR} + \text{TNR}}{2}$$
    $$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{TNR} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

### 1.3 Macro F1-Score
F1-score measures classification precision and recall balance. Macro F1-score treats all classes (Target and Non-Target) with equal weight.
*   **Formulation**:
    $$\text{Macro F1} = \frac{\text{F1}_{\text{Target}} + \text{F1}_{\text{Non-Target}}}{2}$$
    $$\text{F1}_{c} = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
    $$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$

### 1.4 Area Under the ROC Curve (Trial AUC)
Trial AUC represents the Area Under the Receiver Operating Characteristic curve.
*   **Definition**: The probability that a randomly selected target trial is assigned a higher target probability by the model than a randomly selected non-target trial. It provides a threshold-independent measure of classifier discriminability.

### 1.5 Pairwise Accuracy (%)
Pairwise Accuracy evaluates binary classification on balanced trial pairs.
*   **Definition**: The percentage of trials where the average target classification probability is higher than the average non-target probability.
*   **Formulation**:
    For each trial $k$, let $\bar{p}_{\text{tgt}, k}$ be the mean predicted probability of target flashes and $\bar{p}_{\text{nontgt}, k}$ be the mean predicted probability of non-target flashes.
    $$\text{Pairwise Accuracy (\%)} = \frac{1}{K} \sum_{k=1}^{K} \mathbb{I}(\bar{p}_{\text{tgt}, k} > \bar{p}_{\text{nontgt}, k}) \times 100$$
    where $\mathbb{I}(\cdot)$ is the indicator function and $K$ is the total trials.

### 1.6 Information Transfer Rate (ITR)
ITR measures the communication speed in bits per selection and bits per minute.
*   **Bits per Selection ($B$)**: Computed via the Wolpaw ITR formulation:
    $$B = \log_2(N) + P \log_2(P) + (1-P)\log_2\left(\frac{1-P}{N-1}\right)$$
    where:
    *   $N$ is the number of possible choices ($N = 8$ for BCIAUT, $N = 36$ for GIST matrix speller).
    *   $P$ is the spelling/character accuracy fraction ($0 \le P \le 1$).
    *   *Constraint*: If $P \le 1/N$, ITR is set to $0.0$. If $P = 1.0$, $B = \log_2(N)$.
*   **Bits per Minute**:
    $$\text{ITR (bits/min)} = B \times S$$
    where $S$ is the spelling speed in characters/selections per minute.

---

## 2. Model Training Specifications & Hyperparameters

### 2.1 BCIAUT Dataset
*   **Signal Parameters**: 8 EEG Channels, 128 Hz sampling rate. Epoch duration: $[-200, 800]$ ms ($140$ time samples).
*   **Preprocessing**: Baseline correction subtracted the mean of the pre-stimulus interval $[-200, 0]$ ms.
*   **Cohort Split**: Independent test set (sessions 4–7), calibration pool (sessions 1–3).

#### MSR_P300NET (Ours) — BCIAUT
*   **Architecture**: Multi-scale spatial-temporal convolution with dual-attention heads.
*   **Optimizer**: AdamW.
    *   Base Learning Rate: $5 \times 10^{-4}$
    *   Weight Decay: $1 \times 10^{-2}$
*   **LR Scheduler**: OneCycleLR.
    *   Maximum Learning Rate: $5 \times 10^{-3}$
    *   Percentage Start (`pct_start`): $0.3$
*   **Loss Function**: SmoothBalancedFocalLoss.
    *   Focal Gamma ($\gamma$): $2.5$
    *   Positive Weight Bias: $5.0$
    *   Label Smoothing: $0.05$
*   **Batch Size**: 64.
*   **Early Stopping**: Patience limit of $20$ epochs based on validation loss. Max epochs: $200$.
*   **Validation Strategy**: 5-fold Cross-Validation grouped by blocks (all samples within a block stay in the same split; blocks are identified by group IDs `g_pool // 100`).
*   **Calibration**: Logistic regression calibrator fit on the validation predictions of the best cross-validation fold.
*   **Constraint/Regularization**: Dropout and mixed-precision training.

#### EEGNetv4 — BCIAUT
*   **Architecture**: Compact CNN tailored for EEG (Depthwise & Separable convolutions).
    *   Dropout Rate: $0.25$
*   **Optimizer**: Adam.
    *   Base Learning Rate: $5 \times 10^{-4}$
    *   Betas: $(0.9, 0.999)$, $\epsilon = 1 \times 10^{-8}$
*   **Loss Function**: CrossEntropyLoss.
*   **Batching**: BalancedBatchSampler (samples equal target and non-target trials to form a balanced batch).
*   **Batch Size**: 64.
*   **Early Stopping**: Patience limit of $20$ epochs based on validation loss. Max epochs: $200$.
*   **Validation Strategy**: Stratified 80/20 train/validation split (20% of positive and 20% of negative trials shuffled for validation).
*   **Constraint/Regularization**: Gradient clipping (`max_norm = 5.0`).

#### P3CNET — BCIAUT
*   **Architecture**: 3D convolutional network for P300 signals.
*   **Optimizer**: Adam.
    *   Base Learning Rate: $5 \times 10^{-4}$
    *   Betas: $(0.9, 0.999)$, $\epsilon = 1 \times 10^{-8}$
*   **Loss Function**: CrossEntropyLoss with weighted classes.
    *   Class Weights: $[1.0, 7.0]$ (target class penalty is scaled by 7 to penalize false negatives).
*   **Batch Size**: 64 (standard shuffling dataloader).
*   **Early Stopping**: Patience limit of $20$ epochs based on validation loss. Max epochs: $200$.
*   **Validation Strategy**: Stratified 80/20 train/validation split.
*   **Constraint/Regularization**: Strict Max-Norm kernel constraints:
    *   First convolutional kernel (`conv1`): Maximum $L_2$-norm constrained to $\le 1.0$.
    *   Fully connected layer (`fc`): Maximum $L_2$-norm constrained to $\le 0.25$.

---

### 2.2 GIB Dataset
*   **Signal Parameters**: 8 EEG Channels, 128 Hz sampling rate. Epoch duration: $[0, 1000]$ ms ($128$ time samples).
*   **Preprocessing**: Subject-wise Z-Score normalization computed strictly from training data.
*   **Cohort Split**: 
    *   **Calibration Set**: First 2 runs for Datasets 0 & 1, first 1 run for Dataset 2.
    *   **Test Set**: Remaining runs.

#### MSR_P300NET (Ours) — GIB
*   **Optimizer**: AdamW.
    *   Base Learning Rate: $5 \times 10^{-4}$
    *   Weight Decay: $1 \times 10^{-3}$
*   **LR Scheduler**: OneCycleLR.
    *   Maximum Learning Rate: $5 \times 10^{-3}$
    *   Percentage Start (`pct_start`): $0.2$
*   **Loss Function**: SmoothBalancedFocalLoss.
    *   Focal Gamma ($\gamma$): $2.0$
    *   Positive Weight Bias: $5.0$
    *   Label Smoothing: $0.05$
*   **Batch Size**: 64.
*   **Early Stopping**: Patience limit of $25$ epochs based on validation loss (minimum change delta = $1 \times 10^{-4}$). Max epochs: $500$.
*   **Validation Strategy**: Character-level 80/20 train/validation split on the calibration runs (grouped by character ID to prevent leakage).
*   **Data Augmentation**: `SafeEEGAugmenter` applied during training with $p=0.5$:
    *   Scale scaling: uniform range $[0.9, 1.1]$
    *   Gaussian noise addition: $\sigma = 0.05$
    *   Channel dropout: dropout probability = $0.1$
    *   Temporal rolling shifts: $[-2, 2]$ samples (zero-filled bounds)
*   **Constraint/Regularization**: Gradient clipping (`max_norm = 5.0`), Max-Norm constraint on spatial-temporal weights.

#### EEGInception — GIB
*   **Architecture**: Multi-branch inception CNN designed for EEG.
    *   Filters per Branch: 8
    *   Dropout Rate: $0.25$
*   **Optimizer**: Adam.
    *   Base Learning Rate: $1 \times 10^{-3}$
    *   Betas: $(0.9, 0.999)$, $\epsilon = 1 \times 10^{-8}$
*   **Loss Function**: CrossEntropyLoss.
*   **Batching**: BalancedBatchSampler (equal target and non-target ratio).
*   **Batch Size**: 64.
*   **Early Stopping**: Patience limit of $25$ epochs based on validation loss. Max epochs: $1000$.
*   **Validation Strategy**: Character-level 80/20 train/validation split on calibration runs.
*   **Constraint/Regularization**: Gradient clipping (`max_norm = 5.0`).

#### P3CNET — GIB
*   **Optimizer**: Adam.
    *   Base Learning Rate: $5 \times 10^{-4}$
    *   Betas: $(0.9, 0.999)$, $\epsilon = 1 \times 10^{-8}$
*   **Loss Function**: CrossEntropyLoss with weighted classes.
    *   Class Weights: $[1.0, 7.0]$
*   **Batch Size**: 64 (standard shuffling dataloader).
*   **Early Stopping**: Patience limit of $20$ epochs based on validation loss. Max epochs: $1000$.
*   **Validation Strategy**: Character-level 80/20 train/validation split on calibration runs.
*   **Constraint/Regularization**: Strict Max-Norm constraints on convolutional kernels (`conv1` $\le 1.0$) and linear outputs (`fc` $\le 0.25$).

---

### 2.3 GIST Dataset
*   **Signal Parameters**: EEG channels range by configuration (4, 8, 16, or 32 channels), 128 Hz sampling rate. Epoch duration: $[-200, 800]$ ms ($128$ time samples).
*   **Preprocessing**: Bandpass filtering $[0.5, 30.0]$ Hz (FIR filter using `firwin`). Resampled to $128$ Hz. Baseline correction subtracted the mean of the $[-200, 0]$ ms pre-stimulus interval.
*   **Cohort Split**: 
    *   **RSVP Task**: First 50% of sequences are calibration runs; remaining 50% of sequences are test runs.
    *   **P300 Speller Task**: Maximum of 2 calibration training runs; maximum of 4 independent test runs.

#### Shared Specifications for All Models on GIST
All evaluated models on the GIST dataset (**MSR_P300NET (Ours)**, **EEGNetv4**, and **EEGInception**) share the same training pipeline and hyperparameters:
*   **Optimizer**: AdamW.
    *   Base Learning Rate: $1.04 \times 10^{-4}$ ($\text{MAX\_LR} / 25.0$)
    *   Weight Decay: $4 \times 10^{-5}$
*   **LR Scheduler**: OneCycleLR.
    *   Maximum Learning Rate: $2.6 \times 10^{-3}$
    *   Percentage Start (`pct_start`): $0.2$
*   **Loss Function**: ParametricFocalLoss.
    *   Focal Gamma ($\gamma$): $1.0$
    *   Positive Weight Bias: $2.0$
    *   Label Smoothing: $0.05$
*   **Batch Size**: 64.
*   **Early Stopping**: Patience limit of $30$ epochs based on validation loss. Max epochs: $150$.
*   **Validation Strategy**: 80/20 train/validation split on calibration runs:
    *   **RSVP**: Split is grouped sequence-wise.
    *   **P300**: Split is grouped trial-wise.
*   **Data Augmentation**: `SafeEEGAugmenter` applied during training with $p=0.5$:
    *   Scale scaling: uniform range $[0.9, 1.1]$
    *   Gaussian noise addition: $\sigma = 0.05$
    *   Channel dropout: dropout probability = $0.1$
    *   Temporal rolling shifts: $[-2, 2]$ samples (zero-filled bounds)
*   **Constraint/Regularization**: Gradient clipping (`max_norm = 5.0`), Max-Norm constraint on spatial-temporal weights.
