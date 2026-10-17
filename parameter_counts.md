# Trainable Parameter Counts of All Models

This document lists the exact trainable parameter counts of all models across the **BCIAUT**, **GIB**, and **GIST** datasets. 

*Note: The `EEGNet` model used in the BCIAUT dataset and the `EEGNetv4` model used in the GIST dataset are structurally different. The former is a multi-class model (2 outputs for Cross-Entropy loss) while the latter is a binary regression model (1 logit for Binary Cross-Entropy / Focal loss). They are named distinctly as `EEGNet` and `EEGNetv4` respectively to avoid confusion.*

---

## 1. BCIAUT Dataset (Input: 8 Channels × 140 Time Samples)
These models are configured to predict 2 output classes (using standard Cross-Entropy Loss or multi-class logits).

| Model Name | Class Name | Parameter Count | Description |
| :--- | :--- | :---: | :--- |
| **MSR_P300NET (Ours)** | `MSR_P300NET` | **4,037** | Multi-scale spatial-temporal convolution with dual-attention heads. |
| **EEGNet** (Baseline) | `EEGNet` | **1,386** | Standard EEGNet with average pooling and 2 output units. |
| **P3CNET** (Baseline) | `P3CNET` | **4,338** | 3D convolutional network for P300 signals. |

---

## 2. GIB Dataset (Input: 8 Channels × 128 Time Samples)
These models are configured for binary classification (2 output classes or 1 logit with binary labels).

| Model Name | Class Name | Parameter Count | Description |
| :--- | :--- | :---: | :--- |
| **MSR_P300NET (Ours)** | `MSR_P300NET` | **4,037** | Multi-scale spatial-temporal convolution with dual-attention heads (1 logit). |
| **EEGInception** (Baseline) | `EEGInceptionPyTorch` | **15,334** | Multi-branch EEG-Inception PyTorch model (2 output classes). |
| **P3CNET** (Baseline) | `P3CNET` | **4,338** | 3D convolutional network for P300 signals (2 output classes). |

---

## 3. GIST Dataset (Input: Variable Channels × 128 Time Samples)
These models are configured for single-logit binary regression (using Binary Cross-Entropy / Focal Loss).

### 3.1 8-Channel Configuration (RSVP / P300 Speller Baseline)
| Model Name | Class Name | Parameter Count | Description |
| :--- | :--- | :---: | :--- |
| **MSR_P300NET (Ours)** | `MSR_P300NET` | **4,037** | Multi-scale spatial-temporal convolution (1 logit). |
| **EEGNetv4** (Baseline) | `EEGNetv4` | **1,297** | Lawhern's EEGNet v4 architecture (1 logit output). |
| **EEGInception** (Baseline) | `EEGInceptionPyTorch` | **15,093** | Multi-branch EEG-Inception PyTorch model (1 logit output). |

### 3.2 MSR_P300NET (Ours) Channel Configurations
Our model adapts to the number of input channels through its spatial filter dimensions. 

| Channels | Parameter Count |
| :---: | :---: |
| **4 Channels** | **3,941** |
| **8 Channels** (Baseline) | **4,037** |
| **16 Channels** | **4,229** |
| **32 Channels** | **4,613** |
