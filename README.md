# BCI P300 Detection EEG Pipeline

This repository contains a modular Python pipeline for running Brain-Computer Interface (BCI) P300 detection experiments on the **BCIAUT-P300** EEG dataset. It includes our main spatial-temporal model with interpretability, alongside comparative baselines (EEGNet and P3CNET). All models reside in dedicated subfolders.

---

## Directory Structure

*   `datasets/bciaut/`: Raw EEG dataset files organized by subjects and sessions (e.g., `SBJ01/S01/`).
*   `models/bciaut/`: Saved model weights of the best cross-validation folds.
*   `results/bciaut/`: 
    *   Our model's CSV and interpretability JSON results.
    *   `results/bciaut/eegnet/`: Evaluation CSVs for the comparative `EEGNet` model.
    *   `results/bciaut/p3cnet/`: Evaluation CSVs for the comparative `P3CNET` model.
*   `plots/bciaut/`: Output target folder for visual performance plots.
*   `scripts/bciaut/`: Source code modules:
    *   `config.py`: Shared configurations, hyperparameters, and directory paths.
    *   `dataset.py`: Matlab file parsing, preprocessing, signal scaling, custom datasets, and real-time GPU data augmentations.
    *   `evaluation.py`: Shared metric calculations.
    *   `hst_p300/`: Subfolder for our model:
        *   `hst_p300/model.py`: Hybrid Spatial-Temporal network definition (`HST_P300Net`) and weight normalization.
        *   `hst_p300/training.py`: Smooth Balanced Focal Loss, training, and validation loops.
        *   `hst_p300/train.py`: Main training and saliency extraction runner.
    *   `eegnet/`: Subfolder for baseline model 1:
        *   `eegnet/model.py`: Baseline `EEGNet` model.
        *   `eegnet/train.py`: Baseline training pipeline runner (stratified 80/20 split, Balanced Batch Sampler, Cross Entropy loss).
    *   `p3cnet/`: Subfolder for baseline model 2:
        *   `p3cnet/model.py`: Baseline `P3CNET` model (Dağ et al.).
        *   `p3cnet/train.py`: Baseline training pipeline runner (stratified 80/20 split, weighted Cross Entropy loss, Max-Norm weight constraints, batch size 128).

---

## Environment Setup

Ensure Python 3.11+ is installed, and set up your virtual environment:

```bash
# Set up virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (defined in pyproject.toml)
pip install -e .
```

---

## Running the Pipelines

All pipelines can be executed from the command line:

### 1. Running Our Model (HST-P300Net with Saliency & Model Checkpointing)
```bash
# Default config (All subjects 1-15, seeds [42, 1234, 777], 200 epochs)
python scripts/bciaut/hst_p300/train.py

# Debug run: Subject 1, Seed 42, 2 epochs
python scripts/bciaut/hst_p300/train.py --subjects 1 --seeds 42 --epochs 2
```

### 2. Running Baseline 1 (EEGNet with Balanced Batch Sampler)
No checkpoints or interpretability maps are saved for this run; it strictly produces performance CSV files.
```bash
# Default config (All subjects 1-15, seeds [42, 1234, 777], 1000 epochs)
python scripts/bciaut/eegnet/train.py

# Debug run: Subject 1, Seed 42, 2 epochs
python scripts/bciaut/eegnet/train.py --subjects 1 --seeds 42 --epochs 2
```

### 3. Running Baseline 2 (P3CNET with Weighted Loss & Max-Norm Constraints)
No checkpoints or interpretability maps are saved for this run; it strictly produces performance CSV files.
```bash
# Default config (All subjects 1-15, seeds [42, 1234, 777], 1000 epochs)
python scripts/bciaut/p3cnet/train.py

# Debug run: Subject 1, Seed 42, 2 epochs
python scripts/bciaut/p3cnet/train.py --subjects 1 --seeds 42 --epochs 2
```

### Common CLI Arguments
*   `--subjects`: List of subject IDs to process (e.g., `--subjects 1 2 3`, defaults to subjects 1-15).
*   `--seeds`: Random seeds to evaluate (e.g., `--seeds 42 1234`, defaults to `[42, 1234, 777]`).
*   `--epochs`: Maximum training epochs per run (defaults to 200 for HST-P300Net, 1000 for baselines).
*   `--batch-size`: Batch size for training/inference (defaults to 64 for HST-P300Net/EEGNet, 128 for P3CNET).
*   `--patience`: Early stopping patience epochs based on validation loss (defaults to 20 for HST-P300Net, 50 for EEGNet, 45 for P3CNET).
