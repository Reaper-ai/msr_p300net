# BCI P300 Detection EEG Pipelines

This repository contains modular Python pipelines for running Brain-Computer Interface (BCI) P300 detection experiments on the **BCIAUT-P300** and **GIB-UVA ERP-BCI** EEG datasets. It includes spatial-temporal models alongside baseline comparison models.

---

## Directory Structure

*   `datasets/`:
    *   `datasets/bciaut/`: Raw BCIAUT dataset files organized by subjects and sessions.
    *   `datasets/gib/`: Raw GIB HDF5 dataset file (`GIB-UVA ERP-BCI.hdf5`).
*   `models/`:
    *   `models/bciaut/`: Saved model weights of the best cross-validation folds for BCIAUT.
    *   `models/gib/`: Saved model weights for the GIB MSR_P300NET model.
*   `results/`:
    *   `results/bciaut/`: CSV and JSON results for BCIAUT model and baselines.
    *   `results/gib/`:
        *   `results/gib/eeginception/`: Evaluation CSVs for the GIB EEGInception baseline.
        *   `results/gib/p3cnet/`: Evaluation CSVs for the GIB P3CNET baseline.
        *   `results/gib/msr_p300net/`: Evaluation CSVs and JSON interpretability maps for the GIB MSR_P300NET model.
*   `scripts/`:
    *   `scripts/bciaut/`: Pipeline code modules for the BCIAUT dataset:
        *   `eegnet/`: EEGNet baseline model and training orchestrator.
        *   `p3cnet/`: P3CNET baseline model and training orchestrator.
        *   `msr_p300net/`: Rebranded MSR_P300NET model, training orchestrator, and interpretability map extractor.
    *   `scripts/gib/`: Pipeline code modules for the GIB dataset:
        *   `config.py`: Shared GIB metadata and configurations.
        *   `dataset.py`: HDF5 loader and device data loading routines.
        *   `evaluation.py`: Unified evaluation and ITR/accuracy calculations for GIB.
        *   `eeginception/`: EEGInception baseline model and training orchestrator.
        *   `p3cnet/`: P3CNET baseline model and training orchestrator.
        *   `msr_p300net/`: Rebranded MSR_P300NET model, training orchestrator, and interpretability map extractor.

---

## Environment Setup

Ensure Python 3.11+ is installed, and set up your virtual environment:

```bash
# Set up virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
uv add h5py
pip install -e .
```

---

## Running GIB Dataset Pipelines

All pipelines can be executed from the command line:

### 1. EEGInception Baseline (GIB)
No checkpoints or interpretability maps are saved for this run; it strictly produces performance CSV files.
```bash
# Default config (All subjects 0-72, seeds [42, 1234, 777], 1000 epochs)
python scripts/gib/eeginception/train.py

# Debug run: Subject 0, Seed 42, 2 epochs (no-save dry-run)
python scripts/gib/eeginception/train.py --subjects 0 --seeds 42 --epochs 2 --no-save
```

### 2. P3CNET Baseline (GIB)
No checkpoints or interpretability maps are saved for this run; it strictly produces performance CSV files.
```bash
# Default config (All subjects 0-72, seeds [42, 1234, 777], 1000 epochs)
python scripts/gib/p3cnet/train.py

# Debug run: Subject 0, Seed 42, 2 epochs (no-save dry-run)
python scripts/gib/p3cnet/train.py --subjects 0 --seeds 42 --epochs 2 --no-save
```

### 3. MSR_P300NET Model (GIB)
Saves best model weights (under `models/gib/`) and interpretability maps (under `results/gib/msr_p300net/interpretability_results.json`).
```bash
# Default config (All subjects 0-72, seeds [42, 1234, 777], 200 epochs)
python scripts/gib/msr_p300net/train.py

# Debug run: Subject 0, Seed 42, 2 epochs (no-save dry-run)
python scripts/gib/msr_p300net/train.py --subjects 0 --seeds 42 --epochs 2 --no-save
```

### Common GIB CLI Arguments
*   `--subjects`: List of subject IDs to process (e.g., `--subjects 0 1 2`, defaults to subjects 0-72).
*   `--seeds`: Random seeds to evaluate (e.g., `--seeds 42 1234`, defaults to `[42, 1234, 777]`).
*   `--epochs`: Maximum training epochs per run (defaults to 1000 for baselines, 200 for MSR_P300NET).
*   `--batch-size`: Batch size for training/inference (defaults to 64 for eeginception/msr_p300net, 128 for p3cnet).
*   `--patience`: Early stopping patience epochs based on validation loss (defaults to 50 for eeginception, 45 for p3cnet, 25 for msr_p300net).
*   `--no-save`: Dry-run mode, disables saving checkpoints, CSV results, or JSON interpretability maps to disk.



Citation :

V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung,
and B. J. Lance, “EEGNet: a compact convolutional neural network for
EEG-based brain–computer interfaces,” Journal of Neural Engineering,
vol. 15, no. 5, p. 056013, 2018
E. Santamarı́a-Vázquez, V. Martı́nez-Cagigal, F. Vaquerizo-Villar, and
R. Hornero, “EEG-Inception: A novel deep convolutional neural network
for assistive ERP-based brain-computer interfaces,” IEEE Transactions
on Neural Systems and Rehabilitation Engineering, vol. 28, no. 12, pp.
2773–2782, 2020
D. Borra, S. Fantozzi, and E. Magosso, “Convolutional neural network
for a P300 brain-computer interface to improve social attention in autistic
spectrum disorder,” in XV Mediterranean Conference on Medical and
Biological Engineering and Computing – MEDICON 2019, J. Henriques,
N. Neves, and P. de Carvalho, Eds. Cham: Springer, 2020, pp. 1837–
1843.
I. Da, L. G. Dui, S. Ferrante, A. Pedrocchi, and A. Antonietti, “Lever-
aging deep learning techniques to improve P300-based brain computer
interfaces,” IEEE Journal of Biomedical and Health Informatics, vol. 26,
no. 10, pp. 4892–4902, 2022.