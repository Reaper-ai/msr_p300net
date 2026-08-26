import os
import torch

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(ROOT_DIR, "datasets/gib/GIB-UVA ERP-BCI.hdf5")
MODEL_DIR = os.path.join(ROOT_DIR, "models/gib")
RESULT_DIR = os.path.join(ROOT_DIR, "results/gib")

# Ensure target directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Hyperparameters
SUBJECT_IDS = list(range(73))  # 73 subjects (0 to 72)
SEEDS = [42, 1234, 777]  # seeds: [42, 1234, 777]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default values for argparse fallbacks
BATCH_SIZE = 64
EPOCHS = 1000

# Demographic Metadata Mapping
SUBJECT_METADATA = {
    # Dataset 0 (22 Healthy)
    **{i: {'dataset': 0, 'condition': 'Healthy'} for i in range(0, 22)},
    # Dataset 1 (10 Healthy: 22-31, 16 Pathology: 32-47)
    **{i: {'dataset': 1, 'condition': 'Healthy'} for i in range(22, 32)},
    **{i: {'dataset': 1, 'condition': 'Unhealthy'} for i in range(32, 48)},
    # Dataset 2 (10 Healthy: 48-57, 15 Multiple Sclerosis: 58-72)
    **{i: {'dataset': 2, 'condition': 'Healthy'} for i in range(48, 58)},
    **{i: {'dataset': 2, 'condition': 'Unhealthy'} for i in range(58, 73)}
}
