import os
import torch

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(ROOT_DIR, "datasets/bciaut")
MODEL_DIR = os.path.join(ROOT_DIR, "models/bciaut")
RESULT_DIR = os.path.join(ROOT_DIR, "results/bciaut")
PLOT_DIR = os.path.join(ROOT_DIR, "plots/bciaut")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Hyperparameters
SUBJECT_IDS = list(range(1, 16))
SEEDS = [42, 1234, 777]  # 3 seeds as requested: [42, 1234, 777]
EPOCHS = 200
BATCH_SIZE = 64
STATIC_POS_WEIGHT = 5.0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
