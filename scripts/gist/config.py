import os
import torch

MAX_LR = 0.0026
WEIGHT_DECAY = 0.00004
GAMMA = 1.0000
POS_WEIGHT = 2.0000
SMOOTHING = 0.0500
PCT_START = 0.2000

DATA_PATH = '/home/gaurav/Programming/bci/datasets/gist'
FS = 128
T_MIN = -0.200
T_MAX = 0.800
N_SAMPLES = int(round((T_MAX - T_MIN) * FS))  # 128 samples (1.0s)
ALL_SUBJECTS = list(range(1, 56))

# Channel Configurations
CHANNELS_4 = ['Fz', 'Cz', 'Pz', 'Oz']
CHANNELS_8 = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'O1', 'O2']
CHANNELS_16 = ['F3', 'Fz', 'F4', 'FC1', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2', 'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']
CHANNELS_32 = ['FP1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'FP2', 'FZ', 'Cz']

EPOCHS = 150
PATIENCE = 30
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 1234, 777]

# Absolute Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "gist")
MODEL_DIR = os.path.join(BASE_DIR, "models", "gist")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
