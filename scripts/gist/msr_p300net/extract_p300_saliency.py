#!/usr/bin/env python3
"""Inference-only extraction of P300-speller (8-channel) saliency for MSR-P300Net.

Loads the already-saved 8-channel P300 checkpoints, runs a single forward/backward
pass on target-class test trials, and merges the results into
results/gist/msr_p300net/msr_p300net_interpretability.json under the key
"p300_ch_8" (the existing "rsvp" key is left untouched). No training.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from dataset import load_p300_splits
from model import MSR_P300NET

SEEDS = [42, 1234, 777]
SUBJECTS = list(range(1, 56))
TASK_KEY = "p300_ch_8"

CKPT_DIRS = [
    os.path.join(config.MODEL_DIR, "msr_p300net"),
    os.path.join(config.BASE_DIR, "models", "models", "gist", "msr_p300net"),
]
JSON_PATH = os.path.join(config.OUTPUT_DIR, "msr_p300net", "msr_p300net_interpretability.json")


def find_ckpt(subj, seed):
    fname = f"subj_{subj:02d}_channels_8_seed_{seed}_best.pth"
    for d in CKPT_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def main():
    interpretability = {}
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            interpretability = json.load(f)
    if TASK_KEY not in interpretability:
        interpretability[TASK_KEY] = {}

    n_done, n_missing = 0, 0
    for subj in SUBJECTS:
        subj_key = f"SBJ{subj:02d}"
        splits = None
        loaded = False
        for seed in SEEDS:
            seed_key = f"seed_{seed}"
            if subj_key in interpretability[TASK_KEY] and \
               seed_key in interpretability[TASK_KEY][subj_key] and \
               interpretability[TASK_KEY][subj_key][seed_key].get("gradient_saliency"):
                n_done += 1
                continue

            ckpt = find_ckpt(subj, seed)
            if ckpt is None:
                print(f"[MISS] no checkpoint for S{subj:02d} seed {seed}")
                n_missing += 1
                continue

            if not loaded:
                splits = load_p300_splits(subj, channels_to_use=config.CHANNELS_8)
                if splits is None:
                    print(f"[MISS] no P300 data for S{subj:02d}")
                    n_missing += len(SEEDS)
                    break
                (_, _, _), (_, _, _), (X_te, y_te, _) = splits
                loaded = True

            model = MSR_P300NET(n_channels=8, n_samples=config.N_SAMPLES).to(config.DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
            model.eval()

            spatial_w = model.conv_s.weight.detach().abs().mean(dim=(0, 1, 3)).cpu().numpy().tolist()

            target_mask = (y_te == 1)
            if target_mask.any():
                X_t = torch.tensor(X_te[target_mask], dtype=torch.float32, device=config.DEVICE)
                X_t.requires_grad = True
                with torch.enable_grad():
                    with torch.amp.autocast(device_type=config.DEVICE.type):
                        logits = model(X_t)
                    logits.backward(torch.ones_like(logits))
                sal = X_t.grad.detach().abs().mean(dim=0).cpu().numpy().tolist()
            else:
                sal = []

            interpretability[TASK_KEY].setdefault(subj_key, {})[seed_key] = {
                "spatial_weights": spatial_w,
                "temporal_attention": [],
                "gradient_saliency": sal,
            }
            with open(JSON_PATH, "w") as f:
                json.dump(interpretability, f, indent=4)
            n_done += 1
            print(f"[OK] S{subj:02d} seed {seed}  sal shape {np.array(sal).shape}")

        del splits
        import gc
        gc.collect()

    print(f"\nDone: {n_done} extracted/present, {n_missing} missing.")


if __name__ == "__main__":
    main()
