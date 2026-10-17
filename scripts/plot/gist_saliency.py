import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config

def generate_plot():
    json_path = os.path.join(config.ROOT_DIR, "results", "gist", "msr_p300net", "msr_p300net_interpretability.json")
    if not os.path.exists(json_path):
        print("Warning: Missing GIST interpretability results. Skipping saliency maps.")
        return
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # Correct GIST 8 electrode channel labels
    channels = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'O1', 'O2']
    
    task_key = "p300_ch_8"
    if task_key not in data:
        keys = list(data.keys())
        if len(keys) > 0:
            task_key = keys[0]
        else:
            print("Warning: GIST interpretability file has no keys. Skipping saliency maps.")
            return
            
    subj_data = data[task_key]
    all_saliencies = []
    for subj in subj_data.keys():
        for seed_key in subj_data[subj].keys():
            saliency = np.array(subj_data[subj][seed_key]["gradient_saliency"])
            if len(saliency) > 0:
                all_saliencies.append(saliency)
                
    if len(all_saliencies) == 0:
        print("Warning: No saliency maps found in GIST interpretability results.")
        return
        
    avg_saliency = np.mean(all_saliencies, axis=0)  # Shape: (8, 128)
    
    # Normalize to [0, 1] range
    s_min = np.min(avg_saliency)
    s_max = np.max(avg_saliency)
    if s_max > s_min:
        avg_saliency = (avg_saliency - s_min) / (s_max - s_min)
    else:
        avg_saliency = np.zeros_like(avg_saliency)
        
    # Calculate time array in milliseconds from -200ms to 800ms
    times_ms = np.linspace(-200, 800, 128)
    tick_vals = [-200, 0, 200, 400, 600, 800]
    tick_indices = [np.argmin(np.abs(times_ms - val)) + 0.5 for val in tick_vals]
    
    fig, ax = plt.subplots(figsize=(11, 5))
    
    cbar_kws = {'label': 'Normalized Saliency'}
    hm = sns.heatmap(avg_saliency, ax=ax, cmap="viridis", vmin=0.0, vmax=1.0, cbar=True,
                     yticklabels=channels, xticklabels=False, cbar_kws=cbar_kws)
                
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_vals, fontsize=config.TICK_SIZE, fontname='Times New Roman')
    ax.set_yticklabels(channels, rotation=0, fontsize=config.TICK_SIZE, fontname='Times New Roman')
    
    ax.set_ylabel("Electrode Channels", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    ax.set_xlabel("Time (ms)", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    
    # Format colorbar font
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=config.TICK_SIZE)
    cbar.set_label('Normalized Saliency', fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    plt.tight_layout()
    save_pdf = os.path.join(config.PLOT_DIR_GIST, "gist_saliency_maps.pdf")
    save_png = os.path.join(config.PLOT_DIR_GIST, "gist_saliency_maps.png")
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

if __name__ == "__main__":
    generate_plot()
