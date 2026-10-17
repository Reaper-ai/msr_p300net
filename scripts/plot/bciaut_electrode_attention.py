import os
import json
import numpy as np
import matplotlib.pyplot as plt
import mne

import config

def generate_plot():
    json_path = os.path.join(config.ROOT_DIR, "results", "bciaut", "msr_p300net", "interpretability_results.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(config.ROOT_DIR, "results", "bciaut", "interpretability_results.json")
    if not os.path.exists(json_path):
        print("Warning: Missing BCIAUT interpretability_results.json. Skipping Electrode Attention.")
        return
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # Correct BCIAUT 8 electrode channel labels
    ch_names = ['C3', 'Cz', 'C4', 'CPz', 'P3', 'Pz', 'P4', 'POz']
    
    # Collect weights across all subjects and seeds
    all_weights = []
    for subj in data.keys():
        for seed_key in data[subj].keys():
            weights = data[subj][seed_key]["spatial_weights"]
            if len(weights) > 0:
                all_weights.append(weights)
                
    if len(all_weights) == 0:
        print("Warning: No weights found in BCIAUT interpretability results.")
        return
        
    avg_weights = np.mean(all_weights, axis=0)
    # Normalize values from -1 to 1
    w_min = np.min(avg_weights)
    w_max = np.max(avg_weights)
    if w_max != w_min:
        norm_weights = 2.0 * (avg_weights - w_min) / (w_max - w_min) - 1.0
    else:
        norm_weights = np.zeros_like(avg_weights)
    
    info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
    info.set_montage('standard_1020')
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    im, _ = mne.viz.plot_topomap(norm_weights, info, axes=ax, show=False, cmap="RdBu_r", extrapolate="head")
    
    # Clip all colored contours/images strictly inside the head outline circle
    r = 0.095
    for line in ax.lines:
        x, y = line.get_data()
        if len(x) > 50:
            r = np.mean(np.sqrt(x**2 + y**2))
            break
    clip_circle = plt.Circle((0, 0), r, transform=ax.transData)
    ax.add_patch(clip_circle)
    clip_circle.set_visible(False)
    for image in ax.images:
        image.set_clip_path(clip_circle)
    for col in ax.collections:
        if type(col).__name__ in ['QuadMesh', 'PolyCollection', 'QuadContourSet']:
            col.set_clip_path(clip_circle)
            
    # Make electrode sensor marker dots distinct
    for collection in ax.collections:
        if type(collection).__name__ == 'PathCollection':
            collection.set_sizes([180])
    for line in ax.lines:
        if len(line.get_xdata()) < 50:
            line.set_markersize(22)
    
    cbar_ax = fig.add_axes([0.88, 0.2, 0.035, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=config.TICK_SIZE)
    cbar.set_label('Attention Weight', fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    save_pdf = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_electrode_attention.pdf")
    save_png = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_electrode_attention.png")
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

if __name__ == "__main__":
    generate_plot()
