import os
import json
import numpy as np
import matplotlib.pyplot as plt
import mne
import importlib.util

import config

# Load GIST config dynamically
gist_config_path = os.path.join(config.ROOT_DIR, "scripts", "gist", "config.py")
spec = importlib.util.spec_from_file_location("gist_config", gist_config_path)
gist_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gist_config)

def standardize_chan_names(ch_list):
    mapping = {
        'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
        'FP1': 'Fp1', 'FP2': 'Fp2', 'AF3': 'AF3', 'AF4': 'AF4',
        'FC5': 'FC5', 'FC6': 'FC6', 'FC1': 'FC1', 'FC2': 'FC2',
        'CP5': 'CP5', 'CP6': 'CP6', 'CP1': 'CP1', 'CP2': 'CP2',
        'PO3': 'PO3', 'PO4': 'PO4', 'PO7': 'PO7', 'PO8': 'PO8'
    }
    return [mapping.get(c.upper(), c) for c in ch_list]

def generate_plot():
    json_abl = os.path.join(config.ROOT_DIR, "results", "gist", "msr_p300net_ablation", "msr_p300net_ablation_interpretability.json")
    json_main = os.path.join(config.ROOT_DIR, "results", "gist", "msr_p300net", "msr_p300net_interpretability.json")
    
    if not os.path.exists(json_abl) or not os.path.exists(json_main):
        print("Warning: Missing GIST interpretability results. Skipping Electrode Attention.")
        return
        
    with open(json_abl, "r") as f:
        data_abl = json.load(f)
    with open(json_main, "r") as f:
        data_main = json.load(f)
        
    configs = [4, 8, 16, 32]
    ch_lists = {
        4: gist_config.CHANNELS_4,
        8: gist_config.CHANNELS_8,
        16: gist_config.CHANNELS_16,
        32: gist_config.CHANNELS_32
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    
    im = None
    all_norm_weights = {}
    
    for idx, ch_count in enumerate(configs):
        ax = axes[idx]
        ch_names = standardize_chan_names(ch_lists[ch_count])
        
        weights_list = []
        if ch_count == 8:
            task_data = data_main.get("rsvp", data_main.get("p300_ch_8", {}))
        else:
            task_key = f"p300_ch_{ch_count}"
            task_data = data_abl.get(task_key, {})
            
        for subj in task_data:
            for seed in task_data[subj]:
                w = task_data[subj][seed].get("spatial_weights", [])
                if len(w) > 0:
                    weights_list.append(w)
                    
        if len(weights_list) == 0:
            avg_weights = np.ones(len(ch_names)) / len(ch_names)
        else:
            avg_weights = np.mean(weights_list, axis=0)
            
        w_min = np.min(avg_weights)
        w_max = np.max(avg_weights)
        if w_max != w_min:
            norm_weights = 2.0 * (avg_weights - w_min) / (w_max - w_min) - 1.0
        else:
            norm_weights = np.zeros_like(avg_weights)
            
        all_norm_weights[ch_count] = (ch_names, norm_weights)
        
        info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
        info.set_montage('standard_1020')
        
        im, _ = mne.viz.plot_topomap(norm_weights, info, axes=ax, show=False, cmap="RdBu_r", extrapolate="head")
        
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
                
        # Marker sizing adjusted per channel density
        marker_size = 180 if ch_count <= 8 else (120 if ch_count == 16 else 70)
        line_size = 20 if ch_count <= 8 else (14 if ch_count == 16 else 9)
        
        for collection in ax.collections:
            if type(collection).__name__ == 'PathCollection':
                collection.set_sizes([marker_size])
        for line in ax.lines:
            if len(line.get_xdata()) < 50:
                line.set_markersize(line_size)
                
        ax.set_title(f"{ch_count} Channels", fontsize=config.TITLE_SIZE, fontname='Times New Roman', fontweight='bold', pad=12)
        
    cbar_ax = fig.add_axes([0.92, 0.2, 0.018, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=config.TICK_SIZE)
    cbar.set_label('Attention Weight', fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        
    plt.subplots_adjust(wspace=0.25, right=0.90)
    save_pdf = os.path.join(config.PLOT_DIR_GIST, "gist_electrode_attention.pdf")
    save_png = os.path.join(config.PLOT_DIR_GIST, "gist_electrode_attention.png")
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

    # Save individual single-panel figures for each channel count
    for ch_count in configs:
        ch_names, norm_w = all_norm_weights[ch_count]
        info_s = mne.create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
        info_s.set_montage('standard_1020')
        
        fig_s, ax_s = plt.subplots(figsize=(7, 6))
        im_s, _ = mne.viz.plot_topomap(norm_w, info_s, axes=ax_s, show=False, cmap="RdBu_r", extrapolate="head")
        
        clip_circle_s = plt.Circle((0, 0), r, transform=ax_s.transData)
        ax_s.add_patch(clip_circle_s)
        clip_circle_s.set_visible(False)
        for image in ax_s.images:
            image.set_clip_path(clip_circle_s)
        for col in ax_s.collections:
            if type(col).__name__ in ['QuadMesh', 'PolyCollection', 'QuadContourSet']:
                col.set_clip_path(clip_circle_s)
                
        marker_size = 180 if ch_count <= 8 else (120 if ch_count == 16 else 70)
        line_size = 20 if ch_count <= 8 else (14 if ch_count == 16 else 9)
        for collection in ax_s.collections:
            if type(collection).__name__ == 'PathCollection':
                collection.set_sizes([marker_size])
        for line in ax_s.lines:
            if len(line.get_xdata()) < 50:
                line.set_markersize(line_size)
                
        ax_s.set_title(f"{ch_count} Channels Attention", fontsize=config.TITLE_SIZE, fontname='Times New Roman', fontweight='bold', pad=12)
        
        cbar_ax_s = fig_s.add_axes([0.88, 0.2, 0.035, 0.6])
        cbar_s = fig_s.colorbar(im_s, cax=cbar_ax_s)
        cbar_s.ax.tick_params(labelsize=config.TICK_SIZE)
        cbar_s.set_label('Attention Weight', fontsize=config.LABEL_SIZE, fontname='Times New Roman')
        for label in cbar_s.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            
        spdf = os.path.join(config.PLOT_DIR_GIST, f"gist_electrode_attention_{ch_count}ch.pdf")
        spng = os.path.join(config.PLOT_DIR_GIST, f"gist_electrode_attention_{ch_count}ch.png")
        plt.savefig(spdf, dpi=300, bbox_inches='tight')
        plt.savefig(spng, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generate_plot()
