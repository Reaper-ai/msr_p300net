import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

import config

def generate_plot():
    json_path = os.path.join(config.ROOT_DIR, "results", "gib", "msr_p300net", "interpretability_results.json")
    info_path = os.path.join(config.ROOT_DIR, "datasets", "gib", "info.txt")
    
    if not os.path.exists(json_path):
        print("Warning: Missing GIB interpretability_results.json. Skipping Electrode Attention.")
        return
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # Read subject metadata mapping from info.txt
    if os.path.exists(info_path):
        df_info = pd.read_csv(info_path, sep=';', encoding='latin1')
        healthy_subjs = df_info[df_info['Condition'] == 'Healthy']['Subject ID'].tolist()
        ms_subjs = df_info[df_info['Condition'] == 'Multiple sclerosis']['Subject ID'].tolist()
    else:
        # Fallback based on dataset indices if info.txt missing
        healthy_subjs = list(range(0, 32)) + list(range(48, 58))
        ms_subjs = list(range(58, 73))
        
    # Correct GIB 8 electrode channel labels
    ch_names = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'PO7', 'PO8']
    
    healthy_weights = []
    for s_idx in healthy_subjs:
        s_key = f"SBJ{s_idx:02d}"
        if s_key in data:
            for seed in data[s_key]:
                w = data[s_key][seed].get("spatial_weights", [])
                if len(w) > 0:
                    healthy_weights.append(w)
                    
    ms_weights = []
    for s_idx in ms_subjs:
        s_key = f"SBJ{s_idx:02d}"
        if s_key in data:
            for seed in data[s_key]:
                w = data[s_key][seed].get("spatial_weights", [])
                if len(w) > 0:
                    ms_weights.append(w)
                    
    if len(healthy_weights) == 0 or len(ms_weights) == 0:
        print("Warning: Missing weights for Healthy or Multiple Sclerosis cohort.")
        return
        
    avg_healthy = np.mean(healthy_weights, axis=0)
    avg_ms = np.mean(ms_weights, axis=0)
    
    # Normalize values from -1 to 1 across both
    combined = np.concatenate([avg_healthy, avg_ms])
    w_min, w_max = np.min(combined), np.max(combined)
    
    if w_max != w_min:
        norm_healthy = 2.0 * (avg_healthy - w_min) / (w_max - w_min) - 1.0
        norm_ms = 2.0 * (avg_ms - w_min) / (w_max - w_min) - 1.0
    else:
        norm_healthy = np.zeros_like(avg_healthy)
        norm_ms = np.zeros_like(avg_ms)
        
    info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
    info.set_montage('standard_1020')
    
    # 2-subplot figure (Healthy vs Multiple Sclerosis)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    
    cohort_data = [
        ("Healthy Cohort", norm_healthy, axes[0]),
        ("Unhealthy (Multiple Sclerosis)", norm_ms, axes[1])
    ]
    
    im = None
    for title, norm_w, ax in cohort_data:
        im, _ = mne.viz.plot_topomap(norm_w, info, axes=ax, show=False, cmap="RdBu_r", extrapolate="head")
        
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
                
        ax.set_title(title, fontsize=config.TITLE_SIZE, fontname='Times New Roman', fontweight='bold', pad=12)
        
    cbar_ax = fig.add_axes([0.92, 0.2, 0.025, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=config.TICK_SIZE)
    cbar.set_label('Attention Weight', fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        
    plt.subplots_adjust(wspace=0.3, right=0.90)
    save_pdf = os.path.join(config.PLOT_DIR_GIB, "gib_electrode_attention.pdf")
    save_png = os.path.join(config.PLOT_DIR_GIB, "gib_electrode_attention.png")
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

    # Also save separate single-panel figures for Healthy and Unhealthy MS
    for title, norm_w, fname_suffix in [
        ("Healthy Cohort", norm_healthy, "healthy"),
        ("Unhealthy (Multiple Sclerosis)", norm_ms, "unhealthy_ms")
    ]:
        fig_s, ax_s = plt.subplots(figsize=(7, 6))
        im_s, _ = mne.viz.plot_topomap(norm_w, info, axes=ax_s, show=False, cmap="RdBu_r", extrapolate="head")
        clip_circle_s = plt.Circle((0, 0), r, transform=ax_s.transData)
        ax_s.add_patch(clip_circle_s)
        clip_circle_s.set_visible(False)
        for image in ax_s.images:
            image.set_clip_path(clip_circle_s)
        for col in ax_s.collections:
            if type(col).__name__ in ['QuadMesh', 'PolyCollection', 'QuadContourSet']:
                col.set_clip_path(clip_circle_s)
        for collection in ax_s.collections:
            if type(collection).__name__ == 'PathCollection':
                collection.set_sizes([180])
        for line in ax_s.lines:
            if len(line.get_xdata()) < 50:
                line.set_markersize(22)
        ax_s.set_title(title, fontsize=config.TITLE_SIZE, fontname='Times New Roman', fontweight='bold', pad=12)
        
        cbar_ax_s = fig_s.add_axes([0.88, 0.2, 0.035, 0.6])
        cbar_s = fig_s.colorbar(im_s, cax=cbar_ax_s)
        cbar_s.ax.tick_params(labelsize=config.TICK_SIZE)
        cbar_s.set_label('Attention Weight', fontsize=config.LABEL_SIZE, fontname='Times New Roman')
        for label in cbar_s.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            
        spdf = os.path.join(config.PLOT_DIR_GIB, f"gib_electrode_attention_{fname_suffix}.pdf")
        spng = os.path.join(config.PLOT_DIR_GIB, f"gib_electrode_attention_{fname_suffix}.png")
        plt.savefig(spdf, dpi=300, bbox_inches='tight')
        plt.savefig(spng, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generate_plot()
