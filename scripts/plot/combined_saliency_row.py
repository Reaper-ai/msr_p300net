import os
import json
import numpy as np
import matplotlib.pyplot as plt

import config

def generate_plot():
    # 1. BCIAUT Saliency
    f_bciaut = os.path.join(config.ROOT_DIR, "results", "bciaut", "msr_p300net", "interpretability_results.json")
    if not os.path.exists(f_bciaut):
        f_bciaut = os.path.join(config.ROOT_DIR, "results", "bciaut", "interpretability_results.json")
    
    sal_bci = []
    if os.path.exists(f_bciaut):
        with open(f_bciaut) as f:
            d_bci = json.load(f)
        for s in d_bci:
            for seed in d_bci[s]:
                sal = np.array(d_bci[s][seed].get('gradient_saliency', []))
                if len(sal) > 0:
                    sal_bci.append(sal)
                    
    if len(sal_bci) > 0:
        mean_bci = np.mean(sal_bci, axis=0)  # Shape (8, 140)
        times_bci = np.linspace(-100, 1000, mean_bci.shape[1])
        # Trim time strictly <= 1000 ms
        mask_bci = times_bci <= 1000
        mean_bci = mean_bci[:, mask_bci]
        times_bci = times_bci[mask_bci]
        s_min, s_max = mean_bci.min(), mean_bci.max()
        norm_bci = (mean_bci - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(mean_bci)
    else:
        norm_bci = np.zeros((8, 128))
        times_bci = np.linspace(-100, 1000, 128)

    # 2. GIB Healthy, Multiple Sclerosis, and Other Unhealthy Saliency
    f_gib = os.path.join(config.ROOT_DIR, "results", "gib", "msr_p300net", "interpretability_results.json")
    sal_healthy, sal_ms, sal_other = [], [], []
    healthy_ids = list(range(0, 32)) + list(range(48, 58))
    ms_ids = list(range(58, 73))
    other_ids = list(range(32, 48))
    
    if os.path.exists(f_gib):
        with open(f_gib) as f:
            d_gib = json.load(f)
        for s_idx in healthy_ids:
            k = f"SBJ{s_idx:02d}"
            if k in d_gib:
                for seed in d_gib[k]:
                    s_arr = np.array(d_gib[k][seed].get('gradient_saliency', []))
                    if s_arr.ndim == 3:
                        s_arr = s_arr.mean(axis=0)
                    if len(s_arr) > 0:
                        sal_healthy.append(s_arr)
                        
        for s_idx in ms_ids:
            k = f"SBJ{s_idx:02d}"
            if k in d_gib:
                for seed in d_gib[k]:
                    s_arr = np.array(d_gib[k][seed].get('gradient_saliency', []))
                    if s_arr.ndim == 3:
                        s_arr = s_arr.mean(axis=0)
                    if len(s_arr) > 0:
                        sal_ms.append(s_arr)

        for s_idx in other_ids:
            k = f"SBJ{s_idx:02d}"
            if k in d_gib:
                for seed in d_gib[k]:
                    s_arr = np.array(d_gib[k][seed].get('gradient_saliency', []))
                    if s_arr.ndim == 3:
                        s_arr = s_arr.mean(axis=0)
                    if len(s_arr) > 0:
                        sal_other.append(s_arr)

    def normalize_matrix(mat):
        mi, ma = mat.min(), mat.max()
        return (mat - mi) / (ma - mi) if ma > mi else np.zeros_like(mat)

    norm_gib_healthy = normalize_matrix(np.mean(sal_healthy, axis=0)) if len(sal_healthy) > 0 else np.zeros((8, 128))
    norm_gib_ms = normalize_matrix(np.mean(sal_ms, axis=0)) if len(sal_ms) > 0 else np.zeros((8, 128))
    norm_gib_other = normalize_matrix(np.mean(sal_other, axis=0)) if len(sal_other) > 0 else np.zeros((8, 128))
    times_gib = np.linspace(0, 1000, 128)

    # 3. GIST Saliency - P300 speller task only (key p300_ch_8); RSVP is excluded
    f_gist = os.path.join(config.ROOT_DIR, "results", "gist", "msr_p300net", "msr_p300net_interpretability.json")
    sal_gist = []
    if os.path.exists(f_gist):
        with open(f_gist) as f:
            d_gist = json.load(f)
        if 'p300_ch_8' in d_gist:
            subj_data = d_gist['p300_ch_8']
            for s in subj_data:
                for seed in subj_data[s]:
                    s_arr = np.array(subj_data[s][seed].get('gradient_saliency', []))
                    if len(s_arr) > 0:
                        sal_gist.append(s_arr)
        else:
            print("Warning: p300_ch_8 key not found in GIST interpretability JSON; leaving GIST panel blank.")
    else:
        print("Warning: GIST interpretability JSON not found; leaving GIST panel blank.")

    norm_gist = normalize_matrix(np.mean(sal_gist, axis=0)) if len(sal_gist) > 0 else np.zeros((8, 128))
    times_gist = np.linspace(-200, 800, 128)

    # Plot Layout: 1 Row, 5 Subplots, Full Page Wide (7.16 inches)
    fig, axes = plt.subplots(1, 5, figsize=(config.IEEE_DOUBLE_COL_WIDTH, 1.85), sharey=True)

    panel_configs = [
        ("(a) BCIAUT", norm_bci, times_bci, [-100, 0, 300, 1000], axes[0]),
        ("(b) GIB (Healthy)", norm_gib_healthy, times_gib, [0, 300, 1000], axes[1]),
        ("(c) GIB (MS)", norm_gib_ms, times_gib, [0, 300, 1000], axes[2]),
        ("(d) GIB (Other)", norm_gib_other, times_gib, [0, 300, 1000], axes[3]),
        ("(e) GIST", norm_gist, times_gist, [0, 300, 800], axes[4])
    ]

    im = None
    for idx, (title, data_mat, times, x_ticks_vals, ax) in enumerate(panel_configs):
        im = ax.imshow(data_mat, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0, origin='upper',
                       extent=[times[0], times[-1], 8.5, 0.5])
        
        # Stimulus onset reference line at t=0
        if times[0] < 0:
            ax.axvline(0, color='white', linestyle='--', linewidth=0.6, alpha=0.6)
            
        # X-axis configuration
        ax.set_xticks(x_ticks_vals)
        t_labels = ax.set_xticklabels([str(v) for v in x_ticks_vals], fontsize=6.5, fontname='Arial')
        ax.set_xlabel("Time (ms)", fontsize=8.0, fontname='Arial', labelpad=2)
        ax.tick_params(axis='x', which='major', length=2.5, width=0.7, pad=2)
        
        # Prevent edge overlap between adjacent panels and internally
        if idx == 0:
            # BCIAUT: avoid -100 and 0 collision
            t_labels[0].set_ha('right')
            t_labels[1].set_ha('left')
            t_labels[-1].set_ha('right')
        elif idx in [1, 2]:
            t_labels[0].set_ha('left')
            t_labels[-1].set_ha('right')
        elif idx == 3:
            # GIB Other: tuck 1000 rightwards to avoid colliding with GIST -200
            t_labels[0].set_ha('left')
            t_labels[-1].set_ha('right')
        elif idx == 4:
            # GIST: tuck -200 right-aligned/inward, tuck 0 rightwards
            t_labels[0].set_ha('left')
            t_labels[1].set_ha('left')
            t_labels[-1].set_ha('right')

        # Y-axis configuration: add 1, 4, 8 to tell there are 8 channels
        if idx == 0:
            ax.set_yticks([1, 4, 8])
            ax.set_yticklabels(['1', '4', '8'], fontsize=7.5, fontname='Arial')
            ax.set_ylabel("Channel", fontsize=8.5, fontname='Arial', labelpad=2)
            ax.tick_params(axis='y', which='major', length=2.5, width=0.7, pad=2)
        else:
            ax.set_yticks([1, 4, 8])
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', left=False)
            
        ax.set_title(title, fontsize=8.5, fontname='Arial', fontweight='bold', pad=4)

    # Spacing and layout: shrink each panel box so maps are landscape rectangles
    plt.subplots_adjust(left=0.09, right=0.88, bottom=0.30, top=0.70, wspace=0.30)

    # Shared Colorbar on the right
    cbar_ax = fig.add_axes([0.90, 0.30, 0.012, 0.40])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=6.5, length=2, width=0.7, pad=1)
    cbar.set_label("Saliency", fontsize=8.0, fontname='Arial', labelpad=3)

    save_pdf = os.path.join(config.PLOT_DIR_COMBINED, "combined_saliency_row.pdf")
    save_png = os.path.join(config.PLOT_DIR_COMBINED, "combined_saliency_row.png")
    plt.savefig(save_pdf, dpi=350, bbox_inches='tight')
    plt.savefig(save_png, dpi=350, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

if __name__ == "__main__":
    generate_plot()
