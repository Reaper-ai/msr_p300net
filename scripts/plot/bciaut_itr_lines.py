import os
import pandas as pd
import matplotlib.pyplot as plt

import config
from bciaut import load_itr_data

def generate_plot():
    df_ours = load_itr_data("MSR_P300NET (Ours)", "msr_p300net")
    if df_ours is None:
        df_ours = load_itr_data("MSR_P300NET (Ours)", "normal_hst")
    df_eegnet = load_itr_data("EEGNetv4", "eegnet")
    df_p3cnet = load_itr_data("P3CNET", "p3cnet")
    
    if df_ours is None or df_eegnet is None or df_p3cnet is None:
        print("Warning: Missing BCIAUT ITR results. Skipping ITR Lines.")
        return
        
    df_all = pd.concat([df_ours, df_eegnet, df_p3cnet], ignore_index=True)
    
    # Plot 1: Accuracy vs Sequences
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['Sequences'], df_sub['Accuracy (%)'], marker='^', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=2.5, markersize=8)
    config.apply_theme(ax)
    ax.set_xlabel("Sequence Repetitions", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    ax.set_ylabel("Spelling Accuracy (%)", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=True, facecolor='white', edgecolor='none', prop={'size': config.LEGEND_SIZE, 'family': 'Times New Roman'})
    plt.tight_layout()
    save_pdf = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_itr_accuracy.pdf")
    save_png = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_itr_accuracy.png")
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")
    
    # Plot 2: Bits/Selection vs Sequences
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_name in df_all['Model'].unique():
        df_sub = df_all[df_all['Model'] == model_name]
        ax.plot(df_sub['Sequences'], df_sub['Bits/Selection'], marker='^', label=model_name,
                color=config.MODEL_COLORS[model_name], linewidth=2.5, markersize=8)
    config.apply_theme(ax)
    ax.set_xlabel("Sequence Repetitions", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    ax.set_ylabel("ITR (bits/selection)", fontsize=config.LABEL_SIZE, fontname='Times New Roman')
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=True, facecolor='white', edgecolor='none', prop={'size': config.LEGEND_SIZE, 'family': 'Times New Roman'})
    plt.tight_layout()
    save_pdf = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_itr_bitrate.pdf")
    save_png = os.path.join(config.PLOT_DIR_BCIAUT, "bciaut_itr_bitrate.png")
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_pdf} and {save_png}")

if __name__ == "__main__":
    generate_plot()
